// SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

use anyhow::Result;
use dynamo_runtime::{
    component::Component,
    pipeline::{
        async_trait, AsyncEngine, AsyncEngineContextProvider, Error, ManyOut, ResponseStream,
        SingleIn,
    },
    prelude::*,
    protocols::annotated::Annotated,
};
use futures::stream::{self, StreamExt};
use std::sync::Arc;

pub mod indexer;
pub mod metrics_aggregator;
pub mod protocols;
pub mod publisher;
pub mod recorder;
pub mod scheduler;
pub mod scoring;

use crate::{
    kv_router::{
        indexer::{KvIndexer, KvIndexerInterface, RouterEvent},
        metrics_aggregator::KvMetricsAggregator,
        protocols::{LocalBlockHash, RouterRequest, RouterResponse, WorkerSelectionResult},
        scheduler::{KvScheduler, KvSchedulerError, SchedulingRequest},
        scoring::ProcessedEndpoints,
    },
    tokens::Tokens,
};

use dynamo_runtime::traits::events::EventSubscriber;

// [gluo TODO] shouldn't need to be public
// this should be discovered from the component
pub const KV_EVENT_SUBJECT: &str = "kv_events";
pub const KV_HIT_RATE_SUBJECT: &str = "kv-hit-rate";
pub const KV_METRICS_ENDPOINT: &str = "load_metrics";

/// A trait that users can implement to define custom selection logic
pub trait WorkerSelector {
    fn select_worker(
        &self,
        workers: &ProcessedEndpoints,
        request: &SchedulingRequest,
        block_size: usize,
    ) -> Result<WorkerSelectionResult, KvSchedulerError>;
}

pub struct KvRouter {
    indexer: KvIndexer,
    scheduler: KvScheduler,
    block_size: usize,
}

impl KvRouter {
    pub async fn new(
        component: Component,
        block_size: usize,
        selector: Option<Box<dyn WorkerSelector + Send + Sync>>,
    ) -> Result<Arc<Self>> {
        let cancellation_token = component
            .drt()
            .primary_lease()
            .expect("Cannot KV route static workers")
            .primary_token();

        let metrics_aggregator =
            KvMetricsAggregator::new(component.clone(), cancellation_token.clone()).await;
        let indexer = KvIndexer::new(cancellation_token.clone(), block_size);
        let scheduler = KvScheduler::start(
            component.namespace().clone(),
            block_size,
            metrics_aggregator.endpoints_watcher(),
            selector,
        )
        .await?;

        // [gluo TODO] try subscribe_with_type::<RouterEvent>,
        // error checking below will be different.
        let mut kv_events_rx = component.subscribe(KV_EVENT_SUBJECT).await?;
        let kv_events_tx = indexer.event_sender();

        tokio::spawn(async move {
            while let Some(event) = kv_events_rx.next().await {
                let event: RouterEvent = match serde_json::from_slice(&event.payload) {
                    Ok(event) => {
                        tracing::debug!("received kv event: {:?}", event);
                        event
                    }
                    Err(e) => {
                        tracing::warn!("Failed to deserialize RouterEvent: {:?}", e);
                        // Choosing warn and continue to process other events from other workers
                        // A bad event likely signals a problem with a worker, but potentially other workers are still healthy
                        continue;
                    }
                };
                if let Err(e) = kv_events_tx.send(event).await {
                    tracing::trace!("failed to send kv event to indexer; shutting down: {:?}", e);
                }
            }
        });

        Ok(Arc::new(Self {
            scheduler,
            indexer,
            block_size,
        }))
    }

    // [TODO] indexer needs to take 'lora_id' as parameter
    pub async fn schedule(&self, token_ids: &Vec<u32>, _lora_id: u64) -> Result<i64> {
        // Extracting part of the code in KvRouter::generate() for only
        // the decision making part, routing is done by the caller
        let isl_tokens = token_ids.len();
        let overlap_scores = self
            .indexer
            .find_matches_for_request(token_ids.as_slice())
            .await?;
        tracing::debug!("KV router overlap_scores: {:?}", overlap_scores);
        let worker_id = self.scheduler.schedule(overlap_scores, isl_tokens).await?;
        Ok(worker_id)
    }
}

#[async_trait]
impl AsyncEngine<SingleIn<RouterRequest>, ManyOut<Annotated<RouterResponse>>, Error> for KvRouter {
    async fn generate(
        &self,
        request: SingleIn<RouterRequest>,
    ) -> Result<ManyOut<Annotated<RouterResponse>>> {
        let (request, ctx) = request.into_parts();
        let isl_tokens = request.tokens.len();
        let block_size = self.block_size;

        // Compute the block hashes in a blocking task
        let local_block_hashes: Vec<LocalBlockHash> = tokio::task::spawn_blocking(move || {
            Tokens::compute_block_hash(&request.tokens, block_size)
                .into_iter()
                .map(LocalBlockHash)
                .collect()
        })
        .await?;

        let overlap_scores = self.indexer.find_matches(local_block_hashes).await?;
        let worker_id = self.scheduler.schedule(overlap_scores, isl_tokens).await?;

        let response = RouterResponse { worker_id };
        let response = Annotated::from_data(response);
        let stream = stream::iter(vec![response]);
        Ok(ResponseStream::new(Box::pin(stream), ctx.context()))
    }
}
