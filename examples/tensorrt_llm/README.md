<!--
SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
-->

# LLM Deployment Examples using TensorRT-LLM

This directory contains examples and reference implementations for deploying Large Language Models (LLMs) in various configurations using TensorRT-LLM.


## Deployment Architectures

See [deployment architectures](../llm/README.md#deployment-architectures) to learn about the general idea of the architecture.
Note that this TensorRT-LLM version does not support all the options yet.

### Prerequisites

Start required services (etcd and NATS) using [Docker Compose](../../deploy/docker-compose.yml)
```bash
docker compose -f deploy/docker-compose.yml up -d
```

### Build docker

#### Step 1: Build TensorRT-LLM base container image

Because of the known issue of C++11 ABI compatibility within the NGC pytorch container, we rebuild TensorRT-LLM from source.
See [here](https://nvidia.github.io/TensorRT-LLM/installation/linux.html) for more informantion.

Use the helper script to build a TensorRT-LLM container base image. The script uses a specific commit id from TensorRT-LLM main branch.

```bash
./container/build_trtllm_base_image.sh
```

For more information see [here](https://nvidia.github.io/TensorRT-LLM/installation/build-from-source-linux.html#option-1-build-tensorrt-llm-in-one-step) for more details on building from source.
If you already have a TensorRT-LLM container image, you can skip this step.

#### Step 2: Build the Dynamo container

```
./container/build.sh --framework tensorrtllm
```

This build script internally points to the base container image built with step 1. If you skipped previous step because you already have the container image available, you can run the build script with that image as a base.

```bash
# Build dynamo image with other TRTLLM base image.
./container/build.sh --framework TENSORRTLLM --base-image <trtllm-base-image> --base-image-tag <trtllm-base-image-tag>
```

### Run container

```
./container/run.sh --framework tensorrtllm -it
```
## Run Deployment

### Example architectures

#### Aggregated serving
```bash
cd /workspace/examples/tensorrt_llm
dynamo serve graphs.agg:Frontend -f ./configs/agg.yaml
```

#### Aggregated serving with KV Routing
```bash
cd /workspace/examples/tensorrt_llm
dynamo serve graphs.agg_router:Frontend -f ./configs/agg_router.yaml
```

#### Aggregated serving using Dynamo Run

```bash
cd /workspace/examples/tensorrt_llm
dynamo run out=pystr:./engines/agg_engine.py -- --engine_args ./configs/llm_api_config.yaml
```
The above command should load the model specified in `llm_api_config.yaml` and start accepting
text input from the client. For more details on the `dynamo run` command, please refer to the
[dynamo run](/docs/guides/dynamo_run.md#python-bring-your-own-engine) documentation.

Currently only aggregated deployment option is supported by `dynamo run` for TensorRT-LLM.
Adding support for disaggregated deployment is under development. This does *not* require
any other pre-requisites mentioned in the [Prerequisites](#prerequisites) section.


<!--
This is work in progress and will be enabled soon.

#### Disaggregated serving
```bash
cd /workspace/examples/llm
dynamo serve graphs.disagg:Frontend -f ./configs/disagg.yaml
```

#### Disaggregated serving with KV Routing
```bash
cd /workspace/examples/llm
dynamo serve graphs.disagg_router:Frontend -f ./configs/disagg_router.yaml
```
-->

### Client

See [client](../llm/README.md#client) section to learn how to send request to the deployment.

### Close deployment

See [close deployment](../llm/README.md#close-deployment) section to learn about how to close the deployment.

Remaining tasks:

- [ ] Add support for the disaggregated serving.
- [ ] Add integration test coverage.
- [ ] Add instructions for benchmarking.
- [ ] Add multi-node support.
- [ ] Merge the code base with llm example to reduce the code duplication.
- [ ] Use processor from dynamo-llm framework.
- [ ] Explore NIXL integration with TensorRT-LLM.
