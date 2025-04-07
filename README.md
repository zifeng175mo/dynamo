<!--
SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

# NVIDIA Dynamo

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![GitHub Release](https://img.shields.io/github/v/release/ai-dynamo/dynamo)](https://github.com/ai-dynamo/dynamo/releases/latest)
[![Discord](https://dcbadge.limes.pink/api/server/D92uqZRjCZ?style=flat)](https://discord.gg/nvidia-dynamo)

| **[Support Matrix](support_matrix.md)** | **[Guides](docs/guides)** | **[Architecture and Features](docs/architecture.md)** | **[APIs](lib/bindings/python/README.md)** | **[SDK](deploy/dynamo/sdk/README.md)** |

NVIDIA Dynamo is a high-throughput low-latency inference framework designed for serving generative AI and reasoning models in multi-node distributed environments. Dynamo is designed to be inference engine agnostic (supports TRT-LLM, vLLM, SGLang or others) and captures LLM-specific capabilities such as:

- **Disaggregated prefill & decode inference** – Maximizes GPU throughput and facilitates trade off between throughput and latency.
- **Dynamic GPU scheduling** – Optimizes performance based on fluctuating demand
- **LLM-aware request routing** – Eliminates unnecessary KV cache re-computation
- **Accelerated data transfer** – Reduces inference response time using NIXL.
- **KV cache offloading** – Leverages multiple memory hierarchies for higher system throughput

Built in Rust for performance and in Python for extensibility, Dynamo is fully open-source and driven by a transparent, OSS (Open Source Software) first development approach.

### Installation

The following examples require a few system level packages.
Recommended to use Ubuntu 24.04 with a x86_64 CPU. See [support_matrix.md](support_matrix.md)

```
apt-get update
DEBIAN_FRONTEND=noninteractive apt-get install -yq python3-dev python3-pip python3-venv libucx0
python3 -m venv venv
source venv/bin/activate

pip install ai-dynamo[all]
```

> [!NOTE]
> TensorRT-LLM Support is currently available on a [branch](https://github.com/ai-dynamo/dynamo/tree/dynamo/trtllm_llmapi_v1/examples/trtllm#building-the-environment)

### Development Environment

For a consistent development environment, you can use the provided devcontainer configuration. This requires:
- [Docker](https://www.docker.com/products/docker-desktop)
- [VS Code](https://code.visualstudio.com/) with the [Dev Containers extension](https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.remote-containers)

To use the devcontainer:
1. Open the project in VS Code
2. Click on the button in the bottom-left corner
3. Select "Reopen in Container"

This will build and start a container with all the necessary dependencies for Dynamo development.

### Running and Interacting with an LLM Locally

To run a model and interact with it locally you can call `dynamo
run` with a hugging face model. `dynamo run` supports several backends
including: `mistralrs`, `sglang`, `vllm`, and `tensorrtllm`.

#### Example Command

```
dynamo run out=vllm deepseek-ai/DeepSeek-R1-Distill-Llama-8B
```

```
? User › Hello, how are you?
✔ User · Hello, how are you?
Okay, so I'm trying to figure out how to respond to the user's greeting. They said, "Hello, how are you?" and then followed it with "Hello! I'm just a program, but thanks for asking." Hmm, I need to come up with a suitable reply. ...
```

### LLM Serving

Dynamo provides a simple way to spin up a local set of inference
components including:

- **OpenAI Compatible Frontend** – High performance OpenAI compatible http api server written in Rust.
- **Basic and Kv Aware Router** – Route and load balance traffic to a set of workers.
- **Workers** – Set of pre-configured LLM serving engines.

To run a minimal configuration you can use a pre-configured
example.

#### Start Dynamo Distributed Runtime Services

First start the Dynamo Distributed Runtime services:

```bash
docker compose -f deploy/docker-compose.yml up -d
```

#### Start Dynamo LLM Serving Components

Next serve a minimal configuration with an http server, basic
round-robin router, and a single worker.

```bash
cd examples/llm
dynamo serve graphs.agg:Frontend -f configs/agg.yaml
```

#### Send a Request

```bash
curl localhost:8000/v1/chat/completions   -H "Content-Type: application/json"   -d '{
    "model": "deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
    "messages": [
    {
        "role": "user",
        "content": "Hello, how are you?"
    }
    ],
    "stream":false,
    "max_tokens": 300
  }' | jq
```

### Local Development

To develop locally, we recommend working inside of the container

```bash
./container/build.sh
./container/run.sh -it --mount-workspace

cargo build --release
mkdir -p /workspace/deploy/dynamo/sdk/src/dynamo/sdk/cli/bin
cp /workspace/target/release/http /workspace/deploy/dynamo/sdk/src/dynamo/sdk/cli/bin
cp /workspace/target/release/llmctl /workspace/deploy/dynamo/sdk/src/dynamo/sdk/cli/bin
cp /workspace/target/release/dynamo-run /workspace/deploy/dynamo/sdk/src/dynamo/sdk/cli/bin

uv pip install -e .
```
