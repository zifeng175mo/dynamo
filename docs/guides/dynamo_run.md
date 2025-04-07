# Dynamo Run

`dynamo-run` is a CLI tool for exploring the Dynamo components, and an example of how to use them from Rust. It is also available as `dynamo run` if using the Python wheel.

## Quickstart with pip and vllm

If you used `pip` to install `dynamo` you should have the `dynamo-run` binary pre-installed with the `vllm` engine. You must be in a virtual env with vllm installed to use this. For more options see "Full documentation" below.

### Automatically download a model from [Hugging Face](https://huggingface.co/models)

This will automatically download Qwen2.5 3B from Hugging Face (6 GiB download) and start it in interactive text mode:
```
dynamo run out=vllm Qwen/Qwen2.5-3B-Instruct
```

General format for HF download:
```
dynamo run out=<engine> <HUGGING_FACE_ORGANIZATION/MODEL_NAME>
```

For gated models (e.g. meta-llama/Llama-3.2-3B-Instruct) you have to have an `HF_TOKEN` environment variable set.

The parameter can be the ID of a HuggingFace repository (it will be downloaded), a GGUF file, or a folder containing safetensors, config.json, etc (a locally checked out HuggingFace repository).

## Manually download a model from Hugging Face

One of these models should be high quality and fast on almost any machine: https://huggingface.co/bartowski/Llama-3.2-3B-Instruct-GGUF
E.g. https://huggingface.co/bartowski/Llama-3.2-3B-Instruct-GGUF/blob/main/Llama-3.2-3B-Instruct-Q4_K_M.gguf

Download model file:
```
curl -L -o Llama-3.2-3B-Instruct-Q4_K_M.gguf "https://huggingface.co/bartowski/Llama-3.2-3B-Instruct-GGUF/resolve/main/Llama-3.2-3B-Instruct-Q4_K_M.gguf?download=true"
```

## Run a model from local file

*Text interface*
```
dynamo run out=vllm Llama-3.2-3B-Instruct-Q4_K_M.gguf # or path to a Hugging Face repo checkout instead of the GGUF
```

*HTTP interface*
```
dynamo run in=http out=vllm Llama-3.2-3B-Instruct-Q4_K_M.gguf
```

*List the models*
```
curl localhost:8080/v1/models
```

*Send a request*
```
curl -d '{"model": "Llama-3.2-3B-Instruct-Q4_K_M", "max_completion_tokens": 2049, "messages":[{"role":"user", "content": "What is the capital of South Africa?" }]}' -H 'Content-Type: application/json' http://localhost:8080/v1/chat/completions
```

*Multi-node*

You will need [etcd](https://etcd.io/) and [nats](https://nats.io) installed and accessible from both nodes.

Node 1:
```
dynamo run in=http out=dyn://llama3B_pool
```

Node 2:
```
dynamo run in=dyn://llama3B_pool out=vllm ~/llm_models/Llama-3.2-3B-Instruct
```

This will use etcd to auto-discover the model and NATS to talk to it. You can run multiple workers on the same endpoint and it will pick one at random each time.

The `llama3B_pool` name is purely symbolic, pick anything as long as it matches the other node.

Run `dynamo run --help` for more options.

# Full documentation

`dynamo-run` is what `dynamo run` executes. It is an example of what you can build in Rust with the `dynamo-llm` and `dynamo-runtime`. Here is a list of how to build from source and all the features.

## Setup

Libraries Ubuntu:
```
apt install -y build-essential libhwloc-dev libudev-dev pkg-config libssl-dev libclang-dev protobuf-compiler python3-dev
```

Libraries macOS:
- [Homebrew](https://brew.sh/)
```
# if brew is not installed on your system, install it
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
```
- [Xcode](https://developer.apple.com/xcode/)

```
brew install cmake protobuf

# Check that Metal is accessible
xcrun -sdk macosx metal
```
If Metal is accessible, you should see an error like `metal: error: no input files`, which confirms it is installed correctly.

Install Rust:
```
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
source $HOME/.cargo/env
```

## Build

Navigate to launch/ directory
```
cd launch/
```
Optionally can run `cargo build` from any location with arguments:
```
--target-dir /path/to/target_directory` specify target_directory with write privileges
--manifest-path /path/to/project/Cargo.toml` if cargo build is run outside of `launch/` directory
```

- Linux with GPU and CUDA (tested on Ubuntu):
```
cargo build --features cuda
```

- macOS with Metal:
```
cargo build --features metal
```

- CPU only:
```
cargo build
```

The binary will be called `dynamo-run` in `target/debug`
```
cd target/debug
```

Build with `--release` for a smaller binary and better performance, but longer build times. The binary will be in `target/release`.

## sglang

1. Setup the python virtual env:

```
uv venv
source .venv/bin/activate
uv pip install pip
uv pip install sgl-kernel --force-reinstall --no-deps
uv pip install "sglang[all]==0.4.2" --find-links https://flashinfer.ai/whl/cu124/torch2.4/flashinfer/
```

2. Build

```
cargo build --features sglang
```

3. Run

Any example above using `out=sglang` will work, but our sglang backend is also multi-gpu and multi-node.

Node 1:
```
dynamo-run in=http out=sglang --model-path ~/llm_models/DeepSeek-R1-Distill-Llama-70B/ --tensor-parallel-size 8 --num-nodes 2 --node-rank 0 --leader-addr 10.217.98.122:9876
```

Node 2:
```
dynamo-run in=none out=sglang --model-path ~/llm_models/DeepSeek-R1-Distill-Llama-70B/ --tensor-parallel-size 8 --num-nodes 2 --node-rank 1 --leader-addr 10.217.98.122:9876
```

To pass extra arguments to the sglang engine see *Extra engine arguments* below.

## llama_cpp

- `cargo build --features llamacpp,cuda`

- `dynamo-run out=llama_cp ~/llm_models/Llama-3.2-3B-Instruct-Q6_K.gguf`

If the build step also builds llama_cpp libraries into the same folder as the binary ("libllama.so", "libggml.so", "libggml-base.so", "libggml-cpu.so", "libggml-cuda.so"), then `dynamo-run` will need to find those at runtime. Set `LD_LIBRARY_PATH`, and be sure to deploy them alongside the `dynamo-run` binary.

## vllm

Using the [vllm](https://github.com/vllm-project/vllm) Python library. We only use the back half of vllm, talking to it over `zmq`. Slow startup, fast inference. Supports both safetensors from HF and GGUF files.

We use [uv](https://docs.astral.sh/uv/) but any virtualenv manager should work.

Setup:
```
uv venv
source .venv/bin/activate
uv pip install pip
uv pip install vllm==0.7.3 setuptools
```

**Note: If you're on Ubuntu 22.04 or earlier, you will need to add `--python=python3.10` to your `uv venv` command**

Build:
```
cargo build --features vllm
```

Run (still inside that virtualenv) - HF repo:
```
./dynamo-run in=http out=vllm --model-path ~/llm_models/Llama-3.2-3B-Instruct/

```

Run (still inside that virtualenv) - GGUF:
```
./dynamo-run in=http out=vllm ~/llm_models/Llama-3.2-3B-Instruct-Q6_K.gguf
```

+ Multi-node:

Node 1:
```
dynamo-run in=text out=vllm ~/llm_models/Llama-3.2-3B-Instruct/ --tensor-parallel-size 8 --num-nodes 2 --leader-addr 10.217.98.122:6539 --node-rank 0
```

Node 2:
```
dynamo-run in=none out=vllm ~/llm_models/Llama-3.2-3B-Instruct/ --num-nodes 2 --leader-addr 10.217.98.122:6539 --node-rank 1
```

To pass extra arguments to the vllm engine see *Extra engine arguments* below.

## Python bring-your-own-engine

You can provide your own engine in a Python file. The file must provide a generator with this signature:
```
async def generate(request):
```

Build: `cargo build --features python`

### Python does the pre-processing

If the Python engine wants to receive and returns strings - it will do the prompt templating and tokenization itself - run it like this:

```
dynamo-run out=pystr:/home/user/my_python_engine.py
```

- The `request` parameter is a map, an OpenAI compatible create chat completion request: https://platform.openai.com/docs/api-reference/chat/create
- The function must `yield` a series of maps conforming to create chat completion stream response (example below).
- If using an HTTP front-end add the `--model-name` flag. This is the name we serve the model under.

The file is loaded once at startup and kept in memory.

Example engine:
```
import asyncio

async def generate(request):
    yield {"id":"1","choices":[{"index":0,"delta":{"content":"The","role":"assistant"}}],"created":1841762283,"model":"Llama-3.2-3B-Instruct","system_fingerprint":"local","object":"chat.completion.chunk"}
    await asyncio.sleep(0.1)
    yield {"id":"1","choices":[{"index":0,"delta":{"content":" capital","role":"assistant"}}],"created":1841762283,"model":"Llama-3.2-3B-Instruct","system_fingerprint":"local","object":"chat.completion.chunk"}
    await asyncio.sleep(0.1)
    yield {"id":"1","choices":[{"index":0,"delta":{"content":" of","role":"assistant"}}],"created":1841762283,"model":"Llama-3.2-3B-Instruct","system_fingerprint":"local","object":"chat.completion.chunk"}
    await asyncio.sleep(0.1)
    yield {"id":"1","choices":[{"index":0,"delta":{"content":" France","role":"assistant"}}],"created":1841762283,"model":"Llama-3.2-3B-Instruct","system_fingerprint":"local","object":"chat.completion.chunk"}
    await asyncio.sleep(0.1)
    yield {"id":"1","choices":[{"index":0,"delta":{"content":" is","role":"assistant"}}],"created":1841762283,"model":"Llama-3.2-3B-Instruct","system_fingerprint":"local","object":"chat.completion.chunk"}
    await asyncio.sleep(0.1)
    yield {"id":"1","choices":[{"index":0,"delta":{"content":" Paris","role":"assistant"}}],"created":1841762283,"model":"Llama-3.2-3B-Instruct","system_fingerprint":"local","object":"chat.completion.chunk"}
    await asyncio.sleep(0.1)
    yield {"id":"1","choices":[{"index":0,"delta":{"content":".","role":"assistant"}}],"created":1841762283,"model":"Llama-3.2-3B-Instruct","system_fingerprint":"local","object":"chat.completion.chunk"}
    await asyncio.sleep(0.1)
    yield {"id":"1","choices":[{"index":0,"delta":{"content":"","role":"assistant"},"finish_reason":"stop"}],"created":1841762283,"model":"Llama-3.2-3B-Instruct","system_fingerprint":"local","object":"chat.completion.chunk"}
```

Command line arguments are passed to the python engine like this:
```
dynamo-run out=pystr:my_python_engine.py -- -n 42 --custom-arg Orange --yes
```

The python engine receives the arguments in `sys.argv`. The argument list will include some standard ones as well as anything after the `--`.

This input:
```
dynamo-run out=pystr:my_engine.py /opt/models/Llama-3.2-3B-Instruct/ --model-name llama_3.2 --tensor-parallel-size 4 -- -n 1
```

is read like this:
```
async def generate(request):
    .. as before ..

if __name__ == "__main__":
    print(f"MAIN: {sys.argv}")
```

and produces this output:
```
MAIN: ['my_engine.py', '--model-path', '/opt/models/Llama-3.2-3B-Instruct/', '--model-name', 'llama3.2', '--http-port', '8080', '--tensor-parallel-size', '4', '--base-gpu-id', '0', '--num-nodes', '1', '--node-rank', '0', '-n', '1']
```

This allows quick iteration on the engine setup. Note how the `-n` `1` is included. Flags `--leader-addr` and `--model-config` will also be added if provided to `dynamo-run`.

### Dynamo does the pre-processing

If the Python engine wants to receive and return tokens - the prompt templating and tokenization is already done - run it like this:
```
dynamo-run out=pytok:/home/user/my_python_engine.py --model-path <hf-repo-checkout>
```

- The request parameter is a map that looks like this:
```
{'token_ids': [128000, 128006, 9125, 128007, ... lots more ... ], 'stop_conditions': {'max_tokens': 8192, 'stop': None, 'stop_token_ids_hidden': [128001, 128008, 128009], 'min_tokens': None, 'ignore_eos': None}, 'sampling_options': {'n': None, 'best_of': None, 'presence_penalty': None, 'frequency_penalty': None, 'repetition_penalty': None, 'temperature': None, 'top_p': None, 'top_k': None, 'min_p': None, 'use_beam_search': None, 'length_penalty': None, 'seed': None}, 'eos_token_ids': [128001, 128008, 128009], 'mdc_sum': 'f1cd44546fdcbd664189863b7daece0f139a962b89778469e4cffc9be58ccc88', 'annotations': []}
```

- The `generate` function must `yield` a series of maps that look like this:
```
{"token_ids":[791],"tokens":None,"text":None,"cum_log_probs":None,"log_probs":None,"finish_reason":None}
```

- Command like flag `--model-path` which must point to a Hugging Face repo checkout containing the `tokenizer.json`. The `--model-name` flag is optional. If not provided we use the HF repo name (directory name) as the model name.

Example engine:
```
import asyncio

async def generate(request):
    yield {"token_ids":[791]}
    await asyncio.sleep(0.1)
    yield {"token_ids":[6864]}
    await asyncio.sleep(0.1)
    yield {"token_ids":[315]}
    await asyncio.sleep(0.1)
    yield {"token_ids":[9822]}
    await asyncio.sleep(0.1)
    yield {"token_ids":[374]}
    await asyncio.sleep(0.1)
    yield {"token_ids":[12366]}
    await asyncio.sleep(0.1)
    yield {"token_ids":[13]}
```

`pytok` supports the same ways of passing command line arguments as `pystr` - `initialize` or `main` with `sys.argv`.

## trtllm

TensorRT-LLM. Requires `clang` and `libclang-dev`.

Build:
```
cargo build --features trtllm
```

Run:
```
dynamo-run in=text out=trtllm --model-path /app/trtllm_engine/ --model-config ~/llm_models/Llama-3.2-3B-Instruct/
```

Note that TRT-LLM uses it's own `.engine` format for weights.

The `--model-path` you give to `dynamo-run` must contain the `config.json` (TRT-LLM's , not the model's) and `rank0.engine` (plus other ranks if relevant).

## Echo Engines

Dynamo includes two echo engines for testing and debugging purposes:

### echo_core

The `echo_core` engine accepts pre-processed requests and echoes the tokens back as the response. This is useful for testing pre-processing functionality as the response will include the full prompt template.

```
dynamo-run in=http out=echo_core --model-path <hf-repo-checkout>
```

### echo_full

The `echo_full` engine accepts un-processed requests and echoes the prompt back as the response.

```
dynamo-run in=http out=echo_full --model-name my_model
```

### Configuration

Both echo engines use a configurable delay between tokens to simulate generation speed. You can adjust this using the `DYN_TOKEN_ECHO_DELAY_MS` environment variable:

```
# Set token echo delay to 1ms (1000 tokens per second)
DYN_TOKEN_ECHO_DELAY_MS=1 dynamo-run in=http out=echo_full
```

The default delay is 10ms, which produces approximately 100 tokens per second.

## Batch mode

dynamo-run can take a jsonl file full of prompts and evaluate them all:

```
dynamo-run in=batch:prompts.jsonl out=llamacpp <model>
```

The input file should look like this:
```
{"text": "What is the capital of France?"}
{"text": "What is the capital of Spain?"}
```

Each one is passed as a prompt to the model. The output is written back to the same folder in `output.jsonl`. At the end of the run some statistics are printed.
The output looks like this:
```
{"text":"What is the capital of France?","response":"The capital of France is Paris.","tokens_in":7,"tokens_out":7,"elapsed_ms":1566}
{"text":"What is the capital of Spain?","response":".The capital of Spain is Madrid.","tokens_in":7,"tokens_out":7,"elapsed_ms":855}
```

## Defaults

The input defaults to `in=text`. The output will default to `mistralrs` engine. If not available whatever engine you have compiled in (so depending on `--features`).

## Extra engine arguments

The vllm and sglang backends support passing any argument the engine accepts.

Put the arguments in a JSON file:
```
{
    "dtype": "half",
    "trust_remote_code": true
}
```

Pass it like this:
```
dynamo-run out=sglang ~/llm_models/Llama-3.2-3B-Instruct --extra-engine-args sglang_extra.json
```

