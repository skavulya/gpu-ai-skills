# Runtime Caveats

Use this reference when choosing dtype/KV options, explaining VLM or
diffusion behavior, or deciding whether the user needs a benchmark
instead of a fit estimate.

## Quick Arc Pro B70 Reference

Regenerate this table for the current script and model set:

```sh
python3 plugins/intel-gpu-ai-skills/skills/model-can-it-fit/scripts/fit.py \
    --table --runtime vllm --device-vram-gb 32
```

`--device-vram-gb 32` is deliberate here: this table is pinned to the B70
so it regenerates identically from any host. Do not swap in the VRAM of
whatever card the regenerating machine happens to have -- that silently
retargets the table. For the user's actual hardware, run the script
per-model with a measured value instead (see the skill's "Measure VRAM
First").

Typical Arc Pro B70 (32 GB) planning outcomes:

| Model | bf16 / 8K / c=1 | bf16 / 4K / c=4 | int4 / 32K / c=4 |
|---|---|---|---|
| Qwen2.5-1.5B-Instruct | fits | fits | fits |
| Qwen2.5-7B-Instruct | fits | fits | fits |
| Llama-3.1-8B-Instruct | fits | fits | fits |
| Qwen2.5-14B-Instruct | fits or tight | tight | fits |
| Qwen2.5-32B-Instruct | OOM from weights | OOM from weights | fits |
| GPT-OSS-120B | OOM from weights | OOM from weights | needs multi-XPU TP |

Treat this table as orientation only. Use the script for the user's
actual context, concurrency, runtime, and memory-utilization target.

## Dtype And KV Choices

`bf16` is the safest default on XPU.

For vLLM launch planning, set `--gpu-memory-utilization` to the same
value as the planned `vllm serve` command. The script default of `1.0`
means physical-fit only.

For `fp8`, pair with `--kv-cache-dtype fp8` when serving. The vLLM launch may
also need an attention backend that supports fp8 KV on XPU.

For `int4` AWQ, GPTQ, or AutoRound models, the script auto-pairs KV with
`fp8`. Validate live runtime logs for the expected int4 kernel path; if
the runtime falls back to a wider activation path, activation memory and
throughput may differ from the estimate.

`int3` and `int2` are AutoRound-oriented planning modes. Validate output
quality before trusting a deployment based on those sizes.

## VLM Caveats

The script includes the vision tower weights when `vision_config`,
vision-language architecture names, or VLM model types are present.

It does not fully model:

- image-token KV growth from resolution and image count
- vision encoder activation peaks
- processor-side memory spikes

For tight VLM fits, leave extra headroom. As a planning rule, add about
1-3 GiB per concurrent image-bearing request, then verify with
`torch-xpu-bench` at the user's image resolution.

## Diffusion Caveats

The script refuses diffusion pipelines when the root has
`model_index.json` but no top-level LLM config. Diffusion peak memory is
dominated by latent resolution, steps, scheduler, and pipeline component
activation peaks, so a config-only estimate is not reliable.

For a floor estimate, point the script at a component config such as the
UNet or transformer subdirectory. For the real answer, use
`torch-xpu-bench` with one run and read peak XPU memory.

## Hybrid And Sliding-Window Caveats

For a model with recurrent (state-space) layers, the state pool is charged
per running request and does not shrink with context. Lowering `--ctx`
therefore does little for a hybrid model whose state pool dominates;
lowering `--concurrency` is the lever that works.

The state pool is sensitive to `--tp`. When the state heads do not divide
evenly by the tensor-parallel degree, the engine pads them out to a
multiple of it, so a shard can cost more than `total / tp`. The script
models that padding, which is why TP=3 on a 16-head model can look worse
than expected.

For a sliding-window model, the per-layer cache is bounded, so the KV
curve flattens once the context passes the window. Raising `--ctx` past
that point changes almost nothing, and the reported max context is found
by search rather than by division.

Switching `--runtime` changes both of these, because the engines size
their pools differently. SGLang provisions its window pool as a fraction
of the full-attention pool rather than from the window, and forces the SSM
half of recurrent state to fp32. The same model can fit under vLLM and not
under SGLang.

## What This Skill Does Not Predict

- measured tokens/sec, TTFT, TPOT, or ITL
- runtime graph-capture or `torch.compile` buffers
- prefix-cache buffers when prefix caching is enabled
- pipeline-parallel sharding
- diffusion peak memory
- correctness or output quality after aggressive quantization
- block-quantized and group-padded weight layouts. The bytes-per-parameter
  table folds in a typical scale and zero-point overhead for group size
  128. A checkpoint with a smaller group, per-block fp8 scales, or padded
  output dimensions carries a little more than the estimate.
- quantization recipes stated only in a sidecar file. NVIDIA modelopt
  checkpoints put the recipe in `hf_quant_config.json`, which the script
  does not read; pass `--quant` explicitly for those.
- replicated projector weights on a sharded vision encoder. A multimodal
  projector or merger that is not tensor-parallel is counted as sharded,
  which understates the per-device total by up to about 0.16 GB.
- decode-time bandwidth for hybrid models. A recurrent layer reads a fixed
  state instead of a growing cache, so its decode cost does not scale with
  context the way an attention layer's does. That is a throughput question,
  not a memory one; use a benchmark skill.
- SGLang's own recurrent-pool sizing. SGLang derives the number of state
  slots from a memory fraction rather than from the requested concurrency,
  so its actual pool can be larger than the per-request figure here.

Use benchmark/profile skills for measured runtime behavior.

## External References

- apxml VRAM calculator, useful CUDA analogue: <https://apxml.com/tools/vram-calculator>
- vLLM XPU kernels: <https://github.com/vllm-project/vllm-xpu-kernels>
- Intel AutoRound: <https://github.com/intel/auto-round>
