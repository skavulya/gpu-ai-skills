# Coverage And Formulas

Use this reference when the user asks how the estimate is calculated,
why it differs from another calculator, or whether a specific model class
is covered.

## Coverage

| Model class | Behavior | Expected accuracy |
|---|---|---|
| Decoder-only LLMs such as Qwen, Llama, Mistral, and Gemma text models | Full estimate | About 5% |
| MoE models such as Qwen3-MoE, Mixtral, and DeepSeek-V3 | Full estimate, including shared experts | About 10% |
| Hybrid MoE models such as DeepSeek-V2/V3/V4 and Mistral-Large-3 | Counts dense and MoE layers separately via `first_k_dense_replace` | About 10% |
| Sliding-window models such as Mistral, Gemma-2/3, gpt-oss, and Llama-4 | Charges windowed layers a bounded cache, per runtime | About 10% |
| Hybrid state-space models such as Qwen3-Next, Nemotron-H, Bamba, Falcon-H1, and Jamba | Charges recurrent layers a state pool instead of a KV cache | About 10% |
| Pure state-space models such as Mamba and Mamba2 | State pool only, no KV cache | About 10% |
| VLMs such as Qwen2-VL, Gemma-3 vision, LLaVA, and Nemotron-Omni | Adds vision tower and supports `text_config` or `llm_config` | Weights are exact; runtime caveats apply |
| Mistral `params.json` models | Reads `params.json` when `config.json` is absent | About 10% |
| Diffusion models | Refuses as a full estimate | Use empirical benchmarking |

The script reads standard Hugging Face `config.json`, Mistral
`params.json`, or a local JSON path. It detects diffusion repos from
`model_index.json` and exits with a routing message.

## Core Formula

```text
VRAM = weights
     + kv_cache(ctx, concurrency)
     + recurrent_state(concurrency)
     + activations
     + framework
usable_vram = physical_vram * gpu_memory_utilization
fits = VRAM <= usable_vram
```

Weights are estimated from config dimensions:

```text
head_dim = config.head_dim or hidden_size / num_attention_heads
q_proj_dim = num_attention_heads * head_dim
kv_proj_dim = num_key_value_heads * head_dim

attention_per_layer =
    hidden * q_proj_dim
  + hidden * kv_proj_dim
  + hidden * kv_proj_dim
  + q_proj_dim * hidden

dense_ffn_per_layer = 3 * hidden * intermediate_size
moe_ffn_per_layer =
    num_experts * 3 * hidden * moe_intermediate_size
  + num_shared_experts * 3 * hidden * moe_intermediate_size

recurrent_per_layer =
    conv_dim * (conv_kernel - 1)                      # conv state
  + num_state_heads * state_head_dim_k * state_head_dim_v   # SSM state
```

The KV cache is charged per layer, not per model, because the three kinds
of layer cost different amounts:

```text
kv_per_token_per_layer =
  2 * num_key_value_heads * head_dim * bytes_per_kv_dtype

kv_cache = kv_per_token_per_layer * concurrency
  * ( num_full_attention_layers * ctx
    + num_sliding_layers * window_tokens )
```

`window_tokens` is where the runtimes visibly disagree, so it follows the
runtime rather than the model:

| Runtime | Tokens a windowed layer holds |
|---|---|
| `vllm` | `min(ctx, window - 1 + max_in_flight_tokens)` |
| `sglang` | `min(ctx, max(window, 0.8 * ctx))` |
| `torch` | `min(ctx, window)` |

vLLM pages the window and cannot free a block until every token in it has
left the window, so it holds `window - 1` plus the tokens still in flight
across the current scheduler batches. `max_in_flight_tokens` is twice
`max_num_batched_tokens`, which vLLM defaults from device memory: 8192 on
parts at or above 70 GiB, 2048 otherwise. So on Arc Pro hardware a
windowed layer holds `window - 1 + 4096` tokens.

SGLang does not size its window pool to the window at all. It provisions
a fraction of the full-attention pool (`--swa-full-tokens-ratio`, default
0.8) so the radix prefix cache still has blocks to reuse.

Recurrent (state-space) layers hold no KV cache. They hold a fixed state
per running request that does not shrink with context:

```text
recurrent_state = recurrent_per_layer * num_recurrent_layers * concurrency
```

The conv state holds `conv_kernel - 1` columns, not `conv_kernel`: the
current token is computed rather than cached. Neither engine quantizes
recurrent state. vLLM keeps it in the model dtype; SGLang forces the SSM
half to fp32, which is why the same model costs more state under SGLang.

Falcon-H1 and Zamba2 run attention and an SSM inside the *same* layer, so
every layer is charged both a full KV cache and a state.

The script includes embeddings and untied LM head when applicable. For
hybrid MoE models, dense replacement layers use dense FFN dimensions and
remaining layers use MoE expert dimensions. Recurrent layers are charged
their own projection and conv weights instead of Q/K/V/O projections.

Activation memory is a bounded estimate:

```text
activations ~= 2 * concurrency * ctx * hidden_size * bytes_per_param + 512 MiB
```

Runtime framework overhead is a floor estimate:

| Runtime | Overhead |
|---|---:|
| `vllm` | About 2.0 GiB |
| `sglang` | About 1.5 GiB |
| `torch` | About 0.8 GiB |

## Bytes Per Parameter

| Quant | Bytes per parameter |
|---|---:|
| `bf16` / `fp16` | 2.00 |
| `fp8` / `int8` | 1.00 |
| `int4` | 0.55 |
| `int3` | 0.42 |
| `int2` | 0.30 |
| `mxfp4` | 0.55 |

`int4` includes typical scale and zero overhead for grouped
quantization with group size 128. KV dtype bytes are 2 for `bf16` and
`fp16`, and 1 for `fp8` or `int8`.

## Important Modeling Details

Use explicit `head_dim` from config when present. Some models use a
larger head dimension than `hidden_size / num_attention_heads`; ignoring
that undercounts KV cache and Q/O projection parameters.

For MoE models, prefer `moe_intermediate_size`, `expert_hidden_dim`, or
the equivalent MoE-specific FFN field over dense `intermediate_size`.
Shared experts are always active in addition to routed experts.

Which layers use the sliding window has to come from the config or from a
known family. A bare `sliding_window` key is not enough on its own:
DeepSeek-V4-Flash carries `sliding_window: 128` for its sparse-attention
indexer while every layer still caches the full context. Honouring that
key would under-estimate its cache eightfold, and under-estimating is the
direction that reports FITS for a launch that then runs out of memory. So
a window is applied only when `layer_types`, `sliding_window_pattern`, or
a known family says which layers use it; otherwise every layer is charged
full attention.

Recurrent-state geometry is normalized across families into one shape:
`num_state_heads` states of `state_head_dim_k` by `state_head_dim_v`, plus
a conv state `conv_dim` wide. Gated-delta-net configs (Qwen3-Next) give
`linear_num_value_heads`, `linear_key_head_dim` and friends; Mamba2
configs give `mamba_d_state` and `mamba_d_conv` with the inner width
stated as `mamba_d_ssm` (Falcon-H1), an expansion factor, or only a head
split (Nemotron-H). When a config reports recurrent layers but no
geometry resolves, the script exits and names the keys it searched rather
than guessing a pool that often dominates KV.

Quantization is detected from the declared bit width, not from the format
name. `quant_method` is usually the container ("compressed-tensors",
"awq", "gptq", "bitsandbytes"), so the width is read from `bits`,
`w_bit`, `load_in_4bit`, `config_groups[*].weights.num_bits`, or a
`quant_algo` recipe name.

For mixed-precision quantized models, the script reads every spelling of
the exclusion list: `modules_to_not_convert`, `ignore`, `ignored_layers`,
`exclude_modules`, `exclude`, and `llm_int8_skip_modules`. Entries are
matched three ways, because all three are in use: a `re:` prefix is a
prefix-anchored regular expression, a pattern containing `*` is a shell
glob that also covers child modules, and anything else is a substring.
Missing a spelling charges the quantized rate for tensors that are
actually stored at full width.

Two components are treated as structural rather than probed:

- Embedding tables are always full precision. Every scheme here targets
  `Linear` modules and `nn.Embedding` is not one, so no config will ever
  list the embedding table in `targets`.
- The output projection is full precision unless the config asks for it,
  either with an explicit `lm_head: true` or a `config_groups` entry whose
  `targets` match it. An exclusion entry naming it wins either way.

The script prints a component-level weight breakdown whenever any
component differs from the uniform quantized rate.

## Tested Model Families

Dense coverage includes Qwen2.5, Llama 3.1/3.3, Gemma-2, and
Nemotron-70B style configs.

MoE coverage includes Qwen3-30B-A3B, Qwen3-235B-A22B, Mixtral,
DeepSeek-V3/V4-style hybrid MoE, and Mistral-Large-3 style
`params.json` configs.

Multimodal coverage includes Qwen2-VL, Gemma vision configs, LLaVA-like
configs, and Nemotron-Omni style `llm_config` layouts.

Sliding-window coverage includes Mistral and Ministral, Gemma-2 and
Gemma-3, Cohere2, gpt-oss, Llama-4, Phi-3, and Qwen3 configs that set
`use_sliding_window`.

Hybrid and state-space coverage includes Qwen3-Next gated delta net,
Nemotron-H, Bamba, Jamba, Falcon-H1 and Zamba2 parallel layers, Granite-4,
and pure Mamba/Mamba2 configs.

Mixed-precision coverage includes compressed-tensors `config_groups` with
`re:` targets, AWQ and GPTQ `modules_to_not_convert`, bitsandbytes
`llm_int8_skip_modules`, and MXFP4 checkpoints such as gpt-oss where the
router and attention stay at full precision.
