# SPDX-FileCopyrightText: Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Small shared helpers for model-config-recommend scripts."""
from __future__ import annotations

import json
import os
import subprocess
import sys
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path


def hf_headers(user_agent: str) -> dict[str, str]:
    headers = {"User-Agent": user_agent}
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return headers


def http_get_json(url: str, user_agent: str) -> dict | None:
    req = urllib.request.Request(url, headers=hf_headers(user_agent))
    try:
        # Bandit B310 suppression justification: callers build url from the https://huggingface.co literal;
        # scheme and host are not reachable from any parameter.
        with urllib.request.urlopen(req, timeout=30) as response:  # nosec B310
            return json.loads(response.read())
    except urllib.error.HTTPError as exc:
        if exc.code == 404:
            return None
        if exc.code in (401, 403):
            sys.exit(
                f"HTTP {exc.code} on {url}. Either typo, gated/private, "
                f"or rate-limited. Set HF_TOKEN and retry."
            )
        raise


def hf_model_info(model_id: str, user_agent: str) -> dict | None:
    return http_get_json(f"https://huggingface.co/api/models/{model_id}", user_agent)


def looks_like_gguf_repo(info: dict | None) -> bool:
    if not info:
        return False
    siblings = info.get("siblings") or []
    return any((s.get("rfilename") or "").lower().endswith(".gguf")
               for s in siblings)


def is_local_model(model_id: str) -> bool:
    """True if --model names a path on disk rather than a Hub repo id.

    Exported because callers need the same answer fetch_config uses: a local
    checkout has no mutable branch to resolve, so revision-pinning rules apply
    to Hub ids only. Two copies of this test could disagree.
    """
    return model_id.endswith(".json") or model_id.startswith(("/", "."))


def fetch_config(model_id: str, user_agent: str, revision: str = "main") -> dict:
    """Return a local or HF-hosted Transformers config.json.

    A local --model is caller-supplied data, so every way it can fail to be
    a JSON object is turned into a one-line message. Letting json/OSError
    escape here would print a traceback carrying absolute paths.

    ``revision`` is a Hub git revision -- branch, tag, or commit SHA. It
    defaults to ``main`` for backward compatibility, but a caller that pins the
    launch line to a revision must read the config from that same revision:
    recommending against ``main``'s config while pinning the engine elsewhere
    would describe a model the user is not about to run.
    """
    if is_local_model(model_id):
        try:
            cfg = json.loads(Path(model_id).read_text(encoding="utf-8"))
        except OSError as exc:
            sys.exit(f"cannot read {model_id}: {exc}")
        except UnicodeDecodeError as exc:
            sys.exit(f"{model_id} is not valid UTF-8 text: {exc}")
        except json.JSONDecodeError as exc:
            sys.exit(f"{model_id} is not valid JSON: {exc}")
        if not isinstance(cfg, dict):
            sys.exit(
                f"{model_id} must contain a JSON object, got "
                f"{type(cfg).__name__} — point --model at a config.json."
            )
        return cfg
    # quote() the revision, not the model id: a branch name may legitimately
    # contain characters that need escaping in a path segment, and the value
    # arrives from the command line.
    ref = urllib.parse.quote(revision, safe="")
    cfg = http_get_json(f"https://huggingface.co/{model_id}/raw/{ref}/config.json",
                        user_agent)
    if cfg is None:
        info = hf_model_info(model_id, user_agent)
        if looks_like_gguf_repo(info):
            sys.exit(
                f"{model_id} looks like a GGUF repository and has no root "
                f"config.json. Use a Transformers/safetensors checkpoint for "
                f"this analytical LLM path."
            )
        # Name the revision: now that it is caller-supplied, a typo'd SHA or a
        # tag that does not exist lands here, and "no config.json at root" sends
        # the reader looking at the wrong thing.
        sys.exit(
            f"No config.json at root for {model_id} at revision {revision}. "
            f"Either that revision does not exist, or this repo has no "
            f"HF Transformers-style config.json -- which this analytical LLM "
            f"path needs."
        )
    return cfg


# Layer-layout and recurrent-state handling below mirrors the model-can-it-fit
# skill's fit.py. The two skills are installed separately, so neither can import
# the other; the rationale for each rule lives in that skill's
# references/coverage-and-formulas.md.

BYTES_PER_STATE_DTYPE = {
    "bfloat16": 2.0, "bf16": 2.0,
    "float16": 2.0, "fp16": 2.0, "half": 2.0,
    "float32": 4.0, "fp32": 4.0, "float": 4.0,
    "float8_e4m3fn": 1.0, "float8_e5m2": 1.0, "fp8": 1.0,
}

# vLLM picks --max-num-batched-tokens from a device-memory tier table; only
# parts at or above 70 GiB get the larger tier.
VLLM_LARGE_DEVICE_GIB = 70
VLLM_MAX_NUM_BATCHED_TOKENS_LARGE = 8192
VLLM_MAX_NUM_BATCHED_TOKENS_SMALL = 2048

# SGLang's --swa-full-tokens-ratio default: its window pool is a fraction of
# the full-attention pool rather than a multiple of the window.
SGLANG_SWA_FULL_TOKENS_RATIO = 0.8

_RECURRENT_LAYER_TYPES = {
    "mamba", "mamba2", "ssm", "recurrent", "conv",
    "linear_attention", "linear_attn", "gated_delta_net",
}
_SLIDING_LAYER_TYPES = {
    "sliding_attention", "sliding_window_attention", "local_attention",
    "local_sliding_attention", "chunked_attention", "chunked_local_attention",
}
_FFN_ONLY_LAYER_TYPES = {"mlp", "moe", "feed_forward", "ffn"}

# Characters used by `hybrid_override_pattern` (Nemotron-H, Bamba):
# M = Mamba layer, * = attention layer, - = MLP layer.
_PATTERN_CHARS = {"m": "recurrent", "*": "full", "-": "ffn", "_": "ffn"}

# Families that run attention and an SSM inside the same layer, so every layer
# is charged both a KV cache and a state.
_PARALLEL_HYBRID_FAMILIES = {"falcon_h1", "zamba", "zamba2"}

# A bare `sliding_window` key does not prove the model uses a sliding window --
# DeepSeek-V4-Flash carries one for its sparse-attention indexer while every
# layer still holds a full cache. Apply a window only when the config says
# which layers use it, or when the family is known to apply it everywhere.
_GLOBAL_SWA_FAMILIES = {
    "mistral", "mistral3", "ministral", "mixtral",
    "qwen2", "qwen2_moe", "qwen2_vl", "qwen3", "qwen3_moe",
    "phi3", "phi3small", "phimoe",
    "starcoder2",
}
_INTERLEAVED_SWA_PERIOD = {"gemma2": 2, "cohere2": 4}


def _normalize_dtype(name: object, default: str = "bfloat16") -> str:
    text = str(name or "").lower().strip()
    if text in ("", "auto", "none"):
        return default
    text = text.replace("torch.", "")
    return text if text in BYTES_PER_STATE_DTYPE else default


def _lookup(sources: list[dict], keys: tuple[str, ...]) -> tuple[object, str]:
    """First truthy value among `keys` across `sources`, with the key used."""
    for src in sources:
        for key in keys:
            value = src.get(key)
            if value not in (None, 0, "", [], {}):
                return value, key
    return None, ""


def _scale_counts(counts: dict, listed: int, num_layers: int) -> dict:
    """Rescale a per-layer tally that does not cover exactly num_layers.

    The ratio between layer kinds is the load-bearing part, so it is preserved
    rather than the tally being discarded (which would silently charge every
    layer a full KV cache).
    """
    if listed == num_layers or listed <= 0:
        return counts
    scale = num_layers / listed
    return {key: int(round(value * scale)) for key, value in counts.items()}


def _layer_type_counts(types: list) -> dict:
    counts = {"full": 0, "sliding": 0, "recurrent": 0, "ffn": 0}
    for entry in types:
        name = str(entry).lower()
        if name in _RECURRENT_LAYER_TYPES:
            counts["recurrent"] += 1
        elif name in _SLIDING_LAYER_TYPES:
            counts["sliding"] += 1
        elif name in _FFN_ONLY_LAYER_TYPES:
            counts["ffn"] += 1
        else:
            counts["full"] += 1
    return counts


def _sliding_window(cfg: dict, text_cfg: dict) -> int:
    """Attention window in tokens, or 0 if the model is full-attention only."""
    sources = [text_cfg, cfg]
    for src in sources:
        if src.get("use_sliding_window") is False:
            return 0
    window, _ = _lookup(sources, ("sliding_window", "attention_chunk_size",
                                  "window_size", "attention_window_size"))
    if isinstance(window, (list, tuple)):
        window = window[0] if window else 0
    try:
        return max(int(window or 0), 0)
    except (TypeError, ValueError):
        return 0


def _classify_layers(cfg: dict, text_cfg: dict, family: str,
                     num_layers: int, window: int) -> dict:
    """Split num_layers into full-attention / sliding / recurrent / FFN counts."""
    sources = [text_cfg, cfg]
    out = {"full": 0, "sliding": 0, "recurrent": 0, "ffn": num_layers,
           "attn_on_recurrent": False, "layout_source": ""}

    layer_types, key = _lookup(sources, ("layer_types", "layer_type_list"))
    if isinstance(layer_types, (list, tuple)) and layer_types:
        counts = _scale_counts(_layer_type_counts(list(layer_types)),
                               len(layer_types), num_layers)
        out.update(counts, layout_source=key)
        if not counts["ffn"]:
            out["ffn"] = num_layers
        return out

    pattern, key = _lookup(sources, ("hybrid_override_pattern",
                                     "layers_block_type"))
    if isinstance(pattern, (list, tuple)) and pattern:
        counts = _scale_counts(_layer_type_counts(list(pattern)),
                               len(pattern), num_layers)
        out.update(counts, layout_source=key)
        if not counts["ffn"]:
            out["ffn"] = num_layers
        return out
    if isinstance(pattern, str) and pattern:
        counts = {"full": 0, "sliding": 0, "recurrent": 0, "ffn": 0}
        for char in pattern:
            counts[_PATTERN_CHARS.get(char.lower(), "full")] += 1
        if window and counts["full"]:
            counts["sliding"], counts["full"] = counts["full"], 0
        counts = _scale_counts(counts, len(pattern), num_layers)
        out.update(counts, layout_source=key)
        if not counts["ffn"]:
            out["ffn"] = num_layers
        return out

    if family in _PARALLEL_HYBRID_FAMILIES:
        out.update(full=num_layers, recurrent=num_layers,
                   attn_on_recurrent=True, layout_source="model_type")
        return out

    indices, key = _lookup(sources, ("attn_layer_indices",))
    if isinstance(indices, (list, tuple)):
        attn = len({int(i) for i in indices if 0 <= int(i) < num_layers})
        out.update(full=attn, recurrent=num_layers - attn, layout_source=key)
        return out

    period, key = _lookup(sources, ("attn_layer_period",))
    if period:
        attn = num_layers // int(period)
        out.update(full=attn, recurrent=num_layers - attn, layout_source=key)
        return out

    interval, key = _lookup(sources, ("full_attention_interval",))
    if interval:
        full = num_layers // int(interval)
        out.update(full=full, recurrent=num_layers - full, layout_source=key)
        return out

    stride, key = _lookup(sources, ("sliding_window_pattern",
                                    "global_attn_every_n_layers"))
    if stride and window:
        full = max(num_layers // int(stride), 1)
        out.update(full=full, sliding=num_layers - full, layout_source=key)
        return out

    if window and family in _INTERLEAVED_SWA_PERIOD:
        period = _INTERLEAVED_SWA_PERIOD[family]
        full = max(num_layers // period, 1)
        out.update(full=full, sliding=num_layers - full,
                   layout_source="model_type")
        return out

    if window and family in _GLOBAL_SWA_FAMILIES:
        out.update(sliding=num_layers, layout_source="sliding_window")
        return out

    out.update(full=num_layers)
    return out


def _state_geometry(cfg: dict, text_cfg: dict, hidden: int) -> dict | None:
    """Normalize a family's recurrent-state keys into one shape.

    Returns None when no geometry key resolves, so the caller can refuse
    rather than guess at a pool that often exceeds the KV cache.
    """
    sources = [text_cfg, cfg]
    seen: list[str] = []

    def take(keys: tuple[str, ...]):
        value, key = _lookup(sources, keys)
        if key:
            seen.append(key)
        return value

    # Gated delta net (Qwen3-Next, Kimi-Linear).
    num_v_heads = take(("linear_num_value_heads", "num_v_heads"))
    if num_v_heads:
        num_k_heads = take(("linear_num_key_heads", "num_k_heads")) or num_v_heads
        dim_k = take(("linear_key_head_dim", "head_k_dim")) or 0
        dim_v = take(("linear_value_head_dim", "head_v_dim")) or dim_k
        conv_kernel = take(("linear_conv_kernel_dim", "conv_kernel_size")) or 4
        if not (dim_k and dim_v):
            return None
        return {
            "num_state_heads": int(num_v_heads),
            "state_head_dim_k": int(dim_k),
            "state_head_dim_v": int(dim_v),
            "conv_dim": int(2 * dim_k * num_k_heads + dim_v * num_v_heads),
            "conv_kernel": int(conv_kernel),
            "state_groups": int(num_k_heads),
            "state_source": seen,
        }

    # Mamba / Mamba2 (Nemotron-H, Bamba, Falcon-H1, Jamba, Codestral-Mamba).
    d_state = take(("mamba_d_state", "state_size", "ssm_state_size", "d_state"))
    conv_kernel = take(("mamba_d_conv", "conv_kernel", "d_conv"))
    if not (d_state and conv_kernel):
        return None

    n_heads = take(("mamba_n_heads", "mamba_num_heads", "n_mamba_heads"))
    head_dim = take(("mamba_d_head", "mamba_head_dim"))

    inner = take(("mamba_d_ssm", "mamba_expand_inner_size"))
    if not inner:
        expand = take(("mamba_expand", "expand"))
        if expand:
            inner = int(expand) * hidden
    if not inner:
        inner = take(("mamba_intermediate_size", "mamba_d_inner", "d_inner"))
    if not inner and n_heads and head_dim:
        inner = int(n_heads) * int(head_dim)
    if not inner:
        return None

    if n_heads and not head_dim:
        head_dim = int(inner) // int(n_heads)
    elif head_dim and not n_heads:
        n_heads = int(inner) // int(head_dim)
    if not (n_heads and head_dim):
        # Mamba1: one (d_inner, d_state) state per layer, no head split.
        n_heads, head_dim = 1, int(inner)

    groups = take(("mamba_n_groups", "n_groups", "mamba_ngroups")) or 1
    return {
        "num_state_heads": int(n_heads),
        "state_head_dim_k": int(d_state),
        "state_head_dim_v": int(head_dim),
        "conv_dim": int(inner) + 2 * int(groups) * int(d_state),
        "conv_kernel": int(conv_kernel),
        "state_groups": int(groups),
        "state_source": seen,
    }


@dataclass(frozen=True)
class ModelDims:
    family: str
    hidden: int
    num_layers: int
    num_attn_heads: int
    num_kv_heads: int
    head_dim: int
    intermediate: int
    vocab: int
    tied: bool
    is_moe: bool
    # Layer layout. The three attention kinds cost different amounts, so the
    # cache is charged per layer rather than per model.
    num_full_attn_layers: int = 0
    num_sliding_attn_layers: int = 0
    num_recurrent_layers: int = 0
    sliding_window: int = 0
    attn_on_recurrent_layers: bool = False
    layout_source: str = ""
    # Recurrent (state-space) geometry, zero for attention-only models.
    num_state_heads: int = 0
    state_head_dim_k: int = 0
    state_head_dim_v: int = 0
    conv_dim: int = 0
    conv_kernel: int = 0
    state_groups: int = 0
    conv_dtype: str = "bfloat16"
    ssm_dtype: str = "bfloat16"
    state_source: list[str] = field(default_factory=list)


def parse_model_dims(cfg: dict) -> ModelDims:
    """Parse decoder-only LLM dimensions from a Transformers config."""
    family = cfg.get("model_type", "unknown")
    text_cfg = cfg.get("text_config") or cfg
    hidden = text_cfg.get("hidden_size") or text_cfg.get("d_model")
    num_layers = (text_cfg.get("num_hidden_layers") or text_cfg.get("n_layer")
                  or text_cfg.get("num_layers"))
    num_attn = text_cfg.get("num_attention_heads") or text_cfg.get("n_head")
    num_kv = text_cfg.get("num_key_value_heads", num_attn)
    head_dim = text_cfg.get("head_dim") or cfg.get("head_dim")
    if head_dim is None:
        head_dim = hidden // num_attn if (hidden and num_attn) else 0
    intermediate = text_cfg.get("intermediate_size") or 4 * (hidden or 0)
    vocab = text_cfg.get("vocab_size", cfg.get("vocab_size", 32000))
    tied = bool(text_cfg.get("tie_word_embeddings", cfg.get("tie_word_embeddings", False)))
    is_moe = bool(cfg.get("num_local_experts") or cfg.get("num_experts"))

    missing = [k for k, v in {"hidden_size": hidden, "num_hidden_layers": num_layers,
                              "num_attention_heads": num_attn}.items() if not v]
    if missing:
        sys.exit(f"config.json missing {missing}; not a decoder-only LLM.")

    hidden, num_layers = int(hidden), int(num_layers)
    window = _sliding_window(cfg, text_cfg)
    layers = _classify_layers(cfg, text_cfg, family, num_layers, window)
    if not layers["sliding"]:
        # The window belongs to something other than the attention layers.
        window = 0

    state = {}
    if layers["recurrent"]:
        geometry = _state_geometry(cfg, text_cfg, hidden)
        if geometry is None:
            sys.exit(
                f"{family} has {layers['recurrent']} recurrent layer(s) but no "
                f"readable state geometry. Looked for mamba_d_state/state_size, "
                f"mamba_d_conv/conv_kernel, an inner width "
                f"(mamba_d_ssm, mamba_expand, mamba_intermediate_size), and the "
                f"linear-attention keys (linear_num_value_heads, "
                f"linear_key_head_dim). Recurrent state often exceeds the KV "
                f"cache, so this is not safe to guess."
            )
        state = geometry

    dtype = _normalize_dtype(text_cfg.get("torch_dtype") or cfg.get("torch_dtype")
                             or cfg.get("dtype"))
    return ModelDims(
        family, hidden, num_layers, int(num_attn),
        int(num_kv), int(head_dim), int(intermediate),
        int(vocab), tied, is_moe,
        num_full_attn_layers=layers["full"],
        num_sliding_attn_layers=layers["sliding"],
        num_recurrent_layers=layers["recurrent"],
        sliding_window=window,
        attn_on_recurrent_layers=layers["attn_on_recurrent"],
        layout_source=layers["layout_source"],
        num_state_heads=state.get("num_state_heads", 0),
        state_head_dim_k=state.get("state_head_dim_k", 0),
        state_head_dim_v=state.get("state_head_dim_v", 0),
        conv_dim=state.get("conv_dim", 0),
        conv_kernel=state.get("conv_kernel", 0),
        state_groups=state.get("state_groups", 0),
        conv_dtype=dtype,
        ssm_dtype=dtype,
        state_source=state.get("state_source", []),
    )


def count_params(dims: ModelDims) -> int:
    """Estimate dense decoder params from config shape.

    This is the same approximation used by the recommender. It is config
    driven, so it scales to larger models without hand-entered layer counts.
    """
    h = dims.hidden
    q_proj = dims.head_dim * dims.num_attn_heads
    kv_proj = dims.head_dim * dims.num_kv_heads
    attn = h * q_proj + h * kv_proj + h * kv_proj + q_proj * h
    ff = 3 * h * dims.intermediate
    norms = 4 * h
    emb = dims.vocab * h
    head = 0 if dims.tied else dims.vocab * h
    return emb + head + dims.num_layers * (attn + ff + norms)


def _attention_layers(dims: ModelDims) -> tuple[int, int]:
    """(full-attention, sliding-window) layer counts, with a safe fallback."""
    full = dims.num_full_attn_layers
    sliding = dims.num_sliding_attn_layers
    if not (full or sliding):
        # A ModelDims built directly by a caller instead of parsed.
        full = dims.num_layers if not dims.num_recurrent_layers else 0
    return full, sliding


def kv_bytes_per_token(dims: ModelDims, kv_dtype_bytes: float) -> int:
    """Cache bytes each additional context token adds, across all layers.

    This is the roofline's per-token read cost, so it counts every layer that
    caches at all and ignores the window bound: it answers "how much more does
    one more token cost", not "how big is the pool". Use kv_bytes() for pool
    size. Recurrent layers are excluded because their state does not grow with
    context.
    """
    full, sliding = _attention_layers(dims)
    layers = full + sliding
    return int(2 * layers * dims.num_kv_heads * dims.head_dim * kv_dtype_bytes)


def default_max_num_batched_tokens(device_vram_gb: float) -> int:
    """vLLM's default --max-num-batched-tokens, from its device-memory tiers."""
    if device_vram_gb >= VLLM_LARGE_DEVICE_GIB:
        return VLLM_MAX_NUM_BATCHED_TOKENS_LARGE
    return VLLM_MAX_NUM_BATCHED_TOKENS_SMALL


def max_in_flight_tokens(device_vram_gb: float) -> int:
    """Tokens vLLM keeps addressable beyond a sliding window: two batches."""
    return 2 * default_max_num_batched_tokens(device_vram_gb)


def sliding_ctx(dims: ModelDims, ctx: int, runtime: str,
                in_flight_tokens: int) -> int:
    """Tokens one sliding-window layer must hold per request.

    vLLM pages the window and cannot free a block until every token in it has
    left the window, so it holds `window - 1` plus the tokens still in flight.
    SGLang sizes the window pool as a fraction of the full-attention pool so
    its radix prefix cache still has blocks to reuse. PyTorch allocates
    exactly the window.
    """
    window = dims.sliding_window
    if not window:
        return ctx
    if runtime.startswith("vllm"):
        return min(ctx, window - 1 + in_flight_tokens)
    if runtime.startswith("sglang"):
        pool = int(SGLANG_SWA_FULL_TOKENS_RATIO * ctx)
        return min(ctx, max(window, pool))
    return min(ctx, window)


def kv_bytes(dims: ModelDims, ctx: int, concurrency: int,
             kv_dtype_bytes: float, runtime: str = "vllm-xpu",
             in_flight_tokens: int = 0) -> int:
    """KV-cache pool bytes before the tensor-parallel divide.

    Charged per layer: a sliding-window layer holds a bounded number of tokens
    however long the context is, and a recurrent layer holds no KV cache.
    """
    per_token_per_layer = 2 * dims.num_kv_heads * dims.head_dim * kv_dtype_bytes
    full, sliding = _attention_layers(dims)
    tokens = full * ctx
    if sliding:
        tokens += sliding * sliding_ctx(dims, ctx, runtime, in_flight_tokens)
    return int(per_token_per_layer * tokens * concurrency)


def _shard_padded(total: int, shardable_units: int, tp: int) -> int:
    """Per-rank size of `total` split `tp` ways along `shardable_units`.

    When the unit count does not divide the tensor-parallel degree, vLLM adds
    padding units so each rank owns whole units, so an awkward degree costs
    more state rather than less.
    """
    if tp <= 1 or shardable_units <= 0:
        return total
    if shardable_units % tp == 0:
        return total // tp
    padded_units = shardable_units + (tp - shardable_units % tp)
    per_unit = total / shardable_units
    return int(per_unit * padded_units / tp)


def state_page_bytes(dims: ModelDims, tp: int, runtime: str = "vllm-xpu") -> int:
    """Recurrent-state bytes one request occupies, summed over SSM layers.

    Allocated per running request and never shrunk with context. The conv
    state holds `conv_kernel - 1` columns: the current token is computed, not
    cached. Neither engine quantizes this; SGLang forces the SSM half to fp32.
    """
    if not (dims.num_recurrent_layers and dims.num_state_heads):
        return 0

    conv_dim = _shard_padded(dims.conv_dim, dims.state_groups or 1, tp)
    heads = _shard_padded(dims.num_state_heads, dims.num_state_heads, tp)

    conv_bytes = conv_dim * max(dims.conv_kernel - 1, 1)
    conv_bytes *= BYTES_PER_STATE_DTYPE[dims.conv_dtype]

    ssm_dtype = "float32" if runtime.startswith("sglang") else dims.ssm_dtype
    ssm_bytes = heads * dims.state_head_dim_k * dims.state_head_dim_v
    ssm_bytes *= BYTES_PER_STATE_DTYPE[ssm_dtype]

    return int((conv_bytes + ssm_bytes) * dims.num_recurrent_layers)


def docker_image_id(image: str) -> str:
    try:
        return subprocess.check_output(
            ["docker", "inspect", "--format={{.Id}}", image],
            text=True,
        ).strip()
    except (FileNotFoundError, subprocess.CalledProcessError):
        return image


def docker_container_exists(name: str) -> bool:
    out = subprocess.check_output(
        ["docker", "ps", "-a", "--filter", f"name=^{name}$", "--format", "{{.Names}}"],
        text=True,
    )
    return any(line.strip() == name for line in out.splitlines())
