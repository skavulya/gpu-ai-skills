#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
"""Estimate VRAM needed to serve a Hugging Face model on an Intel GPU.

Pulls config.json from the Hub, computes weights + KV cache +
recurrent state + activations + framework overhead, prints a verdict.

Handles decoder-only LLMs, mixture-of-experts, sliding-window attention,
and hybrid state-space (Mamba / Mamba2 / gated-delta-net) models. VLM
towers are counted as weights; diffusion falls through to a weights-only
floor with a clear caveat.
"""
from __future__ import annotations

import argparse
import fnmatch
import json
import re
import sys
import urllib.request
from dataclasses import dataclass, field


GB = 1024 ** 3
MB = 1024 ** 2
 

# Bytes per parameter, including typical scale/zero overhead at
# group_size=128 for grouped quants.
BYTES_PER_PARAM = {
    "bf16":  2.00,
    "fp16":  2.00,
    "fp8":   1.00,
    "int8":  1.00,
    "int4":  0.55,   # AWQ / GPTQ / AutoRound int4
    "int3":  0.42,
    "int2":  0.30,
    "mxfp4": 0.55,
}

BYTES_PER_KV = {
    "bf16": 2.0, "fp16": 2.0, "fp8": 1.0, "int8": 1.0,
}

# Element size of a recurrent (conv / SSM) state tensor, keyed by the dtype
# name a config or a runtime flag can use. Recurrent state is never
# quantized by either engine, so this table is separate from BYTES_PER_KV.
BYTES_PER_STATE_DTYPE = {
    "bfloat16": 2.0, "bf16": 2.0,
    "float16": 2.0, "fp16": 2.0, "half": 2.0,
    "float32": 4.0, "fp32": 4.0, "float": 4.0,
    "float8_e4m3fn": 1.0, "float8_e5m2": 1.0, "fp8": 1.0,
}

# Empirical floors on Arc Pro B70.
FRAMEWORK_OVERHEAD_GB = {
    "vllm":   2.0,
    "sglang": 1.5,
    "torch":  0.8,
}

# vLLM's default --max-num-batched-tokens for an OpenAI API server, from the
# device-memory tier table in vllm/engine/arg_utils.py. Small cards get the
# 2048 tier; only >=70 GiB datacentre parts get 8192.
VLLM_LARGE_DEVICE_GIB = 70
VLLM_MAX_NUM_BATCHED_TOKENS_LARGE = 8192
VLLM_MAX_NUM_BATCHED_TOKENS_SMALL = 2048

# SGLang does not size its sliding-window pool to the window. It provisions
# the window pool as a fraction of the full-attention pool so the radix
# prefix cache still has something to reuse (--swa-full-tokens-ratio).
SGLANG_SWA_FULL_TOKENS_RATIO = 0.8

TABLE_MODELS = [
    "Qwen/Qwen2.5-1.5B-Instruct",
    "Qwen/Qwen2.5-7B-Instruct",
    "Qwen/Qwen2.5-14B-Instruct",
    "Qwen/Qwen2.5-32B-Instruct",
]


def _http_get_json(url: str) -> dict | None:
    """Fetch JSON from a URL. Returns None on 404; exits on other errors."""
    import os
    headers = {"User-Agent": "model-can-it-fit/0.1"}
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if token:
        headers["Authorization"] = f"Bearer {token}"
    req = urllib.request.Request(url, headers=headers)
    try:
        # Bandit B310 suppression justification: url is built from the https://huggingface.co literal at
        # fetch_config(); scheme and host are not reachable from any parameter.
        with urllib.request.urlopen(req, timeout=30) as r:  # nosec B310
            return json.loads(r.read())
    except urllib.error.HTTPError as e:
        if e.code == 404:
            return None
        if e.code in (401, 403):
            sys.exit(
                f"HTTP {e.code} fetching {url}. Either the model id is "
                f"wrong (typo, case mismatch) or the model is gated / "
                f"private. Verify the page exists in a browser; if it "
                f"does, set HF_TOKEN (huggingface-cli login) and retry, "
                f"or pass a local path to config.json instead."
            )
        raise


def fetch_config(model_id: str, revision: str = "main") -> dict:
    """Fetch config.json or params.json from HF Hub or local path.

    For diffusion repos (model_index.json at root, no config.json),
    raises SystemExit with a useful message — diffusion pipelines have
    multi-component memory profiles this calculator cannot model.
    """
    if model_id.startswith(("/", ".")) or model_id.endswith(".json"):
        try:
            with open(model_id, encoding="utf-8") as f:
                cfg = json.load(f)
        except OSError as e:
            sys.exit(f"cannot read {model_id}: {e}")
        except UnicodeDecodeError as e:
            sys.exit(f"{model_id} is not valid UTF-8 text: {e}")
        except json.JSONDecodeError as e:
            sys.exit(f"{model_id} is not valid JSON: {e}")
        if not isinstance(cfg, dict):
            sys.exit(
                f"{model_id} must contain a JSON object, got "
                f"{type(cfg).__name__} — point --model at a config.json."
            )
        return cfg

    base = f"https://huggingface.co/{model_id}/raw/{revision}"
    cfg = _http_get_json(f"{base}/config.json")
    if cfg is not None:
        return cfg

    # No config.json: try params.json (some Mistral models use this)
    params = _http_get_json(f"{base}/params.json")
    if params is not None:
        return params

    # No config.json or params.json: check for diffusion-pipeline shape.
    midx = _http_get_json(f"{base}/model_index.json")
    if midx is not None:
        components = sorted(k for k, v in midx.items()
                            if isinstance(v, list) and v and v[0])
        raise SystemExit(
            f"{model_id} is a diffusion pipeline (model_index.json present, "
            f"no top-level config.json). This calculator does not estimate "
            f"diffusion VRAM — peak usage is dominated by intermediate "
            f"latents during denoising, which depend on resolution, step "
            f"count, and scheduler in ways a config-only calc cannot model.\n"
            f"\n"
            f"Components present: {', '.join(components) or '(empty)'}\n"
            f"\n"
            f"As a floor estimate of the largest component's weights, point "
            f"this script at one subdir directly, e.g.:\n"
            f"    --model {model_id}/unet            (or transformer for DiT)\n"
            f"For a real fit answer, run **torch-xpu-bench**'s diffusion "
            f"snippet with --runs 1 and read the Peak XPU memory line."
        )

    raise SystemExit(
        f"HTTP 404 fetching config for {model_id}. Neither config.json, "
        f"params.json, nor model_index.json exists at the root of revision "
        f"'{revision}'. Verify the model id and revision in a browser."
    )


@dataclass
class ModelDims:
    arch_family: str
    hidden: int
    num_layers: int
    num_attn_heads: int
    num_kv_heads: int
    head_dim: int
    intermediate: int
    vocab: int
    tied: bool
    is_moe: bool
    num_experts: int
    num_experts_per_tok: int
    num_shared_experts: int = 0
    first_k_dense_replace: int = 0
    dense_intermediate: int = 0
    is_vlm: bool = False
    vision_params: int = 0
    vision_num_heads: int = 0

    # --- Layer composition -------------------------------------------------
    # For a plain dense model every layer is full attention and every layer
    # carries a feed-forward block, so these default to num_layers in
    # parse_dims. Hybrids split the total: an SSM layer holds recurrent state
    # instead of a KV cache, a sliding-window layer holds a bounded KV cache,
    # and some families (Nemotron-H, Bamba) interleave attention-free and
    # feed-forward-free layers.
    num_full_attn_layers: int = 0
    num_sliding_attn_layers: int = 0
    num_recurrent_layers: int = 0
    num_ffn_layers: int = 0
    sliding_window: int = 0

    # --- Recurrent state geometry -----------------------------------------
    # Normalized across Mamba, Mamba2 and gated-delta-net. The per-layer SSM
    # state is (num_state_heads, state_head_dim_k, state_head_dim_v) and the
    # per-layer conv state is (conv_dim, conv_kernel - 1).
    num_state_heads: int = 0
    state_head_dim_k: int = 0
    state_head_dim_v: int = 0
    conv_dim: int = 0
    conv_kernel: int = 0
    state_size: int = 0
    state_groups: int = 1
    state_inner: int = 0
    conv_dtype: str = "bfloat16"
    ssm_dtype: str = "bfloat16"
    # Falcon-H1 and Zamba2 run attention and an SSM in the same layer, so
    # those layers are charged a full KV cache *and* a recurrent state.
    attn_on_recurrent_layers: bool = False
    # Which config keys the state geometry was actually read from, for the
    # refusal message and for tests.
    state_source: list[str] = field(default_factory=list)


def _vlm_signal(cfg: dict) -> bool:
    archs = cfg.get("architectures") or []
    arch = archs[0] if archs else ""
    if "vision_config" in cfg:
        return True
    if any(s in arch for s in ("VL", "VisionLanguage", "VLForConditional")):
        return True
    if any(s in cfg.get("model_type", "") for s in ("_vl", "vlm", "vision")):
        return True
    return False


def _vision_param_count(vc: dict) -> int:
    """Approximate parameter count of a ViT-style vision tower."""
    h = vc.get("hidden_size", 0)
    layers = vc.get("depth") or vc.get("num_hidden_layers") or 0
    intermediate = vc.get("intermediate_size", 4 * h)
    if not (h and layers):
        return 0
    # ViT layer: attn (4 h^2) + MLP (2 h * intermediate) + 2 norms (2 h).
    per_layer = 4 * h * h + 2 * h * intermediate + 2 * h
    # Patch embedding: roughly hidden * patch_area * channels. Bound modestly.
    patch = vc.get("patch_size", 14)
    in_channels = vc.get("num_channels", 3)
    embed = patch * patch * in_channels * h
    return embed + layers * per_layer


# Names a config's `layer_types` list can use. Anything unrecognized is
# treated as full attention, which is the conservative direction: a full KV
# cache is the most expensive thing a layer can hold.
_RECURRENT_LAYER_TYPES = {
    "mamba", "mamba2", "ssm", "recurrent", "conv",
    "linear_attention", "linear_attn", "gated_delta_net",
}
_SLIDING_LAYER_TYPES = {
    "sliding_attention", "sliding_window_attention", "local_attention",
    "local_sliding_attention", "chunked_attention", "chunked_local_attention",
}
_FULL_LAYER_TYPES = {"full_attention", "attention", "global_attention", "full"}
_FFN_ONLY_LAYER_TYPES = {"mlp", "moe", "feed_forward", "ffn"}

# Single characters used by the `hybrid_override_pattern` string
# (Nemotron-H, Bamba): M = Mamba layer, * = attention layer, - = MLP layer.
_PATTERN_CHARS = {"m": "recurrent", "*": "full", "-": "ffn", "_": "ffn"}

# Families that run attention and an SSM inside the *same* layer rather than
# alternating them, so every layer is charged both a KV cache and a state.
_PARALLEL_HYBRID_FAMILIES = {"falcon_h1", "zamba", "zamba2"}

# A bare `sliding_window` key does not mean the model uses a sliding window.
# DeepSeek-V4-Flash carries `sliding_window: 128` for its sparse attention
# indexer while every layer still holds a full KV cache -- honouring that key
# blindly under-estimates its cache by 8x, and under-estimating is what
# produces a FITS verdict for a launch that dies at engine init. So a window
# is only applied when the config says which layers use it, or when the
# family is known to apply it everywhere.
_GLOBAL_SWA_FAMILIES = {
    "mistral", "mistral3", "ministral", "mixtral",
    "qwen2", "qwen2_moe", "qwen2_vl", "qwen3", "qwen3_moe",
    "phi3", "phi3small", "phimoe",
    "starcoder2",
}

# Families whose sliding/global interleave lives in the modeling code rather
# than in any config key, mapped to the period between global layers.
_INTERLEAVED_SWA_PERIOD = {"gemma2": 2, "cohere2": 4}


def _normalize_dtype(name: object, default: str = "bfloat16") -> str:
    """Map a config/flag dtype name onto a BYTES_PER_STATE_DTYPE key."""
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

    A layout list is occasionally shorter or longer than num_hidden_layers --
    a repeating block, or a config key this script has matched to the wrong
    family. The ratio between layer kinds is the load-bearing part, so it is
    preserved and applied to the real layer count instead of the tally being
    thrown away (which would silently charge every layer a full KV cache).
    """
    if listed == num_layers or listed <= 0:
        return counts
    scale = num_layers / listed
    return {key: int(round(value * scale)) for key, value in counts.items()}


def _layer_type_counts(types: list) -> dict:
    """Tally an explicit per-layer list into layer kinds."""
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
    """Attention window in tokens, or 0 if the model is full-attention only.

    A window equal to or wider than the trained context is not a window --
    several configs carry `sliding_window` alongside
    `use_sliding_window: false`, and Llama-4 spells its window
    `attention_chunk_size`.
    """
    sources = [text_cfg, cfg]
    for src in sources:
        if src.get("use_sliding_window") is False:
            return 0
    window, _ = _lookup(sources, ("sliding_window", "attention_chunk_size",
                                  "window_size", "attention_window_size"))
    if isinstance(window, (list, tuple)):
        # Some configs give a (left, right) pair; the left half is the span
        # of past tokens a query can see, which is what the cache must hold.
        window = window[0] if window else 0
    try:
        return max(int(window or 0), 0)
    except (TypeError, ValueError):
        return 0


def _classify_layers(cfg: dict, text_cfg: dict, family: str,
                     num_layers: int, window: int) -> dict:
    """Split num_layers into full-attention / sliding / recurrent / FFN counts.

    Reads whichever of the mutually exclusive layout keys a family uses:
    `layer_types` (Gemma-3, Qwen3-Next, gpt-oss, Llama-4),
    `hybrid_override_pattern` (Nemotron-H, Bamba), `attn_layer_indices`, or a
    periodic interleave (`attn_layer_period`, `full_attention_interval`,
    `sliding_window_pattern`).
    """
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
        # Attention and SSM sit side by side in every layer.
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
        # Gemma-3, Cohere2: every `stride`-th layer is global, rest windowed.
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

    # Window present but unattributable: charge full attention everywhere.
    out.update(full=num_layers)
    return out


def _state_geometry(cfg: dict, text_cfg: dict, hidden: int) -> dict | None:
    """Normalize a family's recurrent-state keys into one shape.

    The per-layer SSM state is (num_state_heads, state_head_dim_k,
    state_head_dim_v) and the per-layer conv state is (conv_dim,
    conv_kernel - 1). Mamba1 collapses to a single head whose value
    dimension is the whole inner width. Returns None when no geometry key
    resolves, so the caller can refuse rather than guess.
    """
    sources = [text_cfg, cfg]
    seen: list[str] = []

    def take(keys: tuple[str, ...]):
        value, key = _lookup(sources, keys)
        if key:
            seen.append(key)
        return value

    # Gated delta net (Qwen3-Next, Kimi-Linear): key and value heads are
    # sized independently, and the conv state spans q, k and v.
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
            "state_size": int(dim_k),
            "state_groups": int(num_k_heads),
            "state_inner": int(dim_v * num_v_heads),
            "state_source": seen,
        }

    # Mamba / Mamba2 (Nemotron-H, Bamba, Falcon-H1, Codestral-Mamba, Jamba).
    d_state = take(("mamba_d_state", "state_size", "ssm_state_size", "d_state"))
    conv_kernel = take(("mamba_d_conv", "conv_kernel", "d_conv"))
    if not (d_state and conv_kernel):
        return None

    n_heads = take(("mamba_n_heads", "mamba_num_heads", "n_mamba_heads"))
    head_dim = take(("mamba_d_head", "mamba_head_dim"))

    # Inner width, in descending order of how directly a family states it.
    # Falcon-H1 gives mamba_d_ssm outright; most Mamba2 configs give an
    # expansion factor; Nemotron-H gives only the head split.
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
        "state_size": int(d_state),
        "state_groups": int(groups),
        "state_inner": int(inner),
        "state_source": seen,
    }


def parse_dims(cfg: dict) -> ModelDims:
    arch = (cfg.get("architectures") or ["unknown"])[0]
    family = cfg.get("model_type", "unknown")
    is_vlm = _vlm_signal(cfg)

    # For VLMs the LLM-backbone fields often live under 'text_config' or 'llm_config'
    # rather than at the root.
    # text_config: Qwen2-VL, Llama-3.2-Vision, etc.
    # llm_config: Nemotron-3-Nano-Omni, some other multimodal models
    text_cfg = cfg.get("text_config") or cfg.get("llm_config") or cfg

    # Support both config.json and params.json field names
    # config.json uses: hidden_size, num_hidden_layers, num_attention_heads
    # params.json uses: dim, n_layers, n_heads (Mistral format)
    hidden = text_cfg.get("hidden_size") or text_cfg.get("d_model") or text_cfg.get("dim")
    num_layers = (
        text_cfg.get("num_hidden_layers")
        or text_cfg.get("n_layer")
        or text_cfg.get("num_layers")
        or text_cfg.get("n_layers")
    )
    num_attn = text_cfg.get("num_attention_heads") or text_cfg.get("n_head") or text_cfg.get("n_heads")
    num_kv = text_cfg.get("num_key_value_heads") or text_cfg.get("n_kv_heads") or num_attn

    # Head dimension - use explicit head_dim if present, otherwise calculate
    head_dim = text_cfg.get("head_dim")
    if head_dim is None:
        head_dim = cfg.get("head_dim")
    if head_dim is None:
        head_dim = hidden // num_attn if (hidden and num_attn) else 0

    vocab = text_cfg.get("vocab_size", cfg.get("vocab_size", 32000))
    # params.json uses "tied_embeddings", config.json uses "tie_word_embeddings"
    tied = bool(text_cfg.get("tie_word_embeddings",
                             cfg.get("tie_word_embeddings",
                             cfg.get("tied_embeddings", False))))

    vision_params = 0
    vision_num_heads = 0
    if is_vlm and isinstance(cfg.get("vision_config"), dict):
        vision_cfg = cfg["vision_config"]
        vision_params = _vision_param_count(vision_cfg)
        vision_num_heads = int(
            vision_cfg.get("num_attention_heads")
            or vision_cfg.get("num_heads")
            or 0
        )

    # MoE detection - do this BEFORE reading intermediate_size to avoid picking wrong field
    # Check multiple field names used by different MoE architectures
    # For VLMs, check text_config first (like other backbone fields)
    # params.json (Mistral) nests these under "moe" key
    moe_cfg = cfg.get("moe", text_cfg.get("moe", {}))

    # Use nested get() with defaults to avoid treating 0 as missing
    num_experts = (
        text_cfg.get("num_local_experts",
        text_cfg.get("num_experts",
        text_cfg.get("n_routed_experts",      # DeepSeek-V4
        cfg.get("num_local_experts",
        cfg.get("num_experts",
        cfg.get("n_routed_experts",
        moe_cfg.get("num_experts", 0)))))))   # params.json (Mistral)
    )
    num_experts_per_tok = (
        text_cfg.get("num_experts_per_tok",
        text_cfg.get("num_experts_per_token",
        cfg.get("num_experts_per_tok",
        cfg.get("num_experts_per_token",
        moe_cfg.get("num_experts_per_tok", 0)))))  # params.json (Mistral)
    )
    num_shared_experts = (
        text_cfg.get("n_shared_experts",
        cfg.get("n_shared_experts",
        moe_cfg.get("num_shared_experts", 0)))     # params.json (Mistral)
    )

    # Hybrid MoE: some models (DeepSeek-V2/V3/V4) replace first K layers with dense FFN
    first_k_dense_replace = (
        text_cfg.get("first_k_dense_replace",
        cfg.get("first_k_dense_replace",
        moe_cfg.get("first_k_dense_replace", 0)))
    )

    is_moe = num_experts and num_experts > 1

    # Intermediate size detection - AFTER MoE detection to pick correct field
    # For MoE models: use moe_intermediate_size (expert FFN size)
    # For dense models: use intermediate_size (standard FFN size)
    # Some dense models (BART/OPT/M2M) use ffn_dim, so only check it for non-MoE
    if is_moe:
        # MoE model: prefer moe_intermediate_size / expert_hidden_dim
        intermediate = (
            text_cfg.get("moe_intermediate_size")      # Qwen3 MoE, DeepSeek-V4
            or cfg.get("moe_intermediate_size")        # Root-level fallback
            or moe_cfg.get("expert_hidden_dim")        # params.json (Mistral)
            or text_cfg.get("ffn_dim")                 # Some MoE variants
            or cfg.get("ffn_dim")
            or text_cfg.get("hidden_dim")              # params.json fallback
            or cfg.get("hidden_dim")
            or text_cfg.get("intermediate_size")       # Fallback
        )
        # Dense FFN size for hybrid MoE (first_k_dense_replace layers)
        # params.json (Mistral) uses hidden_dim for dense FFN
        # config.json (DeepSeek) uses intermediate_size for dense FFN
        dense_intermediate = (
            text_cfg.get("intermediate_size")
            or cfg.get("intermediate_size")
            or text_cfg.get("hidden_dim")              # params.json (Mistral)
            or cfg.get("hidden_dim")
            or 0
        )
    else:
        # Dense model: prefer intermediate_size, then ffn_dim, then hidden_dim
        intermediate = (
            text_cfg.get("intermediate_size")          # Standard dense models
            or text_cfg.get("ffn_dim")                 # BART/OPT/M2M
            or cfg.get("ffn_dim")
            or text_cfg.get("hidden_dim")              # params.json
            or cfg.get("hidden_dim")
        )
        dense_intermediate = 0  # Not used for pure dense models

    # Fall back to 4*hidden if nothing found
    if intermediate is None:
        intermediate = 4 * (hidden or 0)

    missing = [k for k, v in {
        "hidden_size": hidden, "num_hidden_layers": num_layers,
        "num_attention_heads": num_attn,
    }.items() if not v]
    if missing:
        sys.exit(
            f"config.json is missing required keys for an LLM: {missing}. "
            f"Architecture reported: {arch} / {family}. "
            f"This calculator only handles decoder-only LLMs reliably."
        )

    window = _sliding_window(cfg, text_cfg)
    layout = _classify_layers(cfg, text_cfg, family, int(num_layers), window)
    if not layout["sliding"]:
        # The config named a window but nothing said which layers use it.
        window = 0

    state = _state_geometry(cfg, text_cfg, int(hidden)) or {}
    if layout["recurrent"] and not state:
        searched = ", ".join((
            "linear_num_value_heads/linear_key_head_dim/linear_value_head_dim/"
            "linear_conv_kernel_dim (gated delta net)",
            "mamba_d_state/state_size/ssm_state_size/d_state",
            "mamba_d_conv/conv_kernel/d_conv",
            "mamba_d_ssm/mamba_expand/mamba_intermediate_size/d_inner",
        ))
        sys.exit(
            f"{arch} / {family} reports {layout['recurrent']} recurrent "
            f"(state-space) layer(s) via '{layout['layout_source']}', but no "
            f"recurrent-state geometry could be read from the config. Without "
            f"it the state pool -- which does not shrink with context and "
            f"often dominates KV on hybrids -- cannot be sized, and a verdict "
            f"would be confidently wrong.\n"
            f"Keys searched: {searched}.\n"
            f"If this family names them differently, pass a local config.json "
            f"with the equivalent keys added, or file an issue with the "
            f"architecture name."
        )

    model_dtype = _normalize_dtype(cfg.get("torch_dtype")
                                   or text_cfg.get("torch_dtype"))
    ssm_dtype = _normalize_dtype(
        _lookup([text_cfg, cfg], ("mamba_ssm_cache_dtype", "ssm_cache_dtype"))[0],
        default=model_dtype,
    )

    return ModelDims(
        arch_family=family,
        hidden=hidden,
        num_layers=num_layers,
        num_attn_heads=num_attn,
        num_kv_heads=num_kv,
        head_dim=head_dim,
        intermediate=intermediate,
        vocab=vocab,
        tied=tied,
        is_moe=bool(is_moe),
        num_experts=int(num_experts or 0),
        num_experts_per_tok=int(num_experts_per_tok or 0),
        num_shared_experts=int(num_shared_experts),
        first_k_dense_replace=int(first_k_dense_replace),
        dense_intermediate=int(dense_intermediate),
        is_vlm=is_vlm,
        vision_params=vision_params,
        vision_num_heads=vision_num_heads,
        num_full_attn_layers=layout["full"],
        num_sliding_attn_layers=layout["sliding"],
        num_recurrent_layers=layout["recurrent"],
        num_ffn_layers=layout["ffn"],
        sliding_window=window,
        num_state_heads=state.get("num_state_heads", 0),
        state_head_dim_k=state.get("state_head_dim_k", 0),
        state_head_dim_v=state.get("state_head_dim_v", 0),
        conv_dim=state.get("conv_dim", 0),
        conv_kernel=state.get("conv_kernel", 0),
        state_size=state.get("state_size", 0),
        state_groups=state.get("state_groups", 1),
        state_inner=state.get("state_inner", 0),
        conv_dtype=model_dtype,
        ssm_dtype=ssm_dtype,
        attn_on_recurrent_layers=layout["attn_on_recurrent"],
        state_source=state.get("state_source", []),
    )


def _recurrent_block_params(d: ModelDims) -> int:
    """Weights of one recurrent (Mamba / Mamba2 / gated-delta-net) layer.

    Mamba2 and gated-delta-net differ in how they name their projections but
    land on the same shapes: one input projection wide enough to feed the
    conv state plus the value stream, a depthwise conv, and one output
    projection back to hidden. The per-head scalars (A_log, D, dt_bias) and
    the gated output norm are small but cheap to include.
    """
    if not d.num_state_heads:
        return 0
    h = d.hidden
    in_proj = h * (d.conv_dim + d.state_inner + 2 * d.num_state_heads)
    conv = d.conv_dim * d.conv_kernel + d.conv_dim
    out_proj = d.state_inner * h
    scalars = 3 * d.num_state_heads
    gate_norm = d.state_head_dim_v * d.num_state_heads
    return in_proj + conv + out_proj + scalars + gate_norm


def _ffn_param_split(d: ModelDims) -> tuple[int, int, int]:
    """(dense-FFN, shared-expert, routed-expert) parameter counts.

    Kept separate because quantization schemes routinely treat them
    differently: a compressed-tensors export commonly quantizes the routed
    experts -- where nearly all the weight is -- while leaving the dense
    replacement layers and the shared expert at full width. Charging one rate
    across all three is wrong by a factor, not a margin.
    """
    h = d.hidden
    ffn_layers = d.num_ffn_layers or d.num_layers
    if not d.is_moe:
        return ffn_layers * 3 * h * d.intermediate, 0, 0

    routed_per_layer = d.num_experts * 3 * h * d.intermediate
    shared_per_layer = d.num_shared_experts * 3 * h * d.intermediate

    dense_layers = 0
    if d.first_k_dense_replace > 0 and d.dense_intermediate > 0:
        dense_layers = min(d.first_k_dense_replace, ffn_layers)
    moe_layers = ffn_layers - dense_layers
    dense = dense_layers * 3 * h * d.dense_intermediate
    return dense, moe_layers * shared_per_layer, moe_layers * routed_per_layer


def count_params(d: ModelDims) -> int:
    h = d.hidden
    # Q projection dimension: for models with explicit head_dim ≠ hidden/num_heads,
    # Q projects to num_attn_heads * head_dim (not h)
    q_proj_dim = d.num_attn_heads * d.head_dim
    kv_proj_dim = d.num_kv_heads * d.head_dim
    attn_block = (
        h * q_proj_dim     # Q: hidden → num_attn_heads * head_dim
        + h * kv_proj_dim  # K: hidden → num_kv_heads * head_dim
        + h * kv_proj_dim  # V: hidden → num_kv_heads * head_dim
        + q_proj_dim * h   # O: num_attn_heads * head_dim → hidden
        + 2 * h            # input + post-attention norms
    )

    # Hybrids replace attention with an SSM on some layers, and families like
    # Nemotron-H also have layers that carry neither. For a plain model all
    # three counts collapse to num_layers.
    attn_layers = d.num_full_attn_layers + d.num_sliding_attn_layers or d.num_layers
    ffn_layers = d.num_ffn_layers or d.num_layers

    total = attn_layers * attn_block
    total += d.num_recurrent_layers * (_recurrent_block_params(d) + 2 * h)

    # Dense replacement layers (DeepSeek-V2/V3/V4's first_k_dense_replace),
    # shared experts, and routed experts, each counted over the layers that
    # actually carry them.
    total += sum(_ffn_param_split(d))
    total += ffn_layers * 2 * h   # pre-FFN norm + residual norm

    emb = d.vocab * h
    head = 0 if d.tied else d.vocab * h
    return emb + head + total


def default_max_num_batched_tokens(device_vram_gb: float) -> int:
    """vLLM's default --max-num-batched-tokens for an OpenAI API server.

    vLLM picks this from a device-memory tier table; only parts at or above
    70 GiB get the 8192 tier. It matters here because it sets how many
    uncommitted tokens a sliding-window block pool must be able to hold.
    """
    if device_vram_gb >= VLLM_LARGE_DEVICE_GIB:
        return VLLM_MAX_NUM_BATCHED_TOKENS_LARGE
    return VLLM_MAX_NUM_BATCHED_TOKENS_SMALL


def max_in_flight_tokens(device_vram_gb: float) -> int:
    """Tokens vLLM must keep addressable beyond a sliding window.

    A windowed layer can only drop a block once every token in it has left
    the window, and vLLM sizes that slack at two scheduler batches.
    """
    return 2 * default_max_num_batched_tokens(device_vram_gb)


def sliding_ctx(d: ModelDims, ctx: int, runtime: str,
                in_flight_tokens: int) -> int:
    """Tokens a sliding-window layer must hold per request.

    The three runtimes charge visibly different amounts for the same window:

    * vllm pages the window and cannot free a block until every token in it
      has left the window, so it holds `window - 1` plus the tokens still in
      flight across the current batches.
    * sglang sizes the window pool as a fraction of the full-attention pool
      (--swa-full-tokens-ratio) instead of to the window, so the radix
      prefix cache still has blocks to reuse.
    * torch / transformers allocates exactly the window.
    """
    window = d.sliding_window
    if not window:
        return ctx
    if runtime == "vllm":
        return min(ctx, window - 1 + in_flight_tokens)
    if runtime == "sglang":
        pool = int(SGLANG_SWA_FULL_TOKENS_RATIO * ctx)
        return min(ctx, max(window, pool))
    return min(ctx, window)


def kv_bytes(d: ModelDims, ctx: int, concurrency: int, kv_dtype: str,
             runtime: str = "vllm", in_flight_tokens: int = 0) -> int:
    """KV-cache bytes across every attention layer, before the TP divide.

    Charged per layer rather than per model: a sliding-window layer holds a
    bounded number of tokens no matter how long the context is, and a
    recurrent layer holds no KV cache at all.
    """
    per_token_per_layer = 2 * d.num_kv_heads * d.head_dim * BYTES_PER_KV[kv_dtype]

    full_layers = d.num_full_attn_layers
    sliding_layers = d.num_sliding_attn_layers
    if not (full_layers or sliding_layers):
        # Pre-layout configs and direct ModelDims construction in callers.
        full_layers = d.num_layers if not d.num_recurrent_layers else 0

    tokens = full_layers * ctx
    if sliding_layers:
        tokens += sliding_layers * sliding_ctx(d, ctx, runtime, in_flight_tokens)
    return int(per_token_per_layer * tokens * concurrency)


def state_page_bytes(d: ModelDims, tp: int, runtime: str = "vllm") -> int:
    """Recurrent-state bytes one request occupies, summed over SSM layers.

    Both engines allocate this per running request and never shrink it with
    context -- an SSM layer's state is the same size for 1 token as for 1M.
    The conv state is (conv_dim, conv_kernel - 1): the current token is
    computed, not stored, so the cache holds one fewer column than the
    kernel is wide.
    """
    if not (d.num_recurrent_layers and d.num_state_heads):
        return 0

    # Head shards must divide the TP degree. vLLM pads the group/head count
    # up to the next multiple rather than splitting a head across ranks, so
    # sharding a hybrid at an awkward TP costs more state, not less.
    conv_dim = _shard_padded(d.conv_dim, d.state_groups or 1, tp)
    heads = _shard_padded(d.num_state_heads, d.num_state_heads, tp)

    conv_bytes = conv_dim * max(d.conv_kernel - 1, 1)
    conv_bytes *= BYTES_PER_STATE_DTYPE[d.conv_dtype]

    ssm_dtype = "float32" if runtime == "sglang" else d.ssm_dtype
    ssm_bytes = heads * d.state_head_dim_k * d.state_head_dim_v
    ssm_bytes *= BYTES_PER_STATE_DTYPE[ssm_dtype]

    return int((conv_bytes + ssm_bytes) * d.num_recurrent_layers)


def _shard_padded(total: int, shardable_units: int, tp: int) -> int:
    """Per-rank size of `total` split `tp` ways along `shardable_units`.

    When the unit count does not divide TP, vLLM adds padding units
    (extra_groups_for_head_shards) so each rank owns whole units.
    """
    if tp <= 1 or shardable_units <= 0:
        return total
    if shardable_units % tp == 0:
        return total // tp
    padded_units = shardable_units + (tp - shardable_units % tp)
    per_unit = total / shardable_units
    return int(per_unit * padded_units / tp)


def activation_bytes(d: ModelDims, ctx: int, concurrency: int, dtype: str) -> int:
    # ~2 hidden buffers' worth + 512 MiB scratch
    bytes_per = BYTES_PER_PARAM[dtype]
    return int(2 * concurrency * ctx * d.hidden * bytes_per) + 512 * MB


def fmt_gb(b: int) -> str:
    return f"{b / GB:6.2f} GB"


# Every key a quantization_config can use to list modules left at full
# precision. Exporters disagree on the name but agree on the meaning, so the
# union is read rather than picking one.
_EXCLUSION_KEYS = (
    "modules_to_not_convert",   # gpt-oss (mxfp4), bitsandbytes 4-bit, some GPTQ
    "ignore",                   # compressed-tensors
    "ignored_layers",           # TensorRT Model Optimizer
    "exclude_modules",          # AutoRound
    "exclude",                  # some AWQ exports
    "llm_int8_skip_modules",    # bitsandbytes int8
)

# Canonical module names used to ask "would this component be quantized?".
# They are probes, not a claim about any one checkpoint's naming: an
# exclusion entry is matched against every probe for a component.
_COMPONENT_PROBES = {
    "attn": (
        "model.layers.0.self_attn.q_proj", "model.layers.0.self_attn.k_proj",
        "model.layers.0.self_attn.v_proj", "model.layers.0.self_attn.o_proj",
        "model.layers.0.attention.wqkv",
    ),
    "state": (
        "model.layers.0.linear_attn.in_proj_qkvz",
        "model.layers.0.linear_attn.out_proj",
        "model.layers.0.mamba.in_proj", "model.layers.0.mamba.out_proj",
        "model.layers.0.mixer.in_proj",
    ),
    "router": (
        "model.layers.0.mlp.gate", "model.layers.0.mlp.router",
        "model.layers.0.block_sparse_moe.gate", "model.layers.0.mlp.gate.wg",
    ),
    # The three FFN kinds are probed separately: a checkpoint that quantizes
    # its routed experts and exempts the dense and shared ones is the common
    # case, not the exception, and nearly all the weight is in the experts.
    "dense_ffn": (
        "model.layers.0.mlp.down_proj", "model.layers.0.mlp.gate_proj",
        "model.layers.0.mlp.up_proj", "model.layers.0.feed_forward.w1",
    ),
    "shared_ffn": (
        "model.layers.0.mlp.shared_experts.down_proj",
        "model.layers.0.mlp.shared_expert.down_proj",
        "model.layers.0.mlp.shared_mlp.down_proj",
    ),
    "expert_ffn": (
        "model.layers.0.mlp.experts.0.down_proj",
        "model.layers.0.mlp.experts.down_proj",
        "model.layers.0.mlp.experts.gate_up_proj",
        "model.layers.0.block_sparse_moe.experts.0.w1",
    ),
    "vision": (
        "visual.blocks.0.attn.qkv", "visual.blocks.0.mlp.down_proj",
        "vision_tower.vision_model.encoder.layers.0.self_attn.q_proj",
        "vision_model.encoder.layers.0.mlp.fc1",
    ),
    "lm_head": ("lm_head", "language_model.lm_head"),
}


def _pattern_matches(pattern: str, name: str) -> bool:
    """Match one exclusion entry against a module name.

    Three spellings are in use and all three have to work, because a
    checkpoint that names attention in a form this misses gets charged the
    quantized rate for tensors that are stored at full width:

    * `re:` prefix -- compressed-tensors. The rest is a regular expression
      applied with `re.match`, so it is anchored at the start of the name.
    * shell globs -- `model.layers.*.self_attn`, common in
      modules_to_not_convert. A pattern naming a parent also covers its
      children, so the glob is retried with a `.*` suffix.
    * anything else -- a plain substring test.
    """
    if pattern.startswith("re:"):
        try:
            return re.match(pattern[3:], name) is not None
        except re.error:
            return False
    if "*" in pattern or "?" in pattern or "[" in pattern:
        return (fnmatch.fnmatchcase(name, pattern)
                or fnmatch.fnmatchcase(name, pattern.rstrip(".") + ".*"))
    return pattern in name


def exclusion_patterns(qcfg: dict) -> list[str]:
    """Every module pattern a quantization_config leaves at full precision."""
    patterns: list[str] = []
    for key in _EXCLUSION_KEYS:
        value = qcfg.get(key)
        if isinstance(value, str):
            patterns.append(value)
        elif isinstance(value, (list, tuple)):
            patterns.extend(str(entry) for entry in value)
    return patterns


def _component_excluded(patterns: list[str], component: str) -> bool:
    """True if any exclusion pattern covers this component's modules."""
    probes = _COMPONENT_PROBES[component]
    return any(_pattern_matches(p, probe) for p in patterns for probe in probes)


def _lm_head_quantized(qcfg: dict) -> bool:
    """True only if the config asks for the output projection to be quantized.

    The default is the other way round: AWQ and GPTQ carry an explicit
    `lm_head: true` when they want it, and compressed-tensors has to name it
    in a config group's `targets`.
    """
    if qcfg.get("lm_head") is True:
        return True
    groups = qcfg.get("config_groups")
    if isinstance(groups, dict):
        for group in groups.values():
            if not isinstance(group, dict):
                continue
            targets = group.get("targets") or []
            if isinstance(targets, str):
                targets = [targets]
            for target in targets:
                if any(_pattern_matches(str(target), probe)
                       for probe in _COMPONENT_PROBES["lm_head"]):
                    return True
    return False


def _key_from_bits(bits: object, dtype: object = "") -> str | None:
    """Map a declared weight bit-width onto a BYTES_PER_PARAM key."""
    try:
        width = int(bits)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    text = str(dtype or "").lower()
    floating = any(token in text for token in
                   ("float", "fp8", "fp4", "nvfp", "mxfp", "e4m3", "e5m2"))
    if width == 16:
        return "bf16"
    if width == 8:
        return "fp8" if floating else "int8"
    if width == 4:
        return "mxfp4" if floating else "int4"
    if width == 3:
        return "int3"
    if width == 2:
        return "int2"
    return None


def detect_quant(cfg: dict) -> str | None:
    """Weight format a pre-quantized checkpoint declares, or None.

    Reading `quant_method` alone misses most checkpoints, because the common
    values ("compressed-tensors", "awq", "gptq", "bitsandbytes") name the
    container rather than the width. The width itself is declared elsewhere,
    so it is read from wherever the exporter put it.
    """
    qcfg = cfg.get("quantization_config")
    if not isinstance(qcfg, dict):
        return None

    method = str(qcfg.get("quant_method", "")).lower()
    if method in BYTES_PER_PARAM:
        return method

    if qcfg.get("load_in_4bit"):
        return "int4"
    if qcfg.get("load_in_8bit"):
        return "int8"

    key = _key_from_bits(
        qcfg.get("bits") or qcfg.get("weight_bits") or qcfg.get("w_bit"),
        qcfg.get("weight_dtype") or qcfg.get("data_type") or method,
    )
    if key:
        return key

    # compressed-tensors declares the width per config group.
    groups = qcfg.get("config_groups")
    if isinstance(groups, dict):
        found = []
        for group in groups.values():
            weights = group.get("weights") if isinstance(group, dict) else None
            if isinstance(weights, dict):
                key = _key_from_bits(weights.get("num_bits"), weights.get("type"))
                if key:
                    found.append(key)
        if found:
            # Widest declared width wins, so a mixed export is never charged
            # less than its largest tensors actually cost.
            return max(found, key=lambda name: BYTES_PER_PARAM[name])

    # Model Optimizer and TensorRT exports name the recipe, not the width.
    algo = str(qcfg.get("quant_algo") or qcfg.get("fmt") or "").lower()
    for token, key in (("fp8", "fp8"), ("nvfp4", "mxfp4"), ("mxfp4", "mxfp4"),
                       ("w4a16", "int4"), ("int4", "int4"),
                       ("w8a8", "int8"), ("int8", "int8")):
        if token in algo:
            return key
    return None


def calculate_mixed_precision_weights(cfg: dict, d: ModelDims, params: int,
                                       quant: str, tp: int) -> tuple[int, dict | None]:
    """Calculate weight bytes accounting for mixed-precision quantization.

    No production quantization scheme converts every tensor. The embedding
    table is always left alone, output projections usually are, and many
    checkpoints also exempt attention or the router
    (quantization_config.ignore, modules_to_not_convert, and four other
    spellings of the same list). Charging every parameter at the quantized
    rate therefore under-estimates weights, which is the dangerous
    direction: it reports FITS for a launch that runs out of memory.

    Returns:
        (weights_bytes, breakdown_dict or None)
    """
    quant_bpp = BYTES_PER_PARAM[quant]
    full_bpp = 2.0

    # Both engines tensor-parallel the vision tower by default (vLLM's
    # mm_encoder_tp_mode defaults to "weights"), so it is divided like the
    # rest. The exception is a tower whose head count does not divide TP:
    # there is no valid split, so every rank holds a full copy.
    vision_replicated = bool(
        d.vision_params and tp > 1
        and (not d.vision_num_heads or d.vision_num_heads % tp)
    )

    if quant_bpp >= full_bpp:
        # Nothing is quantized, so there is no mix to break down.
        weights = params * quant_bpp / tp
        if vision_replicated:
            weights += d.vision_params * quant_bpp * (1 - 1 / tp)
        return int(weights), None

    qcfg = cfg.get("quantization_config", {})
    if not isinstance(qcfg, dict):
        qcfg = {}
    patterns = exclusion_patterns(qcfg)

    # Embedding tables are structurally out of reach of every scheme here:
    # compressed-tensors, AWQ, GPTQ and mxfp4 all target Linear modules, and
    # nn.Embedding is not one. No config will ever list the embedding table,
    # so this is an invariant rather than something to probe for.
    keep_embeddings = True
    keep_lm_head = (_component_excluded(patterns, "lm_head")
                    or not _lm_head_quantized(qcfg))
    keep_attn = _component_excluded(patterns, "attn")
    keep_state = _component_excluded(patterns, "state")
    keep_router = _component_excluded(patterns, "router")
    keep_dense_ffn = _component_excluded(patterns, "dense_ffn")
    keep_shared_ffn = _component_excluded(patterns, "shared_ffn")
    keep_expert_ffn = _component_excluded(patterns, "expert_ffn")
    keep_vision = _component_excluded(patterns, "vision")

    h = d.hidden
    vocab = d.vocab

    embed_params = vocab * h
    head_params = 0 if d.tied else vocab * h

    # Attention blocks, over attention layers only -- a hybrid's recurrent
    # layers have no Q/K/V/O projections to quantize.
    q_proj_dim = d.num_attn_heads * d.head_dim
    kv_proj_dim = d.num_kv_heads * d.head_dim
    attn_per_layer = (
        h * q_proj_dim +      # Q
        h * kv_proj_dim +     # K
        h * kv_proj_dim +     # V
        q_proj_dim * h +      # O
        4 * h                 # norms
    )
    attn_layers = d.num_full_attn_layers + d.num_sliding_attn_layers or d.num_layers
    attn_params = attn_layers * attn_per_layer

    state_params = d.num_recurrent_layers * _recurrent_block_params(d)

    # Router params (small, approximate as 2 * hidden per layer for MoE)
    router_params = d.num_layers * 2 * h if d.is_moe else 0

    dense_ffn_params, shared_ffn_params, expert_ffn_params = _ffn_param_split(d)

    # Whatever the analytic counts above do not account for (norms, biases, a
    # family this script models approximately) is charged with the largest FFN
    # bucket so the components still sum to the total parameter count.
    accounted = (embed_params + head_params + attn_params + state_params
                 + router_params + d.vision_params + dense_ffn_params
                 + shared_ffn_params + expert_ffn_params)
    residual = params - accounted
    if expert_ffn_params:
        expert_ffn_params = max(expert_ffn_params + residual, 0)
    else:
        dense_ffn_params = max(dense_ffn_params + residual, 0)

    components = [
        ("embed", embed_params, keep_embeddings),
        ("head", head_params, keep_lm_head),
        ("attn", attn_params, keep_attn),
        ("state", state_params, keep_state),
        ("router", router_params, keep_router),
        ("dense_ffn", dense_ffn_params, keep_dense_ffn),
        ("shared_ffn", shared_ffn_params, keep_shared_ffn),
        ("expert_ffn", expert_ffn_params, keep_expert_ffn),
        ("vision", d.vision_params, keep_vision),
    ]

    breakdown: dict = {}
    total_bytes = 0.0
    for name, count, keep in components:
        bpp = full_bpp if keep else quant_bpp
        divisor = 1 if (name == "vision" and vision_replicated) else tp
        component_bytes = count * bpp * tp / divisor
        total_bytes += component_bytes
        breakdown[f"{name}_params"] = count
        breakdown[f"{name}_bytes"] = int(count * bpp / divisor)
        breakdown[f"{name}_bpp"] = bpp
    breakdown["vision_replicated"] = vision_replicated
    breakdown["excluded_patterns"] = patterns

    return int(total_bytes / tp), breakdown


def estimate(cfg: dict, quant: str, kv_dtype: str, ctx: int,
             concurrency: int, tp: int, runtime: str,
             device_vram_gb: float, gpu_memory_utilization: float = 1.0) -> dict:
    d = parse_dims(cfg)
    params = count_params(d) + d.vision_params

    # Calculate weights with mixed-precision support
    weights, mixed_breakdown = calculate_mixed_precision_weights(cfg, d, params, quant, tp)

    # KV cache is sharded by attention heads (divided by TP)
    # Each GPU stores KV cache only for its subset of heads
    in_flight = max_in_flight_tokens(device_vram_gb)
    kv = kv_bytes(d, ctx, concurrency, kv_dtype, runtime, in_flight) // tp

    # Recurrent state is allocated per running request and does not shrink
    # with context, so it is charged alongside the KV cache rather than
    # inside it. It is already per-rank: state_page_bytes applies the shard.
    state_page = state_page_bytes(d, tp, runtime)
    state = state_page * concurrency

    act = activation_bytes(d, ctx, concurrency,
                           quant if quant in ("bf16", "fp16") else "bf16")
    framework = int(FRAMEWORK_OVERHEAD_GB[runtime] * GB)
    total = weights + kv + state + act + framework
    device_b = int(device_vram_gb * GB)
    usable_b = int(device_b * gpu_memory_utilization)
    free_for_kv = usable_b - weights - act - framework
    # Base footprint (weights + act + framework) can exceed usable VRAM, in
    # which case free_for_kv is negative and there is no room for any KV.
    # Clamp to 0 here so callers (TP sweep, single-result print) see a
    # consistent "no headroom" signal instead of a negative-divided value.
    if free_for_kv <= 0:
        max_concurrency = 0
        max_context = 0
    else:
        per_request = kv_bytes(d, ctx, 1, kv_dtype, runtime, in_flight) // tp
        max_concurrency = free_for_kv // max(per_request + state_page, 1)
        max_context = _max_context(d, kv_dtype, max(concurrency, 1), tp,
                                   runtime, in_flight,
                                   free_for_kv - state_page * max(concurrency, 1))
    return {
        "dims": d,
        "params": params,
        "weights": weights,
        "kv": kv,
        "state": state,
        "state_page": state_page,
        "state_slots": concurrency if state_page else 0,
        "in_flight_tokens": in_flight,
        "sliding_ctx": sliding_ctx(d, ctx, runtime, in_flight)
                       if d.num_sliding_attn_layers else 0,
        "act": act,
        "framework": framework,
        "total": total,
        "device_vram": device_b,
        "usable_vram": usable_b,
        "fits": total <= usable_b,
        "headroom": usable_b - total,
        "max_concurrency": int(max_concurrency),
        "max_context": int(max_context),
        "mixed_breakdown": mixed_breakdown,
    }


def _max_context(d: ModelDims, kv_dtype: str, concurrency: int, tp: int,
                 runtime: str, in_flight: int, budget: int) -> int:
    """Largest context whose KV cache fits in `budget` bytes.

    Solved by bisection rather than dividing by a per-token cost, because on
    a sliding-window model the cost stops growing once the window is full --
    dividing would report a ceiling far below the real one.
    """
    if budget <= 0:
        return 0
    low, high = 0, 1 << 24
    if kv_bytes(d, high, concurrency, kv_dtype, runtime, in_flight) // tp <= budget:
        return high
    while low < high - 1:
        mid = (low + high) // 2
        if kv_bytes(d, mid, concurrency, kv_dtype, runtime, in_flight) // tp <= budget:
            low = mid
        else:
            high = mid
    return low


def verdict_cell(result: dict, usable_vram_gb: float) -> str:
    if result["fits"]:
        headroom_gb = result["headroom"] / GB
        if headroom_gb < max(1.0, 0.10 * usable_vram_gb):
            return "tight"
        return "fits"
    parts = [
        ("weights", result["weights"]),
        ("KV", result["kv"]),
        ("state", result["state"]),
        ("activations", result["act"]),
        ("framework", result["framework"]),
    ]
    binding = max(parts, key=lambda x: x[1])[0]
    return f"OOM ({binding})"


def print_table(models: list[str], runtime: str, device_vram_gb: float,
                gpu_memory_utilization: float, revision: str) -> int:
    scenarios = [
        ("bf16 / 8K / c=1", "bf16", "bf16", 8192, 1, 1),
        ("bf16 / 4K / c=4", "bf16", "bf16", 4096, 4, 1),
        ("int4 / 32K / c=4", "int4", "fp8", 32768, 4, 1),
    ]
    usable_vram_gb = device_vram_gb * gpu_memory_utilization
    print(f"Runtime: {runtime}, device VRAM: {device_vram_gb:.2f} GB "
          f"(usable {usable_vram_gb:.2f} GB at "
          f"gpu_memory_utilization={gpu_memory_utilization:g})")
    print()
    header = ["Model"] + [s[0] for s in scenarios]
    rows = []
    for model in models:
        cfg = fetch_config(model, revision)
        row = [model.rsplit("/", 1)[-1]]
        for _, quant, kv_dtype, ctx, concurrency, tp in scenarios:
            result = estimate(cfg, quant, kv_dtype, ctx, concurrency,
                              tp, runtime, device_vram_gb,
                              gpu_memory_utilization)
            row.append(verdict_cell(result, usable_vram_gb))
        rows.append(row)

    widths = [max(len(str(x)) for x in col)
              for col in zip(header, *rows)]
    print(" | ".join(str(x).ljust(w) for x, w in zip(header, widths)))
    print("-|-".join("-" * w for w in widths))
    for row in rows:
        print(" | ".join(str(x).ljust(w) for x, w in zip(row, widths)))
    return 0


def parse_tp_sweep(value: str | None) -> list[int]:
    # Sort ascending so the verdict line ("smallest TP that fits") cannot
    # contradict the table when the user passes values out of order
    # (e.g. --tp-sweep 4,2,1). Smallest fit is the cheapest deployment.
    if not value:
        return []
    out: set[int] = set()
    for part in value.split(","):
        part = part.strip()
        if not part:
            continue
        try:
            tp = int(part)
        except ValueError as exc:
            raise argparse.ArgumentTypeError("--tp-sweep must be comma-separated integers") from exc
        if tp < 1:
            raise argparse.ArgumentTypeError("--tp-sweep values must be >= 1")
        out.add(tp)
    if not out:
        raise argparse.ArgumentTypeError("--tp-sweep must include at least one TP value")
    return sorted(out)


def print_tp_sweep(cfg: dict, args: argparse.Namespace, tp_values: list[int]) -> None:
    rows = []
    first_fit = None
    for tp in tp_values:
        result = estimate(cfg, args.quant, args.kv_dtype, args.ctx,
                          args.concurrency, tp, args.runtime,
                          args.device_vram_gb, args.gpu_memory_utilization)
        rows.append((
            tp,
            result["weights"],
            result["kv"],
            result["state"],
            result["act"],
            result["framework"],
            result["total"],
            result["headroom"],
            result["fits"],
            result["max_concurrency"],
            result["max_context"],
        ))
        if result["fits"] and first_fit is None:
            first_fit = tp

    # The state column is dead weight for the overwhelmingly common case of a
    # model with no recurrent layers, so only show it when there is state.
    show_state = any(row[3] for row in rows)

    print()
    print("TP sweep")
    state_head = f"{'state':>9} " if show_state else ""
    print(f"  {'TP':>2}  {'weights':>9} {'KV':>9} {state_head}{'act':>9} "
          f"{'fw':>9} {'total':>9} {'headroom':>9} "
          f"{'fit':>3}  {'max_c':>5}  {'max_ctx':>7}")
    print("  " + "-" * (93 if show_state else 83))
    for (tp, weights, kv, state, act, framework, total, headroom,
         fits, max_c, max_ctx) in rows:
        verdict = "YES" if fits else "NO"
        state_cell = f"{fmt_gb(state)} " if show_state else ""
        print(f"  {tp:>2}  {fmt_gb(weights)} {fmt_gb(kv)} {state_cell}{fmt_gb(act)} "
              f"{fmt_gb(framework)} {fmt_gb(total)} {fmt_gb(headroom)} "
              f"{verdict:>3}  {max_c:>5}  {max_ctx:>7}")
    if first_fit is None:
        print("TP sweep verdict: no requested TP fits.")
    else:
        print(f"TP sweep verdict: smallest requested TP that fits = {first_fit}")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", help="HF model id or path to config.json")
    p.add_argument("--table", action="store_true",
                   help="Print a quick verdict table for common public LLMs.")
    p.add_argument("--table-model", action="append", default=[],
                   help="Model id to include with --table. May be repeated.")
    p.add_argument("--quant", default=None, choices=list(BYTES_PER_PARAM) + [None],
                   help="Quantization format. If not specified, auto-detects from model config (quantization_config.quant_method) or defaults to bf16.")
    p.add_argument("--kv-dtype", default=None, choices=list(BYTES_PER_KV) + [None])
    p.add_argument("--ctx", type=int, default=4096)
    p.add_argument("--concurrency", type=int, default=1)
    p.add_argument("--tp", type=int, default=1, help="Tensor parallel degree (divides weights+KV).")
    p.add_argument("--tp-sweep", default=None,
                   help="Comma-separated TP values to evaluate after the main verdict, e.g. 1,2,4,8.")
    p.add_argument("--runtime", default="vllm", choices=list(FRAMEWORK_OVERHEAD_GB))
    # No default: an assumed VRAM figure produces a confident, wrong
    # verdict. Guessing high is the dangerous direction -- it reports FITS
    # for a launch that OOMs at engine init. Make the caller supply a
    # measured number instead.
    p.add_argument("--device-vram-gb", type=float, default=None,
                   help="Required. Per-device VRAM in GB. Confirm the card "
                        "with `xpu-smi discovery -d <id>` rather than recalling "
                        "it from a spec sheet. Arc Pro B70 = 32, B65 = 32, "
                        "B60 = 24, B50 = 16, B580 = 12. For tensor "
                        "parallelism, pass the smallest card in the set.")
    p.add_argument("--gpu-memory-utilization", type=float, default=1.0,
                   help="Usable fraction of device VRAM for the runtime. "
                        "Use the vLLM --gpu-memory-utilization value for launch planning.")
    p.add_argument("--revision", default="main")
    args = p.parse_args(argv)

    if not (0 < args.gpu_memory_utilization <= 1.0):
        p.error("--gpu-memory-utilization must be > 0 and <= 1")
    if args.device_vram_gb is None:
        p.error(
            "--device-vram-gb is required; there is no safe default.\n"
            "  Identify the target card first -- do not use a remembered spec:\n"
            "      xpu-smi discovery -d <id> | grep -i 'Device Name\\|Memory "
            "Physical Size'\n"
            "  then pass that SKU's GB: B70 32, B65 32, B60 24, B50 16, "
            "B580 12.\n"
            "  For tensor parallelism, pass the SMALLEST card in the set.")
    # Bind before the try so the name is defined on every path. p.error()
    # raises SystemExit, but static analysis cannot see that and flags the
    # later `if tp_sweep:` reads as possibly-uninitialized.
    tp_sweep: list[int] = []
    try:
        tp_sweep = parse_tp_sweep(args.tp_sweep)
    except argparse.ArgumentTypeError as exc:
        p.error(str(exc))

    if args.table:
        models = args.table_model or TABLE_MODELS
        return print_table(models, args.runtime, args.device_vram_gb,
                           args.gpu_memory_utilization, args.revision)
    if not args.model:
        p.error("--model is required unless --table is set")

    cfg = fetch_config(args.model, args.revision)

    # Auto-detect pre-quantized models from config if --quant not specified
    auto_quant = False
    if args.quant is None:
        detected = detect_quant(cfg)
        if detected:
            args.quant = detected
            auto_quant = True
        else:
            args.quant = "bf16"

    auto_kv = args.kv_dtype is None
    if auto_kv:
        args.kv_dtype = "fp8" if args.quant in ("fp8", "int4", "int3", "int2", "mxfp4") else "bf16"
    result = estimate(cfg, args.quant, args.kv_dtype, args.ctx,
                      args.concurrency, args.tp, args.runtime,
                      args.device_vram_gb, args.gpu_memory_utilization)
    d = result["dims"]
    params = result["params"]
    bpp = BYTES_PER_PARAM[args.quant]
    weights = result["weights"]
    kv = result["kv"]
    act = result["act"]
    framework = result["framework"]
    total = result["total"]
    usable_b = result["usable_vram"]
    fits = result["fits"]
    headroom = result["headroom"]

    state = result["state"]

    arch_label = "decoder-only LLM"
    if d.is_moe:
        arch_label = "MoE"
    if d.num_recurrent_layers:
        arch_label = "hybrid SSM + attention" if (
            d.num_full_attn_layers or d.num_sliding_attn_layers
        ) else "state-space model"
        if d.is_moe:
            arch_label += ", MoE"
    elif d.num_sliding_attn_layers:
        arch_label += ", sliding-window attention"
    if d.is_vlm:
        arch_label = "VLM (LLM backbone + vision tower)"

    print(f"Model:             {args.model}")
    print(f"Architecture:      {d.arch_family}  ({arch_label})")
    if d.num_recurrent_layers or d.num_sliding_attn_layers:
        pieces = []
        if d.num_full_attn_layers:
            pieces.append(f"{d.num_full_attn_layers} full attention")
        if d.num_sliding_attn_layers:
            pieces.append(f"{d.num_sliding_attn_layers} sliding "
                          f"(window {d.sliding_window})")
        if d.num_recurrent_layers:
            pieces.append(f"{d.num_recurrent_layers} recurrent")
        print(f"Layers:            {d.num_layers} total: {', '.join(pieces)}")
        if d.attn_on_recurrent_layers:
            print("                   (attention and SSM run in the same "
                  "layer, so both are charged)")
        if d.num_recurrent_layers:
            print(f"Recurrent state:   {d.num_state_heads} head(s) x "
                  f"{d.state_head_dim_k} x {d.state_head_dim_v}, conv "
                  f"{d.conv_dim} x {d.conv_kernel - 1}  "
                  f"({result['state_page'] / MB:.1f} MiB per request)")
    if d.is_moe:
        expert_str = f"{d.num_experts} routed"
        if d.num_shared_experts:
            expert_str += f" + {d.num_shared_experts} shared"

        # Calculate active experts: routed + shared
        # Shared experts are always-on in addition to top-k routed, not instead of them
        if d.num_experts_per_tok:
            # Known: report routed + shared
            active = d.num_experts_per_tok + d.num_shared_experts
            active_str = f"~{active} active per token"
        elif d.num_shared_experts:
            # Unknown routed top-k, but shared experts present
            # Report "? + N shared" to clarify shared are in addition to unknown routed
            active_str = f"? routed + {d.num_shared_experts} shared active per token"
        else:
            # No information about active experts
            active_str = "? active per token"

        print(f"Experts:           {expert_str}, {active_str}")
        if d.first_k_dense_replace > 0:
            print(f"                   (first {d.first_k_dense_replace} layer(s) use dense FFN)")
    if d.is_vlm:
        print(f"Vision tower:      {d.vision_params/1e9:.2f} B params "
              f"(included in weights below)")
    print(f"Total parameters:  {params/1e9:.2f} B")
    print(f"Quantization:      {args.quant}  ({bpp:.2f} bytes/param)", end="")
    if auto_quant:
        print(f"  (auto-detected from config)")
    else:
        print()
    if auto_kv and args.kv_dtype != "bf16":
        print(f"KV dtype:          {args.kv_dtype}  (auto-paired with --quant {args.quant}; "
              f"override with --kv-dtype)")
    if args.tp > 1:
        print(f"Tensor parallel:   {args.tp}  (weights + KV split across devices)")
    if args.gpu_memory_utilization < 1.0:
        print(f"Memory limit:      {args.gpu_memory_utilization:.2f} of physical VRAM "
              f"(runtime allocation target)")

    # Show mixed-precision breakdown if available
    mixed_breakdown = result.get("mixed_breakdown")
    if mixed_breakdown:
        print()
        print("Mixed-precision weight breakdown:")
        for key, label in (("embed", "Embeddings"), ("head", "Output head"),
                           ("attn", "Attention"), ("state", "SSM / conv"),
                           ("router", "Routers"), ("dense_ffn", "Dense FFN"),
                           ("shared_ffn", "Shared experts"),
                           ("expert_ffn", "Routed experts"),
                           ("vision", "Vision tower")):
            # Below 5M params both columns round to 0.00 and the line is noise.
            if mixed_breakdown[f"{key}_params"] < 5_000_000:
                continue
            print(f"  {label:<15} {fmt_gb(mixed_breakdown[f'{key}_bytes'])}   "
                  f"({mixed_breakdown[f'{key}_params']/1e9:.2f}B params "
                  f"@ {mixed_breakdown[f'{key}_bpp']:.2f} B/p)")
        print(f"  -----              -----")
        print(f"  Weights total   {fmt_gb(weights)}")
        if mixed_breakdown["vision_replicated"]:
            print(f"  (vision tower replicated on every rank: its "
                  f"{d.vision_num_heads or 'unknown'} head(s) do not divide "
                  f"TP {args.tp})")

    print()
    print("VRAM breakdown")
    print(f"  Weights         {fmt_gb(weights)}")
    kv_note = (f"({args.ctx} tok x concurrency {args.concurrency}, "
               f"kv_dtype {args.kv_dtype})")
    print(f"  KV cache        {fmt_gb(kv)}   {kv_note}")
    if d.num_sliding_attn_layers and result["sliding_ctx"] < args.ctx:
        print(f"                  {d.num_sliding_attn_layers} windowed layer(s) "
              f"hold {result['sliding_ctx']} tok, not {args.ctx} "
              f"({args.runtime} accounting)")
    if state:
        print(f"  Recurrent state {fmt_gb(state)}   "
              f"({result['state_slots']} request slot(s) x "
              f"{result['state_page'] / MB:.1f} MiB; independent of context)")
    print(f"  Activations     {fmt_gb(act)}   (estimate)")
    print(f"  Framework       {fmt_gb(framework)}   ({args.runtime})")
    print(f"  -----              -----")
    print(f"  Total           {fmt_gb(total)}")
    print()
    per_device = f"  (per device, x{args.tp} TP)" if args.tp > 1 else ""
    print(f"Device VRAM:       {args.device_vram_gb:6.2f} GB{per_device}")
    if args.gpu_memory_utilization < 1.0:
        print(f"Usable VRAM:       {usable_b / GB:6.2f} GB "
              f"(gpu_memory_utilization={args.gpu_memory_utilization:.2f})")
    else:
        # Every line qualifying the VRAM figure is gated on gmu < 1.0, so
        # the default run would otherwise show a nameplate number with
        # nothing marking it optimistic. Both gaps beneath the nameplate --
        # vendor rounding and the driver's allocatable ceiling -- run the
        # same direction, so a tight FITS here is not a launchable result.
        print("Usable VRAM:       all of it  (physical-fit only; no "
              "--gpu-memory-utilization given)")
        print("Note: that is nameplate VRAM, and both gaps beneath it are "
              "optimistic. Vendors round up (a \"24 GB\" Arc Pro B60 exposes "
              "23.91 GiB, measured) and the driver's allocatable ceiling sits "
              "~5% below physical. Re-run with the --gpu-memory-utilization "
              "the runtime will actually use before trusting a tight verdict.")

    if fits:
        pct = 100 * headroom / usable_b
        print(f"Verdict:           FITS  (headroom {fmt_gb(headroom)}, {pct:.1f}%)")
        print(f"Capacity:          up to {result['max_concurrency']} concurrent "
              f"request(s) at ctx {args.ctx}, or ctx {result['max_context']} "
              f"at concurrency {args.concurrency}  (memory-only ceiling)")
        if tp_sweep:
            print_tp_sweep(cfg, args, tp_sweep)
        if d.is_vlm:
            print()
            print("Note: VLM runtime memory grows with image-token count, which "
                  "depends on the resolution and number of images per request. "
                  "The number above covers weights + text-side KV; add ~1-3 "
                  "GiB headroom per concurrent image-bearing request for "
                  "vision-encoder activations.")
        return 0

    deficit = -headroom
    print(f"Verdict:           DOES NOT FIT  (deficit {fmt_gb(deficit)})")

    parts = sorted(
        [("weights", weights), ("KV cache", kv), ("recurrent state", state),
         ("activations", act), ("framework", framework)],
        key=lambda item: -item[1],
    )
    binding, _ = parts[0]
    print(f"  Binding constraint: {binding} ({fmt_gb(parts[0][1])}) dominates")

    in_flight = result["in_flight_tokens"]

    def kv_at(ctx: int, concurrency: int, kv_dtype: str) -> int:
        return kv_bytes(d, ctx, concurrency, kv_dtype, args.runtime,
                        in_flight) // args.tp

    suggestions: list[str] = []
    if binding == "KV cache":
        if args.ctx > 1024:
            new_ctx = max(1024, args.ctx // 2)
            new_kv = kv_at(new_ctx, args.concurrency, args.kv_dtype)
            suggestions.append(
                f"drop ctx {args.ctx} -> {new_ctx} (saves {fmt_gb(kv - new_kv)})"
            )
        if args.kv_dtype in ("bf16", "fp16"):
            new_kv = kv_at(args.ctx, args.concurrency, "fp8")
            suggestions.append(
                f"kv-dtype {args.kv_dtype} -> fp8 (saves {fmt_gb(kv - new_kv)})"
            )
        if args.concurrency > 1:
            new_kv = kv_at(args.ctx, 1, args.kv_dtype)
            suggestions.append(
                f"concurrency {args.concurrency} -> 1 (saves {fmt_gb(kv - new_kv)})"
            )
    if binding == "recurrent state":
        # The state pool scales only with the number of running requests --
        # shortening the context does nothing for it.
        if args.concurrency > 1:
            new_concurrency = max(1, args.concurrency // 2)
            saved = result["state_page"] * (args.concurrency - new_concurrency)
            suggestions.append(
                f"concurrency {args.concurrency} -> {new_concurrency} "
                f"(saves {fmt_gb(saved)}; the state pool scales with running "
                f"requests, not context)"
            )
        if args.tp == 1:
            suggestions.append(
                f"--tp 2 shards the state pool across two XPUs (saves about "
                f"{fmt_gb(state - state_page_bytes(d, 2, args.runtime) * args.concurrency)})"
            )
    if binding == "weights":
        if args.quant in ("bf16", "fp16"):
            for q in ("fp8", "int4", "int3"):
                new_w = int(params * BYTES_PER_PARAM[q] / args.tp)
                suggestions.append(
                    f"quant {args.quant} -> {q} (saves {fmt_gb(weights - new_w)})"
                )
        if args.tp == 1:
            suggestions.append(
                f"--tp 2 splits weights across two XPUs (saves {fmt_gb(weights // 2)})"
            )
    if not suggestions:
        suggestions.append(
            "model is too big for this device at any reasonable setting; "
            "try a smaller model or more XPUs (--tp)"
        )
    # Always show the TP option for weights-bound cases, even past the cap.
    cap = 4 if (binding == "weights" and args.tp == 1) else 3
    print(f"  Try first: {' or '.join(suggestions[:cap])}")
    if tp_sweep:
        print_tp_sweep(cfg, args, tp_sweep)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
