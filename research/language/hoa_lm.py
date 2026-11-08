"""Shared pieces of the GPT backbone: configuration defaults, initialisation, LayerNorm, dense layers and matrix
products with optional bfloat16 inputs.

Only the 'gpt' architecture is built here (model.config always selects it; the join and other branches are added by
model.init): token and learned position embeddings (std 0.02), and per layer two LayerNorms, a fused QKV projection,
an output projection and a two-layer MLP, the output projection and the MLP's second matrix scaled by 1/sqrt(2L);
then a final LayerNorm. model.backbone removes the position table when rotary positions are used.
"""
import argparse
import math

import jax
import jax.numpy as jnp


def dense(key, i, o, scale=1.0):
    return {"w": jax.random.normal(key, (i, o)) * scale / math.sqrt(i), "b": jnp.zeros(o)}


MM = None  # set to jnp.bfloat16 by the worker: matmuls in bf16 with float32 accumulation


def mm(a, b, spec=None):
    if MM is None:
        return jnp.einsum(spec, a, b) if spec else a @ b
    a, b = a.astype(MM), b.astype(MM)
    if spec:
        return jnp.einsum(spec, a, b, preferred_element_type=jnp.float32)
    return jnp.matmul(a, b, preferred_element_type=jnp.float32)


def lin(p, x):
    return mm(x, p["w"]) + p["b"]


def ln_init(d):
    return {"g": jnp.ones(d), "b": jnp.zeros(d)}


def ln(p, x):
    mu = x.mean(-1, keepdims=True)
    var = ((x - mu) ** 2).mean(-1, keepdims=True)
    return (x - mu) / jnp.sqrt(var + 1e-5) * p["g"] + p["b"]


def init(key, cfg):
    if cfg.arch != "gpt":
        raise ValueError("only the gpt backbone is built here")
    D, L, T = cfg.d, cfg.layers, cfg.ctx
    ks = iter(jax.random.split(key, 16 + 16 * L))
    p = {"emb": jax.random.normal(next(ks), (cfg.vocab, D)) * 0.02, "pos": jax.random.normal(next(ks), (T, D)) * 0.02,
         "lnf": ln_init(D), "layers": []}
    for _ in range(L):
        lyr = {"ln1": ln_init(D), "ln2": ln_init(D)}
        lyr["qkv"] = dense(next(ks), D, 3 * D)
        lyr["o"] = dense(next(ks), D, D, scale=1 / math.sqrt(2 * L))
        lyr["fc1"] = dense(next(ks), D, cfg.mlp)
        lyr["fc2"] = dense(next(ks), cfg.mlp, D, scale=1 / math.sqrt(2 * L))
        p["layers"].append(lyr)
    return p


def project_qkv(lyr, x):
    """Fused Q/K/V projection, split into three."""
    return jnp.split(lin(lyr["qkv"], x), 3, -1)


def build_parser():
    """Defaults of the configuration namespace; model.config overrides every field it uses."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", default="gpt")
    ap.add_argument("--d", type=int, default=256)
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--mlp", type=int, default=0)
    ap.add_argument("--ctx", type=int, default=256)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--vocab", type=int, default=256)
    ap.add_argument("--bf16", type=int, default=0)
    return ap
