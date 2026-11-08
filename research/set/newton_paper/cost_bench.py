"""Cost of one layer: softmax attention, the full join (2-simplicial attention over all ordered pairs) and Newton
attention (order 3), at width D = 96 and rank 64 as in the SET bench. Forward + backward of mean(y^2) with respect to
the layer's weights and its input, jitted. Each (method, n, batch) cell runs in a fresh process so the peak-memory
counter is its own. Batch B = 16,384 // n sequences (same tokens per step); a cell that runs out of memory is
recorded and retried once with B = 1.

Job (via modal_app.run): {"script": "newton_paper/cost_bench.py", "ns": [...], "methods": [...]}
Cell: python cost_bench.py --cell <method> <n> <B>
"""
import argparse, json, math, os, subprocess, sys, time
from pathlib import Path

TOKENS, D, R = 16384, 96, 64


def cell(method, n, B):
    import numpy as np, jax, jax.numpy as j
    import bench2
    import hoa_polybench as legacy
    keys = iter(jax.random.split(jax.random.PRNGKey(0), 16))
    mat = lambda r, c: jax.random.normal(next(keys), (r, c)) / r ** .5
    if method == "softmax":
        p = dict(q=mat(D, R), k=mat(D, R), v=mat(D, R), o=mat(R, D))

        def layer(p, h):
            s = (h @ p["q"]) @ (h @ p["k"]).swapaxes(-1, -2) / R ** .5
            return jax.nn.softmax(s, -1) @ (h @ p["v"]) @ p["o"]
    elif method == "join":
        p = {k: mat(D, R) for k in ("q", "k1", "k2", "v1", "v2")} | dict(o=mat(R, D))
        layer = lambda p, h: bench2.cubic(p, h, dict(kind="cubic"))          # the paper's join layer, as trained
    else:
        p = dict(th=dict(w=mat(D, R), b=j.zeros(R)), g=dict(w=mat(D, R), b=j.zeros(R)), o=dict(w=mat(6 * R, D), b=j.zeros(D)))
        c = argparse.Namespace(model="newton", heads=1, d=D, rank=R, order=3, phase=2., bosfix=1, fnorm=1, nfloor=1e-6, readout="full")
        layer = lambda p, h: legacy.attention(c, p, h)                    # the bench's Newton layer (newton_bare's)
    h = jax.random.normal(next(keys), (B, n, D))
    f = jax.jit(jax.value_and_grad(lambda p, h: (layer(p, h) ** 2).mean(), argnums=(0, 1)))
    row = dict(method=method, n=n, B=B, tokens=n * B, params=int(sum(w.size for w in jax.tree.leaves(p))))
    try:
        compiled = f.lower(p, h).compile()
        m = compiled.memory_analysis()
        row["compiled_bytes"] = dict(temp=int(m.temp_size_in_bytes), arguments=int(m.argument_size_in_bytes), outputs=int(m.output_size_in_bytes))
        jax.block_until_ready(f(p, h)); jax.block_until_ready(f(p, h))
        times = []
        for _ in range(10):
            t0 = time.perf_counter()
            jax.block_until_ready(f(p, h))
            times.append(time.perf_counter() - t0)
        row["ms_median"] = 1000 * float(np.median(times))
        row["ms_min"] = 1000 * float(np.min(times))
        st = jax.devices()[0].memory_stats() or {}
        row["peak_bytes"] = int(st.get("peak_bytes_in_use", -1))
        row["bytes_limit"] = int(st.get("bytes_limit", -1))
        row["status"] = "ok"
    except Exception as e:                                               # out of memory, or anything else
        msg = str(e)
        row["status"] = "oom" if ("RESOURCE_EXHAUSTED" in msg or "out of memory" in msg.lower() or "Out of memory" in msg) else "error"
        row["error"] = msg[:300]
    print(json.dumps(row), flush=True)


def main(job):
    out = Path(job["dest"])
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for method in job["methods"]:
        for n in job["ns"]:
            for B in (max(1, TOKENS // n), 1):
                try:
                    r = subprocess.run([sys.executable, __file__, "--cell", method, str(n), str(B)], capture_output=True, text=True, timeout=600,
                                       cwd="/root/conn", env=os.environ)
                    lines = [l for l in r.stdout.splitlines() if l.startswith("{")]
                    row = json.loads(lines[-1]) if lines else dict(method=method, n=n, B=B, status="crash", returncode=r.returncode, error=r.stderr[-300:])
                except subprocess.TimeoutExpired:
                    row = dict(method=method, n=n, B=B, status="timeout")
                rows.append(row)
                with (out / "cost.jsonl").open("a") as f:
                    f.write(json.dumps(row) + "\n")
                print(json.dumps({k: row.get(k) for k in ("method", "n", "B", "status", "ms_median", "peak_bytes")}), flush=True)
                if row["status"] == "ok" or B == 1:
                    break
    import jax
    (out / "device.json").write_text(json.dumps(dict(device=str(jax.devices()[0].device_kind), jax=jax.__version__,
                                                       mem_fraction=os.environ.get("XLA_PYTHON_CLIENT_MEM_FRACTION"))))


if __name__ == "__main__":
    if sys.argv[1] == "--cell":
        cell(sys.argv[2], int(sys.argv[3]), int(sys.argv[4]))
    else:
        main(json.loads(sys.argv[1]))
