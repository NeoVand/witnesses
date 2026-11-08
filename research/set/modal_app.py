"""Modal app for the SET runs: modal deploy modal_app.py, then a spawner (spawn2.py, spawn3.py, spawn8.py,
spawn_nset.py, newton_paper/spawn_cost.py, expnewton/spawn_exp.py). Each job runs one script on one A10G and writes to
the volume, so results outlive the client; the job's source hashes must match the image."""
import hashlib, json, os, subprocess, sys, time
from pathlib import Path
import modal
HERE = Path(__file__).resolve().parent
app = modal.App("witnesses-set")
volume = modal.Volume.from_name("witnesses-set", create_if_missing=True)
image = modal.Image.debian_slim(python_version="3.12").pip_install("jax[cuda12]==0.8.0", "optax==0.2.6", "numpy==2.3.4")
for name in ("bench2.py", "bench3.py", "bench4.py", "bench8.py", "bench_nset.py", "newton_chars.py", "newton_paper/cost_bench.py",
             "operators.py", "tasks.py", "hoa_polybench.py", "groups.py"):
    image = image.add_local_file(HERE / name, "/root/conn/" + name)
for name in ("bench_exp.py", "diag_exp.py"):                     # exponential Newton, flat beside the bench modules
    image = image.add_local_file(HERE / "expnewton" / name, "/root/conn/" + name)


@app.function(image=image, gpu="A10G", cpu=2, memory=32768, timeout=3550, max_containers=14, scaledown_window=2, retries=0, volumes={"/vol": volume})
def run(job):
    for name, digest in job["sources"].items():
        if hashlib.sha256(Path("/root/conn", name).read_bytes()).hexdigest() != digest:
            raise RuntimeError("stale image: " + name)
    job = dict(job, dest="/vol/" + job["root"] + "/" + job["id"])
    Path(job["dest"]).mkdir(parents=True, exist_ok=True)
    t = time.time()
    p = subprocess.run([sys.executable, "/root/conn/" + job["script"], json.dumps(job)], capture_output=True, text=True, cwd="/root/conn", env=dict(os.environ, PYTHONPATH="/root/conn", XLA_PYTHON_CLIENT_MEM_FRACTION=".85"), timeout=3500)
    Path(job["dest"], "done.json").write_text(json.dumps(dict(returncode=p.returncode, wall=time.time() - t, stdout=p.stdout[-4000:], stderr=p.stderr[-3000:])))
    volume.commit()
    return p.returncode
