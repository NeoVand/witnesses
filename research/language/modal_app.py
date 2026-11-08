"""Modal deployment for the language-model and chess runs. Work goes through deploy, spawn and the volume, so it does
not depend on the launching machine staying connected.

    modal deploy modal_app.py
    python spawn.py <job list>          # see spawn.py

One volume, witnesses-language, mounted at /vol: /vol/fineweb_edu (tokens), /vol/data-v1 (held-out sets) and
/vol/results/<job id> (each job's result.json, rows.json, test.npz and saved parameters).
"""
from pathlib import Path
import modal
ROOT = Path(__file__).resolve().parent
app = modal.App('witnesses-language')
vol = modal.Volume.from_name('witnesses-language', create_if_missing=True)
image = modal.Image.debian_slim(python_version='3.12').pip_install('jax[cuda12]==0.8.0', 'optax==0.2.6', 'numpy==2.3.4', 'tiktoken==0.12.0')
for name in ('model.py', 'cubics.py', 'worker.py', 'hoa_lm.py', 'chess_train.py', 'chess_train2.py'):
    image = image.add_local_file(ROOT / name, '/root/lab/' + name, copy=True)


cpu_image = (modal.Image.debian_slim(python_version='3.12')
             .pip_install('chess==1.11.2', 'zstandard==0.23.0', 'numpy==2.3.4', 'tiktoken', 'pyarrow', 'huggingface_hub[hf_transfer]')
             .env({'HF_HUB_ENABLE_HF_TRANSFER': '1'}))
for name in ('chess_data.py', 'prepare.py'):
    cpu_image = cpu_image.add_local_file(ROOT / name, '/root/lab/' + name, copy=True)


@app.function(image=cpu_image, cpu=16, memory=65536, timeout=7200, retries=0, volumes={'/vol': vol})
def run_cpu(job):
    """Data preparation: prepare.py (FineWeb-Edu tokens and held-out sets) or chess_data.py (Lichess puzzles)."""
    import json, os, subprocess, sys, time
    vol.reload()
    dest = Path('/vol/results') / job['id']
    dest.mkdir(parents=True, exist_ok=True)
    (dest / 'started.json').write_text(json.dumps(job))
    begin = time.monotonic()
    env = dict(os.environ, PYTHONPATH='/root/lab', LAB_DEST=str(dest), PYTHONDONTWRITEBYTECODE='1')
    with (dest / 'stdout.log').open('w') as so, (dest / 'stderr.log').open('w') as se:
        r = subprocess.run([sys.executable, '/root/lab/' + job['script'], json.dumps(job)], env=env, stdout=so, stderr=se, timeout=job['timeout'])
    out = dict(job=job, returncode=r.returncode, wall_seconds=time.monotonic() - begin)
    if r.returncode:
        out['error'] = (dest / 'stderr.log').read_text()[-4000:]
    else:
        out['result'] = json.loads((dest / 'stdout.log').read_text().splitlines()[-1])
    (dest / 'result.json').write_text(json.dumps(out))
    vol.commit()
    return r.returncode


@app.function(image=image, gpu='H100', cpu=2, memory=32768, timeout=14000, max_containers=44, scaledown_window=2, retries=0, volumes={'/vol': vol})
def run(job):
    """One training job (worker.py, chess_train.py or chess_train2.py) on one H100. Refuses to run if the image's
    sources differ from the hashes the spawner recorded in job['sources']."""
    import hashlib, json, os, subprocess, sys, time
    vol.reload()
    dest = Path('/vol/results') / job['id']
    dest.mkdir(parents=True, exist_ok=True)
    with (dest / 'started.json').open('x') as f:
        json.dump(job, f)
    vol.commit()
    begin = time.monotonic()
    out = dict(job=job)
    try:
        for name, digest in job['sources'].items():
            assert hashlib.sha256(Path('/root/lab', name).read_bytes()).hexdigest() == digest, 'stale image: ' + name
        env = dict(os.environ, PYTHONPATH='/root/lab', LAB_DEST=str(dest), PYTHONDONTWRITEBYTECODE='1', XLA_PYTHON_CLIENT_PREALLOCATE='false')
        script = job.get('script', 'worker.py')
        with (dest / 'stdout.log').open('w') as so, (dest / 'stderr.log').open('w') as se:
            r = subprocess.run([sys.executable, '/root/lab/' + script, json.dumps(job)], env=env, stdout=so, stderr=se, timeout=job['timeout'])
        out['returncode'] = r.returncode
        if r.returncode:
            out['error'] = (dest / 'stderr.log').read_text()[-4000:]
        else:
            out['result'] = json.loads((dest / 'stdout.log').read_text().splitlines()[-1])
    except Exception as e:
        out['error'] = type(e).__name__ + ': ' + str(e)
    out['wall_seconds'] = time.monotonic() - begin
    (dest / 'result.json').write_text(json.dumps(out))
    vol.commit()
    return out.get('returncode')
