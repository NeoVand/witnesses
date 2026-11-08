"""Modal app witnesses-clrs. Deploy, then spawn jobs with spawn_clrs.py; everything lands on the witnesses-clrs volume
under clrs/: data/<task>/*.npz from DeepMind's library (gen), runs/<id>/ from training.
gen runs dm-clrs 2.0.3 (numpy 2.5.4, the versions used to regenerate the splits locally for scoring) on CPU; train runs
our JAX code (clrs_train.py) on one NVIDIA L4 and never imports the library.
modal deploy modal_clrs.py"""
from pathlib import Path
import modal
HERE = Path(__file__).resolve().parent
app = modal.App('witnesses-clrs')
vol = modal.Volume.from_name('witnesses-clrs', create_if_missing=True)
gen_image = (modal.Image.debian_slim(python_version='3.12').pip_install('dm-clrs==2.0.3', 'numpy==2.5.4')
             .add_local_file(HERE / 'gen_data.py', '/root/clrs/gen_data.py', copy=True))
train_image = (modal.Image.debian_slim(python_version='3.12').pip_install('jax[cuda12]==0.8.0', 'optax==0.2.6', 'numpy==2.3.4')
               .add_local_file(HERE / 'clrs_train.py', '/root/clrs/clrs_train.py', copy=True))


@app.function(image=gen_image, cpu=2, memory=8192, timeout=3600, max_containers=40, scaledown_window=2, retries=0, volumes={'/vol': vol})
def gen(task, what, per_size=80000):
    import json, subprocess, sys
    r = subprocess.run([sys.executable, '-I', '/root/clrs/gen_data.py', '/vol/clrs/data', task, what, str(per_size)], capture_output=True, text=True)
    vol.commit()
    if r.returncode:
        return dict(task=task, what=what, error=r.stderr[-3000:])
    return json.loads(r.stdout.splitlines()[-1])


@app.function(image=train_image, gpu='L4', cpu=4, memory=16384, timeout=14400, max_containers=36, scaledown_window=2, retries=0, volumes={'/vol': vol})
def train(job):
    import json, os, subprocess, sys, time
    vol.reload()
    dest = Path('/vol/clrs/runs') / job['id']
    dest.mkdir(parents=True, exist_ok=True)
    (dest / 'started.json').write_text(json.dumps(job))
    vol.commit()
    begin = time.monotonic()
    env = dict(os.environ, LAB_DEST=str(dest), PYTHONDONTWRITEBYTECODE='1', XLA_PYTHON_CLIENT_PREALLOCATE='false')
    out = dict(job=job)
    with (dest / 'stdout.log').open('w') as so, (dest / 'stderr.log').open('w') as se:
        r = subprocess.run([sys.executable, '/root/clrs/clrs_train.py', json.dumps(job)], env=env, stdout=so, stderr=se, timeout=job['timeout'])
    out['returncode'] = r.returncode
    if r.returncode:
        out['error'] = (dest / 'stderr.log').read_text()[-4000:]
    else:
        out['result'] = json.loads((dest / 'stdout.log').read_text().splitlines()[-1])
    out['wall_seconds'] = time.monotonic() - begin
    (dest / 'result.json').write_text(json.dumps(out))
    vol.commit()
    return r.returncode
