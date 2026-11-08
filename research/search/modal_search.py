"""Modal app for the search benchmark. `modal deploy modal_search.py`, then spawn jobs with spawn_search.py; results on
the witnesses-search volume under search/<id>. Saparov et al.'s generator is compiled from their code at image build:
github.com/asaparov/learning_to_search at commit 2c3f32b and github.com/asaparov/core at commit 1f3b022, cloned at
those commits inside the image (not vendored into this repo)."""
from pathlib import Path
import modal
HERE = Path(__file__).resolve().parent
app = modal.App('witnesses-search')
vol = modal.Volume.from_name('witnesses-search', create_if_missing=True)
image = (modal.Image.debian_slim(python_version='3.12').apt_install('g++', 'git')
         .pip_install('jax[cuda12]==0.8.0', 'optax==0.2.6', 'numpy==2.3.4', 'pybind11==2.13.6')
         .run_commands(
             'git clone https://github.com/asaparov/learning_to_search.git /root/l2s_repo && cd /root/l2s_repo && git checkout 2c3f32bd8073bba2144e112748bd6e8216bd7399',
             'git clone https://github.com/asaparov/core.git /root/l2s_inc/core && cd /root/l2s_inc/core && git checkout 1f3b022d2506b4806d14f5732feaa20487ef6601',
             'mkdir -p /root/l2s && cp /root/l2s_repo/generator.cpp /root/l2s_repo/train.py /root/l2s/ && cd /root/l2s && '
             'g++ -O3 -DNDEBUG -shared -fPIC -std=c++17 $(python -m pybind11 --includes) -I/root/l2s_inc generator.cpp -o generator$(python3-config --extension-suffix 2>/dev/null || python -c "import sysconfig;print(sysconfig.get_config_var(\'EXT_SUFFIX\'))")',
             'cd /root/l2s && python -c "import generator; print(\'generator ok\')"')
         .add_local_file(HERE / 'search_train.py', '/root/search/search_train.py', copy=True)
         .add_local_file(HERE / 'l2s_eval.py', '/root/search/l2s_eval.py', copy=True)
         .add_local_file(HERE / 'loop_train.py', '/root/search/loop_train.py', copy=True)
         .add_local_file(HERE / 'bench_layers.py', '/root/search/bench_layers.py', copy=True)
         .add_local_file(HERE / 'newton_or_checks.py', '/root/search/newton_or_checks.py', copy=True)
         .add_local_file(HERE / 'propagator_checks.py', '/root/search/propagator_checks.py', copy=True))


torch_image = (modal.Image.debian_slim(python_version='3.12').apt_install('g++', 'git')
               .pip_install('torch==2.5.1', 'numpy==2.3.4', 'pybind11==2.13.6')
               .run_commands(
                   'git clone https://github.com/asaparov/learning_to_search.git /root/l2s_repo && cd /root/l2s_repo && git checkout 2c3f32bd8073bba2144e112748bd6e8216bd7399',
                   'git clone https://github.com/asaparov/core.git /root/l2s_repo/core && cd /root/l2s_repo/core && git checkout 1f3b022d2506b4806d14f5732feaa20487ef6601',
                   'cd /root/l2s_repo && python -c "import train" 2>&1 | tail -3 || true')
               .add_local_file(HERE / 'their_baseline.py', '/root/search/their_baseline.py', copy=True))


@app.function(image=image, gpu='H100', cpu=4, memory=32768, timeout=14000, max_containers=20, scaledown_window=2, retries=0, volumes={'/vol': vol})
def run(job):
    import json, os, subprocess, sys, time
    vol.reload()
    dest = Path('/vol/search') / job['id']
    dest.mkdir(parents=True, exist_ok=True)
    (dest / 'started.json').write_text(json.dumps(job))
    vol.commit()
    begin = time.monotonic()
    env = dict(os.environ, LAB_DEST=str(dest), PYTHONDONTWRITEBYTECODE='1', XLA_PYTHON_CLIENT_PREALLOCATE='false')
    out = dict(job=job)
    with (dest / 'stdout.log').open('w') as so, (dest / 'stderr.log').open('w') as se:
        r = subprocess.run([sys.executable, '/root/search/' + job.get('script', 'search_train.py'), json.dumps(job)], env=env, stdout=so, stderr=se, timeout=job['timeout'])
    out['returncode'] = r.returncode
    if r.returncode:
        out['error'] = (dest / 'stderr.log').read_text()[-4000:]
    else:
        out['result'] = json.loads((dest / 'stdout.log').read_text().splitlines()[-1])
    out['wall_seconds'] = time.monotonic() - begin
    (dest / 'result.json').write_text(json.dumps(out))
    vol.commit()
    return r.returncode


@app.function(image=image, cpu=8, memory=16384, timeout=14000, max_containers=4, scaledown_window=2, retries=0, volumes={'/vol': vol})
def run_cpu(job):
    """CPU-only jobs (the hand-set OR layer on the benchmark's test graphs); same layout as run."""
    import json, os, subprocess, sys, time
    vol.reload()
    dest = Path('/vol/search') / job['id']
    dest.mkdir(parents=True, exist_ok=True)
    (dest / 'started.json').write_text(json.dumps(job))
    vol.commit()
    begin = time.monotonic()
    env = dict(os.environ, LAB_DEST=str(dest), PYTHONDONTWRITEBYTECODE='1', JAX_PLATFORMS='cpu')
    out = dict(job=job)
    with (dest / 'stdout.log').open('w') as so, (dest / 'stderr.log').open('w') as se:
        r = subprocess.run([sys.executable, '/root/search/' + job['script'], json.dumps(job)], env=env, stdout=so, stderr=se, timeout=job['timeout'])
    out['returncode'] = r.returncode
    if r.returncode:
        out['error'] = (dest / 'stderr.log').read_text()[-4000:]
    else:
        out['result'] = json.loads((dest / 'stdout.log').read_text().splitlines()[-1])
    out['wall_seconds'] = time.monotonic() - begin
    (dest / 'result.json').write_text(json.dumps(out))
    vol.commit()
    return r.returncode


@app.function(image=torch_image, gpu='H100', cpu=4, memory=32768, timeout=14000, max_containers=20, scaledown_window=2, retries=0, volumes={'/vol': vol})
def run_theirs(job):
    """Their own Transformer training (train.py, unmodified), stopped at a cap by their_baseline.py."""
    import json, os, subprocess, sys, time
    vol.reload()
    dest = Path('/vol/search') / job['id']
    dest.mkdir(parents=True, exist_ok=True)
    (dest / 'started.json').write_text(json.dumps(job))
    vol.commit()
    begin = time.monotonic()
    env = dict(os.environ, LAB_DEST=str(dest), PYTHONDONTWRITEBYTECODE='1')
    out = dict(job=job)
    with (dest / 'stdout.log').open('w') as so, (dest / 'stderr.log').open('w') as se:
        r = subprocess.run([sys.executable, '/root/search/their_baseline.py', json.dumps(job)], env=env, stdout=so, stderr=se, timeout=job['timeout'])
    out['returncode'] = r.returncode
    if r.returncode:
        out['error'] = (dest / 'stderr.log').read_text()[-4000:]
    else:
        out['result'] = json.loads((dest / 'stdout.log').read_text().splitlines()[-1])
    out['wall_seconds'] = time.monotonic() - begin
    (dest / 'result.json').write_text(json.dumps(out))
    vol.commit()
    return r.returncode
