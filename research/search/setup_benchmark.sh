#!/usr/bin/env bash
# Clone the search benchmark of Saparov et al. (2025) at the commits every run pins and build its C++ generator.
# Their code is not vendored here; it keeps its own licence. Usage: ./setup_benchmark.sh [dest]   (default ./l2s)
# Needs git, g++ (C++17) and a Python with pybind11 (pip install pybind11==2.13.6) and numpy.
# Afterwards <dest>/build holds the generator module and their train.py (the file l2s_eval.load reads)
# (pass it as job['generator_dir'] or put it on PYTHONPATH).
set -euo pipefail
DEST=${1:-l2s}
L2S=2c3f32bd8073bba2144e112748bd6e8216bd7399     # github.com/asaparov/learning_to_search
CORE=1f3b022d2506b4806d14f5732feaa20487ef6601    # github.com/asaparov/core (header library)
PY=${PYTHON:-python3}
mkdir -p "$DEST"
[ -d "$DEST/repo/.git" ] || git clone https://github.com/asaparov/learning_to_search.git "$DEST/repo"
git -C "$DEST/repo" checkout --quiet "$L2S"
[ -d "$DEST/inc/core/.git" ] || git clone https://github.com/asaparov/core.git "$DEST/inc/core"
git -C "$DEST/inc/core" checkout --quiet "$CORE"
mkdir -p "$DEST/build"
cp "$DEST/repo/generator.cpp" "$DEST/repo/train.py" "$DEST/build/"
EXT=$("$PY" -c "import sysconfig; print(sysconfig.get_config_var('EXT_SUFFIX'))")
FLAGS="-O3 -DNDEBUG -shared -fPIC -std=c++17"
[ "$(uname)" = Darwin ] && FLAGS="$FLAGS -undefined dynamic_lookup"
( cd "$DEST/build" && g++ $FLAGS $("$PY" -m pybind11 --includes) -I../inc generator.cpp -o "generator$EXT" )
( cd "$DEST/build" && "$PY" -c "import generator; print('generator ok')" )
"$PY" -I "$(dirname "$0")/l2s_eval.py" "$DEST/repo/train.py" | head -3
