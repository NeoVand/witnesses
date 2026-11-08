"""Cobalt and Rose: the figure palette (data/style/palette.json) and the base matplotlib style
(data/style/house.mplstyle). Every colour map is white at its zero end.
"""
import json, pathlib
import matplotlib as mpl
from matplotlib.colors import LinearSegmentedColormap

HERE = pathlib.Path(__file__).resolve().parent / "data" / "style"
P = json.loads((HERE / "palette.json").read_text())

def use():
    mpl.style.use(str(HERE / "house.mplstyle"))

cmap_density = LinearSegmentedColormap.from_list("cr_density", ["#FFFFFF", "#DCE6F5", "#9DB8E0", "#4F7EC4", "#2557A7", "#2A2F6E", "#3B1F4F"])
cmap_exact = LinearSegmentedColormap.from_list("cr_exact", ["#FFFFFF", "#FBE3EB", "#F1A3BC", "#E3678F", "#D6336C", "#9E1F4B", "#5E1430"])
cmap_div = LinearSegmentedColormap.from_list("cr_div", ["#1B3F7A", "#2557A7", "#9DB8E0", "#FFFFFF", "#F1A3BC", "#D6336C", "#7A1F3D"])
for c in (cmap_density, cmap_exact, cmap_div):
    try: mpl.colormaps.register(c)
    except ValueError: pass
