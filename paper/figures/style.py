"""Shared house style for the five figures: Cobalt and Rose on white, Helvetica Neue, layout in millimetres from the
top-left corner, 180 mm wide. Everything is re-exported from house.py (house style, legend and layout helpers, data
readers) and mapdraw.py (the 49-town map). Usage in a figure script:

    from style import *          # figure, text, axes_mm, canvas, legend, legend_at, hkey, save, md, palette names
    fig = figure(90)             # height in mm
    ...
    save(fig, "fig3")            # fig3.pdf, fig3.svg (text as outlines), fig3.png (300 dpi) in FIG_OUT
"""
import pathlib
import sys

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import house  # noqa: E402  (house style, helpers, data readers)
from house import (W, MM, SMALL, LABEL, LETTER, DOT, ROSE, ROSE_L, ROSE_D, COB, COB_L, INK, SOFT, GREY, MIST,  # noqa: E402,F401
                   VIOLET, TINT, cmap_exact, cmap_div, figure, text, canvas, hkey, legend, legend_at, axes_mm, blank,
                   clean)
import mapdraw as md  # noqa: E402,F401  (the 49-town map: layout, roads, water, towns)

DATA = HERE / "data"
OUT = house.OUT                    # FIG_OUT, default out/ beside this file
save = house.save
