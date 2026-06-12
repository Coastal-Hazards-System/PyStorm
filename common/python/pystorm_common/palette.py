"""pystorm_common.palette — named colors for PyStorm figures.

Anchored on Toyota "Wave Maker" 0796 (#06ABC4), sampled from vehicle paint.
Designed for light backgrounds; categorical cycle is hero-first and
lightness-interleaved so neighbouring colors separate in grayscale.

Canonical design-handoff palette - values are final; use verbatim. This is the
single source of truth shared across modules via the CyHAN common library (5.2);
do not fork per-module copies.
"""
from cycler import cycler
from matplotlib.colors import LinearSegmentedColormap, ListedColormap
import matplotlib as mpl

# --- anchor ramp (Wave Maker 0796) ---
WAVE_MAKER = "#06ABC4"
RAMP = {50: "#ECF9FB", 100: "#CDEFF4", 200: "#A2E3EB", 300: "#66D2E4",
        400: "#2BBCD4", 500: "#06ABC4", 600: "#058B9F", 700: "#046E7E",
        800: "#02525E", 900: "#013A43"}

# --- categorical cycle (hero-first, lightness-interleaved) ---
CYCLE = ["#06ABC4", "#1F4E79", "#EE5A45", "#8159C9",
         "#EBA12C", "#2C9E8F", "#C73E86", "#6E828D"]
NAMES = ["wave_maker", "deep_ocean", "coral", "iris",
         "amber", "sea_green", "magenta", "slate"]
C = dict(zip(NAMES, CYCLE))

# --- structure / neutrals ---
INK       = "#13242C"   # axes, spines, title & body text
BODY      = "#3A4A52"   # tick labels, annotations
MUTED     = "#6B7C84"   # secondary labels, captions
EMPIRICAL = "#9FB1BA"   # empirical / WPP points below mu
GRID      = "#DEE5E9"   # gridlines
BAND      = "#D6EFF3"   # confidence-band fill (Wave Maker @ ~15%)
PANEL     = "#F5F8F9"   # optional shaded axes face
PAPER     = "#FFFFFF"   # default figure & axes background

# --- emphasis (warm complement to cyan) ---
EMPHASIS  = "#EE5A45"   # exceedance / peak markers
EMPH_DARK = "#D23A28"   # emphasis lines / selected-threshold dashes
EMPH_TINT = "#FBDDD7"   # emphasis fill / highlight band
REF_DASH  = "#13242C"   # reference & location (mu) dashed lines

# coordinated linestyle + marker cycle for grayscale-safe figures
STYLE_CYCLE = (cycler(color=CYCLE)
    + cycler(linestyle=["-", "--", "-.", ":", "-", "--", "-.", ":"])
    + cycler(marker=["o", "s", "^", "D", "v", "P", "X", "*"]))

# sequential colormap built on the Wave Maker ramp (optional convenience)
WAVE_MAKER_CMAP = LinearSegmentedColormap.from_list(
    "wave_maker", [RAMP[50], RAMP[300], RAMP[500], RAMP[700], RAMP[900]])
CYCLE_CMAP = ListedColormap(CYCLE, name="pystorm_cycle")


def apply(style_cycle: bool = False):
    """Activate the palette on the current matplotlib rcParams.

    style_cycle=False -> color-only cycle (default)
    style_cycle=True  -> color + linestyle + marker (for grayscale-critical figures)
    """
    mpl.rcParams["axes.prop_cycle"] = STYLE_CYCLE if style_cycle else cycler(color=CYCLE)


def band(ax, x, lo, hi, **kw):
    """Confidence-band helper using the Wave Maker tint."""
    return ax.fill_between(x, lo, hi, color=BAND, lw=0, **kw)


__all__ = [
    "WAVE_MAKER", "RAMP", "CYCLE", "NAMES", "C",
    "INK", "BODY", "MUTED", "EMPIRICAL", "GRID", "BAND", "PANEL", "PAPER",
    "EMPHASIS", "EMPH_DARK", "EMPH_TINT", "REF_DASH",
    "STYLE_CYCLE", "WAVE_MAKER_CMAP", "CYCLE_CMAP",
    "apply", "band",
]
