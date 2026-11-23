"""Make the picture of the library: docs/thumbnail.png.

Typical Earth orbits (LEO, SSO, MEO, GTO, Molniya, GEO) computed with
pyastrodyn-edu (``coe2rv`` for the orbits, ``rot1`` and ``rot3`` for the
viewing direction); matplotlib only draws them. The picture is used in the
README, as the social preview of the GitHub repository and on LinkedIn.
Run it from the repository root:

    python tools/make_thumbnail.py
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.collections import LineCollection  # noqa: E402
from matplotlib.patches import Circle  # noqa: E402

import pyastrodyn_edu as ad  # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "docs" / "thumbnail.png"
WIDTH, HEIGHT = 1200, 627           # pixels: the size of a LinkedIn link card; GitHub accepts it too

mu, R = ad.MU_EARTH, ad.R_EARTH
deg = np.radians
r_geo = ad.synchronous_radius(ad.SIDEREAL_DAY, mu)
rp_gto = R + 250.0

# label, a [km], e, i, raan, argp, colour, line width (nodes and periapses chosen for the picture)
orbits = [
    ("LEO", R + 550.0, 0.001, deg(53.0), deg(20.0), 0.0, "#7FDBFF", 1.4),
    ("LEO", R + 550.0, 0.001, deg(53.0), deg(80.0), 0.0, "#7FDBFF", 1.4),
    ("LEO", R + 550.0, 0.001, deg(53.0), deg(140.0), 0.0, "#7FDBFF", 1.4),
    ("SSO", R + 700.0, 0.001, deg(98.2), deg(-30.0), 0.0, "#B3E5FC", 1.2),
    ("MEO", 26560.0, 0.01, deg(55.0), deg(0.0), 0.0, "#FFD166", 1.4),
    ("MEO", 26560.0, 0.01, deg(55.0), deg(120.0), 0.0, "#FFD166", 1.4),
    ("MEO", 26560.0, 0.01, deg(55.0), deg(240.0), 0.0, "#FFD166", 1.4),
    ("GTO", 0.5 * (rp_gto + r_geo), (r_geo - rp_gto) / (r_geo + rp_gto),
     deg(28.5), deg(40.0), deg(200.0), "#EF476F", 1.6),
    ("Molniya", 26560.0, 0.74, deg(63.4), deg(200.0), deg(270.0), "#C77DFF", 1.6),
    ("GEO", r_geo, 0.0, 0.0, 0.0, 0.0, "#06D6A0", 1.8),
]
C_VN = ad.rot1(deg(-58.0)) @ ad.rot3(deg(-25.0))      # viewing basis V from N (passive)

fig = plt.figure(figsize=(WIDTH / 100.0, HEIGHT / 100.0), dpi=100)
fig.patch.set_facecolor("#0B1A2E")
stars = fig.add_axes([0, 0, 1, 1], zorder=-1)
stars.axis("off")
rng = np.random.default_rng(7)
stars.scatter(rng.uniform(0, 1, 240), rng.uniform(0, 1, 240), s=rng.uniform(0.2, 2.2, 240),
              c="white", alpha=0.35, linewidths=0)
stars.set_xlim(0, 1)
stars.set_ylim(0, 1)

ax = fig.add_axes([0.42, 0.02, 0.58, 0.96])
ax.set_facecolor("none")
ax.set_aspect("equal")
ax.axis("off")
f = np.linspace(0.0, 2.0 * np.pi, 1500)
for label, a, e, i, raan, argp, colour, lw in orbits:
    r_N, _ = ad.coe2rv(a, e, i, raan, argp, f, mu)
    x, y, z = (r_N @ C_VN.T / R).T                      # components in the viewing basis, Earth radii
    hidden = (z < 0.0) & (x**2 + y**2 < 1.0)            # behind the Earth
    seg = np.stack([np.column_stack([x[:-1], y[:-1]]), np.column_stack([x[1:], y[1:]])], axis=1)
    front = z[:-1] >= 0.0
    ax.add_collection(LineCollection(seg[~front & ~hidden[:-1]], colors=colour, linewidths=lw,
                                     alpha=0.45, zorder=1))
    ax.add_collection(LineCollection(seg[front], colors=colour, linewidths=lw, alpha=0.95, zorder=4))
    k = int(0.18 * len(f))                               # one satellite on each orbit
    if not hidden[k]:
        ax.plot(x[k], y[k], "o", ms=4.5, color=colour, zorder=5 if z[k] >= 0 else 1,
                markeredgecolor="white", markeredgewidth=0.6)
ax.add_patch(Circle((0, 0), 1.0, color="#1D4E89", zorder=3))
for k in range(30):                                      # a simple shading of the Earth
    t = k / 30
    ax.add_patch(Circle((-0.28 * t, 0.28 * t), 1.0 - 0.95 * t, color="#3A86C8", alpha=0.045, zorder=3))
ax.add_patch(Circle((0, 0), 1.0, fill=False, ec="#9DD3FF", lw=1.0, alpha=0.6, zorder=3))
ax.add_patch(Circle((0, 0), 1.06, fill=False, ec="#9DD3FF", lw=4.0, alpha=0.08, zorder=3))
ax.set_ylim(-7.6, 7.6)
ax.set_xlim(-7.6 * 1.13, 7.6 * 1.13)

fig.text(0.045, 0.62, "pyastrodyn-edu", color="white", fontsize=42, fontweight="bold")
fig.text(0.047, 0.52, "Educational astrodynamics in Python", color="#9DD3FF", fontsize=20)
fig.text(0.047, 0.40, "Readable functions  ·  worked examples  ·  Jupyter notebooks", color="#D0DCE8", fontsize=13)
fig.text(0.047, 0.34, "Open source, MIT License  ·  DOI 10.5281/zenodo.24479683", color="#D0DCE8", fontsize=13)
fig.text(0.047, 0.07, "LEO, SSO, MEO, GTO, Molniya and GEO orbits computed with pyastrodyn-edu",
         color="#7A8CA3", fontsize=10)

fig.savefig(OUT, dpi=100, facecolor=fig.get_facecolor())
print(f"written: {OUT.relative_to(OUT.parents[1])}")
