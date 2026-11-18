"""3D renders of one showcase case folder (made by prep.py), slide-ready.

Usage: python render.py <case_dir> [--views anterior,left,oblique]
Outputs in <case_dir>/renders/:
  panels/<source>_<view>.png      transparent RGBA, cropped to the same box for every source of a view (drop on slides)
  figure_<view>_{dark,light}.png  composed comparison: class-map surfaces (top) and density volume renders (bottom),
                                  titles, legend, density colour bar, case caption
  legend_*.png, colorbar_*.png    separate pieces
Surface approach follows yb107 MedImgResearch src/research/utils/visualizer.py (marching cubes -> PyVista mesh in world
coordinates, its organ colours). Density fields are continuous, so they are volume-rendered (an isosurface would show
one density level only). Same groups / colours for our classes, TotalSegmentator and the TrueCT phantom. TotalSeg has
no emphysema class and only pulmonary veins (our 'vessel' = all pulmonary vessels). Parallel projection, so the panels
of one view share one scale. CT panel: 1+HU/1000 through the density transfer function (approximate, for display).
"""
import argparse, json, os
from pathlib import Path

os.environ.setdefault("VTK_OPENGL_USE_EGL", "1")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import pyvista as pv
import scipy.ndimage as ndi
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Patch
from PIL import Image
from skimage.measure import marching_cubes

pv.OFF_SCREEN = True
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11})
# name, colour, opacity, our class ids (truect.data.CLASSES), TotalSeg name test
GROUPS = [("Lung", (0.66, 0.76, 0.93), 0.14, [3, 5, 6, 7], lambda n: n.startswith("lung_")),
          ("Bone", (0.957, 0.886, 0.776), 1.0, [9, 10, 11, 12, 14, 15, 16, 17],
           lambda n: n.startswith(("rib_", "vertebrae_T", "scapula_")) or n in ("sternum", "vertebrae_C7", "vertebrae_L1")),
          ("Aorta", (0.84, 0.12, 0.13), 1.0, [1], lambda n: n == "aorta"),
          ("Pulmonary vessels", (0.62, 0.20, 0.42), 1.0, [18], lambda n: n == "pulmonary_vein"),
          ("Esophagus", (0.99, 0.55, 0.45), 1.0, [4], lambda n: n == "esophagus"),
          ("Emphysema", (1.0, 0.78, 0.05), 1.0, [3], lambda n: False)]
DENS_CMAP = LinearSegmentedColormap.from_list("dens", [(0, "#ffd21a"), (0.04, "#ff7a1a"), (0.12, "#3d7bd9"),
                                                       (0.3, "#9fc3ff"), (0.45, "#e8e2d0"), (1, "#ffffff")])
CLIM = (0.0, 2.0)  # g/cm3
# opacity transfer function (density, opacity per UNIT_MM): emphysema/lung visible, soft tissue clear, bone opaque.
# Set on the VTK property directly: pyvista's array `opacity=` silently rendered nothing here (vtk 9.3.1 / pv 0.47.1).
TF = [(0, 0), (0.03, 0), (0.06, 0.5), (0.12, 0.12), (0.3, 0.03), (0.6, 0), (1.08, 0), (1.2, 0.35), (1.4, 0.8), (2.0, 0.95)]
UNIT_MM = 8
WIN = 1600
TITLES = {"pred_class": "Predicted phantom (ours)", "gt_class": "Ground-truth phantom (TrueCT)",
          "totalseg": "TotalSegmentator", "ct": "CT input", "pred_density": "Predicted density (ours)",
          "gt_density": "Ground-truth density (TrueCT)"}


def canon(path):
    img = nib.as_closest_canonical(nib.load(path))
    return np.asarray(img.dataobj), img.affine


def plotter():
    pl = pv.Plotter(off_screen=True, window_size=(WIN, WIN), lighting="light_kit")
    pl.enable_anti_aliasing("ssaa"); pl.enable_parallel_projection()
    return pl


def shoot(pl, center, view, path, scale=230, volume=False):
    c = np.asarray(center)
    eye = {"anterior": (0, 1, 0), "left": (-1, 0, 0), "oblique": (-0.6, 0.8, 0.12)}[view]  # RAS: +y anterior, -x left
    pl.camera_position = [tuple(c + 1000 * np.asarray(eye)), tuple(c), (0, 0, 1)]
    pl.camera.parallel_scale = scale
    if not volume:
        pl.screenshot(path, transparent_background=True); pl.close(); return
    # volume rays leave alpha at 0 in transparent screenshots: render on black, alpha from brightness (emission look)
    pl.set_background("black"); rgb = pl.screenshot(None, return_img=True)[..., :3]; pl.close()
    a = np.clip(rgb.max(-1).astype(np.float32) * 1.6, 0, 255)
    Image.fromarray(np.dstack([np.clip(rgb / np.maximum(a[..., None] / 255, 1e-3), 0, 255), a]).astype(np.uint8)).save(path)


def surf_meshes(lab, aff, ids_of):
    out = []
    for name, col, op, _, _ in GROUPS:
        ids = ids_of(name)
        m = np.isin(lab, ids) if ids else None
        if m is None or m.sum() < 200:
            continue
        v, f, _, _ = marching_cubes(np.pad(m, 1).astype(np.float32), 0.5, step_size=1 if name != "Lung" else 2)
        v = (aff @ np.c_[v - 1, np.ones(len(v))].T).T[:, :3]
        out.append((name, pv.PolyData(v, np.c_[np.full(len(f), 3), f].ravel()).smooth_taubin(n_iter=30, pass_band=0.05), col, op))
    return out


def body_mask(case, shape):
    """CT-derived TotalSegmentator body mask (canonical) if it is on this grid, else None. Model outputs outside the
    body (air, outside the TrueCT reconstruction disc) are meaningless, and the real-CT inference already zeroed them."""
    b = nib.as_closest_canonical(nib.load(case / "totalseg_body.nii.gz"))
    return np.asarray(b.dataobj) > 0 if b.shape == shape else None


def body_masked(dens, body):
    """Zero density outside the (eroded) body so the skin/air boundary does not fog the render."""
    if body is None:
        body = np.stack([ndi.binary_fill_holes(s) for s in np.moveaxis(dens > 0.5, 2, 0)], 2)
    return np.where(ndi.binary_erosion(body, iterations=2), dens, 0).astype(np.float32)


def caption(case):
    if (case / "gt_class.nii.gz").exists():
        c = np.asarray(nib.load(case / "gt_class.nii.gz").dataobj)[::2, ::2, ::2]
        return f"TrueCT validation subject {case.name.split('_')[1]}  ·  emphysema {100 * (c == 3).sum() / np.isin(c, [3, 5, 6, 7]).sum():.0f}% of lung (phantom)"
    ct = np.asarray(nib.load(case / "ct.nii.gz").dataobj)
    lung = np.isin(np.asarray(nib.load(case / "totalseg_total.nii.gz").dataobj), [i for i, n in ts_names(case).items() if n.startswith("lung_")])
    return f"Real patient CT ({case.name.split('_', 1)[1].replace('_', ' ')}, contrast-enhanced)  ·  LAA-950 {100 * (ct[lung] < -950).mean():.1f}%"


def ts_names(case):
    return {int(k): v for k, v in json.load(open(case.parent / "totalseg_class_map.json")).items()}


def render_panels(case, views, pdir):
    pred = next(case.glob("pred_*_class.nii.gz")).name.split("_class")[0]
    ours = lambda g: next(x[3] for x in GROUPS if x[0] == g)  # noqa: E731
    names = ts_names(case)
    surf = {"pred_class": (case / f"{pred}_class.nii.gz", ours), "gt_class": (case / "gt_class.nii.gz", ours),
            "totalseg": (case / "totalseg_total.nii.gz", lambda g: [i for i, n in names.items() if next(x[4] for x in GROUPS if x[0] == g)(n)])}
    vol = {"ct": case / "ct.nii.gz", "pred_density": case / f"{pred}_density.nii.gz", "gt_density": case / "gt_density_ctgrid.nii.gz"}
    lab, aff = canon(surf["pred_class"][0])
    center = (aff @ np.r_[np.argwhere(np.isin(lab[::4, ::4, ::4], [3, 5, 6, 7])).mean(0) * 4, 1])[:3]  # predicted lung centre
    for src, (p, ids_of) in surf.items():
        if not p.exists():
            continue
        lab, aff = canon(p)
        if src == "pred_class" and (b := body_mask(case, lab.shape)) is not None:
            lab = np.where(b, lab, 0)
        meshes = surf_meshes(lab, aff, ids_of)
        for view in views:
            pl = plotter(); pl.enable_depth_peeling(8)
            for n, m, col, op in meshes:
                pl.add_mesh(m, color=col, opacity=op, smooth_shading=True, specular=0.25, specular_power=20, ambient=0.15)
            shoot(pl, center, view, pdir / f"{src}_{view}.png")
    for src, p in vol.items():
        if not p.exists():
            continue
        v, aff = canon(p)
        b = body_mask(case, v.shape)
        v, b = v[::2, ::2, ::2].astype(np.float32), (b[::2, ::2, ::2] if b is not None else None)  # 1.2 mm is ample
        if src == "ct":
            v = np.clip(1 + v / 1000, 0, 2.5)
        g = pv.ImageData(dimensions=v.shape, spacing=np.abs(np.diag(aff)[:3]) * 2, origin=aff[:3, 3])
        g.point_data["density"] = body_masked(v, b).ravel(order="F")
        for view in views:
            pl = plotter()
            a = pl.add_volume(g, scalars="density", cmap=DENS_CMAP, clim=CLIM, opacity="linear", opacity_unit_distance=UNIT_MM,
                              shade=False, show_scalar_bar=False)
            f = a.prop.GetScalarOpacity(); f.RemoveAllPoints()
            for x, o in TF:
                f.AddPoint(x, o)
            shoot(pl, center, view, pdir / f"{src}_{view}.png", volume=True)


def crop_view(pdir, view, pad=30):
    """Crop every panel of one view to the union of their non-transparent boxes (keeps one shared scale)."""
    fs = sorted(pdir.glob(f"*_{view}.png"))
    ims = [Image.open(f).convert("RGBA") for f in fs]
    boxes = np.array([im.getchannel("A").point(lambda a: 255 if a > 8 else 0).getbbox() or (0, 0, WIN, WIN) for im in ims])
    box = (max(boxes[:, 0].min() - pad, 0), max(boxes[:, 1].min() - pad, 0), min(boxes[:, 2].max() + pad, WIN), min(boxes[:, 3].max() + pad, WIN))
    for f, im in zip(fs, ims):
        im.crop(box).save(f)


def compose(case, pdir, view, theme):
    bg, fg = ("#0d1117", "#e6edf3") if theme == "dark" else ("#ffffff", "#1f2328")
    rows = [[s for s in ("pred_class", "gt_class", "totalseg") if (pdir / f"{s}_{view}.png").exists()],
            [s for s in ("ct", "pred_density", "gt_density") if (pdir / f"{s}_{view}.png").exists()]]
    n = max(map(len, rows))
    fig = plt.figure(figsize=(4.2 * n + 2.4, 9.6), facecolor=bg)
    gs = fig.add_gridspec(2, n + 1, width_ratios=[1] * n + [0.5], left=0.01, right=0.99, top=0.9, bottom=0.02, wspace=0.03, hspace=0.12)
    for r, row in enumerate(rows):
        for j, s in enumerate(row):
            ax = fig.add_subplot(gs[r, j]); ax.set_facecolor(bg if r == 0 else "#0d1117")  # bright volume renders need dark ground
            ax.imshow(Image.open(pdir / f"{s}_{view}.png")); ax.axis("off")
            ax.set_title(TITLES[s], color=fg, fontsize=13, pad=6)
    lax = fig.add_subplot(gs[0, n]); lax.axis("off")
    lax.legend(handles=[Patch(color=c, label=nm) for nm, c, *_ in GROUPS], loc="center left", frameon=False,
               labelcolor=fg, fontsize=11, handlelength=1.2, handleheight=1.2)
    cax = fig.add_subplot(gs[1, n]); cax.axis("off")
    cb = fig.colorbar(plt.cm.ScalarMappable(plt.Normalize(*CLIM), DENS_CMAP), ax=cax, fraction=0.5, aspect=18, location="left")
    cb.set_label("Mass density (g/cm³)", color=fg); cb.ax.tick_params(colors=fg); cb.outline.set_visible(False)
    fig.text(0.01, 0.965, caption(case), color=fg, fontsize=15, weight="bold", va="center")
    fig.text(0.99, 0.965, f"{view} view", color=fg, fontsize=11, ha="right", va="center", alpha=0.7)
    fig.savefig(case / "renders" / f"figure_{view}_{theme}.png", dpi=200, facecolor=bg); plt.close(fig)


if __name__ == "__main__":
    a = argparse.ArgumentParser(); a.add_argument("case", type=Path); a.add_argument("--views", default="anterior,left,oblique")
    a.add_argument("--compose-only", action="store_true")
    x = a.parse_args()
    views, pdir = x.views.split(","), x.case / "renders" / "panels"
    pdir.mkdir(parents=True, exist_ok=True)
    if not x.compose_only:
        render_panels(x.case, views, pdir)
        for v in views:
            crop_view(pdir, v)
    for v in views:
        for t in ("dark", "light"):
            compose(x.case, pdir, v, t)
    print(x.case.name, "rendered", flush=True)
