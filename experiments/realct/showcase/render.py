"""3D renders of one showcase case folder (made by prep.py): surfaces of class maps, volume renders of density fields.

Usage: python render.py <case_dir> [--bg black|white] [--views anterior,left,oblique]
Surface approach follows yb107 MedImgResearch src/research/utils/visualizer.py (marching cubes -> PyVista mesh in
world coordinates, same organ colours); density fields (continuous) are volume-rendered instead, since an
isosurface of a density field only shows one density level.
Same structure groups / colours for our phantom classes, TotalSegmentator and the TrueCT phantom (GT), so panels
compare directly. TotalSeg has no emphysema class and only pulmonary veins (our 'vessel' = all pulmonary vessels).
"""
import argparse, json
from pathlib import Path

import os
os.environ.setdefault("VTK_OPENGL_USE_EGL", "1")
import nibabel as nib
import numpy as np
import pyvista as pv
import scipy.ndimage as ndi
from matplotlib.colors import LinearSegmentedColormap
from PIL import Image, ImageDraw, ImageFont
from skimage.measure import marching_cubes

pv.OFF_SCREEN = True
BONE = (0.957, 0.886, 0.776)
# name, colour, opacity, our class ids (truect.data.CLASSES), TotalSeg name test
GROUPS = [("lung", (0.659, 0.757, 0.933), 0.12, [3, 5, 6, 7], lambda n: n.startswith("lung_")),
          ("bone", BONE, 1.0, [9, 10, 11, 12, 14, 15, 16, 17],
           lambda n: n.startswith(("rib_", "vertebrae_T", "vertebrae_C7", "vertebrae_L1", "scapula_")) or n == "sternum"),
          ("aorta", (0.85, 0.1, 0.1), 1.0, [1], lambda n: n == "aorta"),
          ("pulmonary vessels", (0.75, 0.25, 0.35), 1.0, [18], lambda n: n == "pulmonary_vein"),
          ("esophagus", (0.988, 0.498, 0.490), 1.0, [4], lambda n: n == "esophagus"),
          ("emphysema", (1.0, 0.8, 0.1), 1.0, [3], lambda n: False)]
DS = 2  # 0.6 mm -> 1.2 mm for rendering; shortcut: coarser surfaces, set 1 for final high-res figures
DENS_CMAP = LinearSegmentedColormap.from_list("dens", [(0, "#ffd21a"), (0.04, "#ff7a1a"), (0.12, "#3d7bd9"),
                                                       (0.3, "#9fc3ff"), (0.45, "#e8e2d0"), (1, "#ffffff")])
CLIM = (0.0, 2.0)  # g/cm3
OPAC = np.interp(np.linspace(*CLIM, 256), [0, 0.03, 0.06, 0.12, 0.3, 0.6, 1.15, 1.3, 1.6, 2.0],
                 [0, 0, 0.5, 0.12, 0.03, 0.0, 0.0, 0.25, 0.7, 0.9])  # emphysema/lung visible, soft tissue clear


def canon(path):
    img = nib.as_closest_canonical(nib.load(path))
    return np.asarray(img.dataobj), img.affine


def camera(pl, center, view, dist=900):
    c = np.asarray(center)
    eye = {"anterior": (0, dist, 0), "left": (-dist, 0, 0), "oblique": (-0.6 * dist, 0.8 * dist, 0.15 * dist)}[view]
    pl.camera_position = [tuple(c + eye), tuple(c), (0, 0, 1)]  # RAS world: +y anterior, -x patient left


def surf_meshes(lab, aff, ids_of):
    out = []
    lab = lab[::DS, ::DS, ::DS]
    aff = aff @ np.diag([DS, DS, DS, 1])
    for name, col, op, _, _ in GROUPS:
        ids = ids_of(name)
        if not ids:
            continue
        m = np.isin(lab, ids)
        if m.sum() < 50:
            continue
        v, f, _, _ = marching_cubes(np.pad(m, 1).astype(np.float32), 0.5)
        v = (aff @ np.c_[v - 1, np.ones(len(v))].T).T[:, :3]
        mesh = pv.PolyData(v, np.c_[np.full(len(f), 3), f].ravel()).smooth_taubin(n_iter=20, pass_band=0.1)
        out.append((name, mesh, col, op))
    return out


def body_masked(dens):
    """Zero density outside the (hole-filled, slightly eroded) body so the skin/air boundary does not fog the render."""
    body = np.stack([ndi.binary_fill_holes(s) for s in np.moveaxis(dens > 0.5, 2, 0)], 2)
    return np.where(ndi.binary_erosion(body, iterations=2), dens, 0).astype(np.float32)


def render(case, bg, views):
    out = case / "renders"; out.mkdir(exist_ok=True)
    ts_names = {int(k): v for k, v in json.load(open(case.parent / "totalseg_class_map.json")).items()}
    pred = next(case.glob("pred_*_class.nii.gz")).name.split("_class")[0]
    srcs = {f"{pred}_class": (case / f"{pred}_class.nii.gz", lambda g: next(x[3] for x in GROUPS if x[0] == g))}
    if (case / "gt_class.nii.gz").exists():
        srcs["truect_phantom_class"] = (case / "gt_class.nii.gz", srcs[f"{pred}_class"][1])
    if (case / "totalseg_total.nii.gz").exists():
        srcs["totalseg"] = (case / "totalseg_total.nii.gz",
                            lambda g: [i for i, n in ts_names.items() if next(x[4] for x in GROUPS if x[0] == g)(n)])
    files = []
    for src, (p, ids_of) in srcs.items():
        lab, aff = canon(p)
        meshes = surf_meshes(lab, aff, ids_of)
        ctr = np.mean([m.center for n, m, _, _ in meshes if n == "lung"] or [meshes[0][1].center], 0)
        for view in views:
            pl = pv.Plotter(off_screen=True, window_size=(1400, 1400)); pl.set_background(bg)
            for n, m, col, op in meshes:
                pl.add_mesh(m, color=col, opacity=op, smooth_shading=True, specular=0.3)
            camera(pl, ctr, view); files.append(out / f"surface_{src}_{view}.png"); pl.screenshot(files[-1]); pl.close()
    dens = {f"{pred}_density": case / f"{pred}_density.nii.gz", "truect_phantom_density": case / "gt_density_ctgrid.nii.gz",
            "ct_as_density(1+HU/1000)": case / "ct.nii.gz"}
    for src, p in dens.items():
        if not p.exists():
            continue
        v, aff = canon(p)
        v = v[::DS, ::DS, ::DS].astype(np.float32)
        if src.startswith("ct_"):
            v = np.clip(1 + v / 1000, 0, 2.5)
        v = body_masked(v)
        g = pv.ImageData(dimensions=v.shape, spacing=np.abs(np.diag(aff)[:3]) * DS, origin=aff[:3, 3])
        g.point_data["density"] = v.ravel(order="F")
        for view in views:
            pl = pv.Plotter(off_screen=True, window_size=(1400, 1400)); pl.set_background(bg)
            pl.add_volume(g, scalars="density", cmap=DENS_CMAP, clim=CLIM, opacity=OPAC, shade=False, show_scalar_bar=False)
            camera(pl, g.center, view)
            files.append(out / f"volume_{src}_{view}.png"); pl.screenshot(files[-1]); pl.close()
    for view in views:  # labelled side-by-side strips for quick review
        for kind in ("surface", "volume"):
            ims = sorted(f for f in files if f.name.startswith(kind) and f.stem.endswith(view))
            if ims:
                strip(ims, out / f"compare_{kind}_{view}.png", bg)
    legend(out / "legend.png", bg)
    print(case.name, len(files), "renders", flush=True)


def strip(paths, dst, bg, w=700):
    ims = [Image.open(p).convert("RGB").resize((w, w)) for p in paths]
    canvas = Image.new("RGB", (w * len(ims), w + 40), bg)
    d = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 22)
    except OSError:
        font = ImageFont.load_default()
    for i, (p, im) in enumerate(zip(paths, ims)):
        canvas.paste(im, (i * w, 40))
        d.text((i * w + 10, 8), p.stem.split("_", 1)[1].rsplit("_", 1)[0], fill="white" if bg == "black" else "black", font=font)
    canvas.save(dst)


def legend(dst, bg):
    fg = "white" if bg == "black" else "black"
    im = Image.new("RGB", (420, 40 * len(GROUPS) + 20), bg); d = ImageDraw.Draw(im)
    for i, (n, c, *_) in enumerate(GROUPS):
        d.rectangle([10, 15 + 40 * i, 40, 45 + 40 * i], fill=tuple(int(255 * x) for x in c)); d.text((55, 22 + 40 * i), n, fill=fg)
    im.save(dst)


if __name__ == "__main__":
    a = argparse.ArgumentParser(); a.add_argument("case", type=Path); a.add_argument("--bg", default="black")
    a.add_argument("--views", default="anterior,left,oblique")
    x = a.parse_args()
    render(x.case, x.bg, x.views.split(","))
