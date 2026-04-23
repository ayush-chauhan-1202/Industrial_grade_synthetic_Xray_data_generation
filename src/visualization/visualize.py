"""
Stage 1 — Visualisation
========================
All plotting functions for Stage 1: volume inspection, projection views,
sinogram stacks, and the geometry diagram.

All functions return a matplotlib Figure so callers can save or display
them independently.  Pass save_path to auto-save as PNG.
"""

from __future__ import annotations
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1 import make_axes_locatable

# ── Style constants ────────────────────────────────────────────────────────
BG      = "#0f0f0f"
PANEL   = "#181818"
EDGE    = "#303030"
TEXT    = "#eeeeee"
DIM     = "#888888"
CMAP_MU = "gray"
CMAP_SN = "magma"
CMAP_DF = "inferno"

METHOD_COLORS = {
    "FBP":  "#4fc3f7",
    "SART": "#aed581",
    "SIRT": "#ffb74d",
    "UNet": "#f48fb1",
}

MATERIAL_COLORS = {
    "Air":       "#111111",
    "Polymer":   "#2979ff",
    "Aluminium": "#69f0ae",
    "Steel":     "#ff6d00",
    "Tungsten":  "#ea80fc",
    "Lead":      "#ff1744",
}


def _styled(ax, title="", xlabel="", ylabel="", border=None):
    ax.set_facecolor(PANEL)
    ax.tick_params(colors=DIM, labelsize=7)
    for sp in ax.spines.values():
        sp.set_edgecolor(border or EDGE)
        sp.set_linewidth(1.6 if border else 0.8)
    if title:
        ax.set_title(title, color=TEXT, fontsize=9, fontweight="bold",
                     fontfamily="monospace", pad=4)
    if xlabel:
        ax.set_xlabel(xlabel, color=DIM, fontsize=7.5, fontfamily="monospace")
    if ylabel:
        ax.set_ylabel(ylabel, color=DIM, fontsize=7.5, fontfamily="monospace")


def _cbar(fig, ax, im, label=""):
    div = make_axes_locatable(ax)
    cax = div.append_axes("right", size="5%", pad=0.06)
    cax.set_facecolor(PANEL)
    cb  = fig.colorbar(im, cax=cax)
    cb.ax.tick_params(colors=DIM, labelsize=6)
    cb.set_label(label, color=DIM, fontsize=6.5)
    cb.outline.set_edgecolor(EDGE)


def _save(fig, path):
    if path:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=BG)
    return fig


# ---------------------------------------------------------------------------
# 1. Three-plane orthogonal view of the volume
# ---------------------------------------------------------------------------

def plot_volume_orthogonal(volume: np.ndarray,
                           voxel_size_mm: float = 0.5,
                           title: str = "Industrial Phantom — Orthogonal Views",
                           save_path: str | None = None) -> plt.Figure:
    """
    Plot axial, coronal, and sagittal centre slices of the 3D volume.

    Args:
        volume:        (Nz, Ny, Nx) float32 attenuation volume.
        voxel_size_mm: Physical voxel pitch for axis labels.
        title:         Figure suptitle.
        save_path:     Optional PNG save path.
    """
    Nz, Ny, Nx = volume.shape
    vlo, vhi   = 0.0, float(volume.max())

    axial    = volume[Nz//2]            # (Ny, Nx)
    coronal  = volume[:, Ny//2, :]      # (Nz, Nx)
    sagittal = volume[:, :, Nx//2]      # (Nz, Ny)

    fig = plt.figure(figsize=(15, 5), facecolor=BG)
    fig.text(0.5, 0.98, title, ha="center", va="top",
             color=TEXT, fontsize=13, fontweight="bold", fontfamily="monospace")

    gs = gridspec.GridSpec(1, 4, figure=fig, wspace=0.30,
                           left=0.04, right=0.96, top=0.88, bottom=0.08)

    slices = [
        (axial,    "Axial  (z = centre)",    "x  (mm)", "y  (mm)",
         Nx*voxel_size_mm, Ny*voxel_size_mm),
        (coronal,  "Coronal  (y = centre)",  "x  (mm)", "z  (mm)",
         Nx*voxel_size_mm, Nz*voxel_size_mm),
        (sagittal, "Sagittal  (x = centre)", "y  (mm)", "z  (mm)",
         Ny*voxel_size_mm, Nz*voxel_size_mm),
    ]

    for col, (sl, ttl, xl, yl, W, H) in enumerate(slices):
        ax = fig.add_subplot(gs[0, col])
        im = ax.imshow(sl, cmap=CMAP_MU, vmin=vlo, vmax=vhi,
                       extent=[-W/2, W/2, -H/2, H/2], origin="lower")
        _styled(ax, ttl, xl, yl)
        _cbar(fig, ax, im, "μ (normalised)")

    # Material legend
    ax_leg = fig.add_subplot(gs[0, 3])
    ax_leg.set_facecolor(PANEL)
    ax_leg.axis("off")
    ax_leg.set_title("Materials", color=TEXT, fontsize=9, fontweight="bold",
                     fontfamily="monospace")
    items = [("Air / defects", 0.00), ("Polymer",    0.14),
             ("Aluminium",     0.36), ("Steel",       0.63),
             ("Tungsten",      0.79), ("Lead incl.",  0.87)]
    for i, (name, mu_n) in enumerate(items):
        y = 0.85 - i * 0.14
        ax_leg.add_patch(plt.Rectangle((0.05, y-0.04), 0.18, 0.09,
                         color=plt.cm.gray(mu_n / 1.0), transform=ax_leg.transAxes))
        ax_leg.text(0.28, y, f"{name}  (μ̃={mu_n:.2f})",
                    color=TEXT, fontsize=7.5, va="center",
                    transform=ax_leg.transAxes, fontfamily="monospace")

    return _save(fig, save_path)


# ---------------------------------------------------------------------------
# 2. Defect inspection panel
# ---------------------------------------------------------------------------

def plot_defect_panel(volume: np.ndarray,
                      voxel_size_mm: float = 0.5,
                      save_path: str | None = None) -> plt.Figure:
    """
    Show multiple axial slices to highlight how defects evolve along z.
    """
    Nz = volume.shape[0]
    n_show = min(8, Nz)
    z_indices = np.linspace(0, Nz-1, n_show, dtype=int)
    vlo, vhi  = 0.0, float(volume.max())

    fig, axes = plt.subplots(2, n_show//2, figsize=(16, 5), facecolor=BG)
    fig.patch.set_facecolor(BG)
    fig.suptitle("Axial Slices — Defect Evolution Along z-axis",
                 color=TEXT, fontsize=12, fontweight="bold",
                 fontfamily="monospace", y=1.01)

    for ax, z in zip(axes.flat, z_indices):
        im = ax.imshow(volume[z], cmap=CMAP_MU, vmin=vlo, vmax=vhi)
        _styled(ax, f"z = {z}  ({z*voxel_size_mm:.1f} mm)")
        ax.axis("off")

    plt.tight_layout(pad=0.5)
    return _save(fig, save_path)


# ---------------------------------------------------------------------------
# 3. Projection views (single angle + stack)
# ---------------------------------------------------------------------------

def plot_projection(projections: np.ndarray,
                    angle_idx: int = 0,
                    angles_deg: np.ndarray | None = None,
                    det_pixel_mm: float = 0.8,
                    save_path: str | None = None) -> plt.Figure:
    """
    Show a single cone-beam projection image (radiograph).

    Args:
        projections:  (n_angles, det_rows, det_cols) array.
        angle_idx:    Which angle to display.
        angles_deg:   Array of angles for the title.
        det_pixel_mm: Detector pixel pitch for axis labels.
    """
    proj = projections[angle_idx]
    n_a, nr, nc = projections.shape
    ang_str = f"{angles_deg[angle_idx]:.1f}°" if angles_deg is not None else f"#{angle_idx}"

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), facecolor=BG)
    fig.patch.set_facecolor(BG)
    fig.suptitle(f"Cone-Beam Projection at φ = {ang_str}",
                 color=TEXT, fontsize=12, fontweight="bold",
                 fontfamily="monospace")

    # Projection image
    ext = [-nc*det_pixel_mm/2, nc*det_pixel_mm/2,
           -nr*det_pixel_mm/2, nr*det_pixel_mm/2]
    im = axes[0].imshow(proj, cmap=CMAP_SN, extent=ext, origin="lower")
    _styled(axes[0], "Projection (∫μ dl)", "u  (mm)", "v  (mm)")
    _cbar(fig, axes[0], im, "mm·cm⁻¹")

    # Horizontal profile through centre row
    mid = nr // 2
    axes[1].set_facecolor(PANEL)
    axes[1].plot(np.arange(nc)*det_pixel_mm - nc*det_pixel_mm/2,
                 proj[mid], color="#4fc3f7", lw=1.4)
    axes[1].fill_between(np.arange(nc)*det_pixel_mm - nc*det_pixel_mm/2,
                          proj[mid], alpha=0.15, color="#4fc3f7")
    _styled(axes[1], "Central Row Profile", "u  (mm)", "∫μ dl  (mm·cm⁻¹)")
    axes[1].grid(color=EDGE, lw=0.4, alpha=0.7)

    plt.tight_layout()
    return _save(fig, save_path)


# ---------------------------------------------------------------------------
# 4. Sinogram stack (u-angle for each detector row)
# ---------------------------------------------------------------------------

def plot_sinogram_stack(projections: np.ndarray,
                        angles_deg: np.ndarray | None = None,
                        det_pixel_mm: float = 0.8,
                        save_path: str | None = None) -> plt.Figure:
    """
    Plot sinograms for three representative detector rows
    (top, centre, bottom).

    Args:
        projections:  (n_angles, det_rows, det_cols).
        angles_deg:   Projection angles array.
        det_pixel_mm: Detector pixel pitch.
    """
    n_a, nr, nc = projections.shape
    rows = [0, nr//4, nr//2, 3*nr//4, nr-1]
    labels = ["Top", "Upper", "Centre", "Lower", "Bottom"]

    ang = angles_deg if angles_deg is not None else np.arange(n_a)

    fig, axes = plt.subplots(1, len(rows), figsize=(16, 4), facecolor=BG)
    fig.patch.set_facecolor(BG)
    fig.suptitle("Sinogram Stack — Representative Detector Rows",
                 color=TEXT, fontsize=12, fontweight="bold",
                 fontfamily="monospace")

    for ax, row, lbl in zip(axes, rows, labels):
        sino = projections[:, row, :]    # (n_angles, det_cols)
        ext  = [ang[0], ang[-1],
                -nc*det_pixel_mm/2, nc*det_pixel_mm/2]
        im   = ax.imshow(sino.T, cmap=CMAP_SN, aspect="auto",
                         extent=ext, origin="lower")
        _styled(ax, f"{lbl}  (row {row})", "angle (°)", "u (mm)")
        _cbar(fig, ax, im, "∫μ dl")

    plt.tight_layout()
    return _save(fig, save_path)


# ---------------------------------------------------------------------------
# 5. Geometry diagram
# ---------------------------------------------------------------------------

def plot_geometry(geo, save_path: str | None = None) -> plt.Figure:
    """
    Draw a schematic top-view of the cone-beam geometry.

    Args:
        geo: ConeBeamGeometry instance.
    """
    fig, ax = plt.subplots(figsize=(8, 8), facecolor=BG)
    ax.set_facecolor(PANEL)
    ax.set_aspect("equal")
    _styled(ax, "Cone-Beam CT Geometry (Top View — XY Plane)",
            "x  (mm)", "y  (mm)")

    r_iso = 60   # draw isocenter circle for reference

    # Isocenter
    ax.add_patch(plt.Circle((0, 0), r_iso, fill=False,
                             edgecolor="#444444", lw=0.8, linestyle="--"))
    ax.plot(0, 0, "w+", ms=12, mew=2, zorder=5)
    ax.text(4, 4, "ISO", color=TEXT, fontsize=8, fontfamily="monospace")

    # Draw a few projection angles
    sample_angles = np.linspace(0, 2*np.pi, 8, endpoint=False)
    for phi in sample_angles:
        sx = -geo.SID * np.cos(phi)
        sy = -geo.SID * np.sin(phi)
        dx = (geo.SDD - geo.SID) * np.cos(phi)
        dy = (geo.SDD - geo.SID) * np.sin(phi)

        # Central ray
        ax.plot([sx, dx], [sy, dy], color="#4fc3f7", lw=0.6, alpha=0.5)

        # Source dot
        ax.plot(sx, sy, "o", color="#ff6d00", ms=5, zorder=6)

        # Detector line (perpendicular)
        det_half = geo.det_cols * geo.det_pixel_mm / 2
        ux = -np.sin(phi) * det_half
        uy =  np.cos(phi) * det_half
        ax.plot([dx - ux, dx + ux], [dy - uy, dy + uy],
                color="#aed581", lw=2, zorder=6)

    # Highlight φ=0 with labels
    phi0 = 0.0
    sx0  = -geo.SID
    dx0  =  geo.SDD - geo.SID

    ax.annotate("", xy=(sx0, 0), xytext=(-geo.SID*0.55, 0),
                arrowprops=dict(arrowstyle="->", color="#ff6d00", lw=1.5))
    ax.text(sx0 - 10, 8, f"Source\nSID={geo.SID:.0f}mm",
            color="#ff6d00", fontsize=7.5, ha="center", fontfamily="monospace")

    ax.annotate("", xy=(dx0, 0), xytext=(dx0*0.5, 0),
                arrowprops=dict(arrowstyle="->", color="#aed581", lw=1.5))
    ax.text(dx0 + 12, 8, f"Detector\nSDD={geo.SDD:.0f}mm",
            color="#aed581", fontsize=7.5, ha="center", fontfamily="monospace")

    # Source trajectory circle
    theta_full = np.linspace(0, 2*np.pi, 300)
    ax.plot(-geo.SID*np.cos(theta_full), -geo.SID*np.sin(theta_full),
            "--", color="#ff6d00", lw=0.8, alpha=0.4,
            label=f"Source trajectory  (r={geo.SID}mm)")

    ax.legend(loc="lower right", facecolor="#111111", edgecolor=EDGE,
              labelcolor=TEXT, fontsize=7.5)
    ax.set_xlim(-geo.SID*1.15, geo.SDD*1.05)
    ax.set_ylim(-geo.SID*1.15, geo.SID*1.15)
    ax.grid(color=EDGE, lw=0.4, alpha=0.5)

    plt.tight_layout()
    return _save(fig, save_path)


# ---------------------------------------------------------------------------
# 6. Full Stage 1 summary figure
# ---------------------------------------------------------------------------

def plot_stage1_summary(volume: np.ndarray,
                        projections_clean: np.ndarray,
                        projections_noisy: np.ndarray,
                        geo,
                        voxel_size_mm: float = 0.5,
                        noise_label: str = "medium",
                        save_path: str | None = None) -> plt.Figure:
    """
    One-page summary figure for Stage 1.

    Layout  (4 rows × 5 columns)
    ─────────────────────────────
    Row 0 : axial  |  coronal  |  sagittal  |  μ histogram  |  material legend
    Row 1 : defect slices (5 evenly spaced axial slices)
    Row 2 : clean projection  |  noisy projection  |  difference  |  row profile
    Row 3 : sinograms for top / centre / bottom detector rows  |  geometry schematic
    """
    Nz, Ny, Nx = volume.shape
    n_a, nr, nc = projections_clean.shape
    vlo, vhi = 0.0, float(volume.max())
    ang = geo.angles_deg

    fig = plt.figure(figsize=(22, 20), facecolor=BG)
    fig.patch.set_facecolor(BG)
    fig.text(0.5, 0.992,
             "Stage 1 — 3D Industrial Phantom & Cone-Beam Projections",
             ha="center", va="top", fontsize=16, fontweight="bold",
             color=TEXT, fontfamily="monospace")
    fig.text(0.5, 0.984,
             f"Volume {Nx}×{Ny}×{Nz}  ·  voxel {voxel_size_mm}mm  ·  "
             f"{n_a} angles  ·  detector {nr}×{nc}  ·  noise: {noise_label}",
             ha="center", va="top", fontsize=9, color=DIM, fontfamily="monospace")

    gs = gridspec.GridSpec(4, 5, figure=fig, hspace=0.42, wspace=0.32,
                           left=0.04, right=0.96, top=0.975, bottom=0.03)

    # ── Row 0: orthogonal views ───────────────────────────────────────────────
    for col, (sl, ttl, xl, yl) in enumerate([
        (volume[Nz//2],         "Axial  (centre z)",   "x", "y"),
        (volume[:,Ny//2,:],     "Coronal  (centre y)", "x", "z"),
        (volume[:,:,Nx//2],     "Sagittal  (centre x)","y", "z"),
    ]):
        ax = fig.add_subplot(gs[0, col])
        im = ax.imshow(sl, cmap=CMAP_MU, vmin=vlo, vmax=vhi)
        _styled(ax, ttl, xl, yl)
        _cbar(fig, ax, im)

    # μ histogram
    ax_hist = fig.add_subplot(gs[0, 3])
    ax_hist.set_facecolor(PANEL)
    vals = volume[volume > 0.01].ravel()
    ax_hist.hist(vals, bins=80, color="#4fc3f7", edgecolor="none", alpha=0.85)
    _styled(ax_hist, "μ Distribution (non-air)", "μ (normalised)", "count")
    ax_hist.grid(color=EDGE, lw=0.4, alpha=0.6)

    # material legend
    ax_leg = fig.add_subplot(gs[0, 4])
    ax_leg.set_facecolor(PANEL)
    ax_leg.axis("off")
    ax_leg.set_title("Material Key", color=TEXT, fontsize=9, fontweight="bold",
                     fontfamily="monospace", pad=4)
    from src.phantom.industrial_volume import MU, MU_PHYSICAL
    entries = [("Air/defects","air"),("Polymer","polymer"),
               ("Aluminium","aluminium"),("Steel","steel"),
               ("Tungsten","tungsten"),("Lead incl.","lead")]
    for i,(label,key) in enumerate(entries):
        y = 0.88 - i*0.15
        mu_n = MU.get(key, 0)
        ax_leg.add_patch(plt.Rectangle((0.04, y-0.045), 0.14, 0.09,
                         color=plt.cm.gray(min(mu_n,1.0)),
                         transform=ax_leg.transAxes, zorder=3))
        ax_leg.text(0.22, y,
                    f"{label}\n  μ̃={mu_n:.2f}  ({MU_PHYSICAL.get(key,0):.3f} cm⁻¹)",
                    color=TEXT, fontsize=6.8, va="center",
                    transform=ax_leg.transAxes, fontfamily="monospace")

    # ── Row 1: defect evolution slices ────────────────────────────────────────
    z_ids = np.linspace(0, Nz-1, 5, dtype=int)
    for col, z in enumerate(z_ids):
        ax = fig.add_subplot(gs[1, col])
        ax.imshow(volume[z], cmap=CMAP_MU, vmin=vlo, vmax=vhi)
        _styled(ax, f"z={z}  ({z*voxel_size_mm:.1f}mm)")
        ax.set_xticks([]); ax.set_yticks([])

    # ── Row 2: projection views ───────────────────────────────────────────────
    p_clean = projections_clean[0]
    p_noisy = projections_noisy[0]
    p_diff  = np.abs(p_noisy - p_clean)
    vp_max  = float(max(p_clean.max(), p_noisy.max()))

    for col, (img, ttl, cm, vmax) in enumerate([
        (p_clean, f"Clean proj. (φ=0°)",  CMAP_SN, vp_max),
        (p_noisy, f"Noisy proj.  [{noise_label}]", CMAP_SN, vp_max),
        (p_diff,  "Noise difference",      CMAP_DF, None),
    ]):
        ax = fig.add_subplot(gs[2, col])
        im = ax.imshow(img, cmap=cm, vmax=vmax)
        _styled(ax, ttl, "u (det cols)", "v (det rows)")
        _cbar(fig, ax, im)

    # central row profile
    ax_pr = fig.add_subplot(gs[2, 3:5])
    ax_pr.set_facecolor(PANEL)
    mid  = nr // 2
    u    = (np.arange(nc) - nc/2) * geo.det_pixel_mm
    ax_pr.plot(u, p_clean[mid], color="#ffffff", lw=1.4, label="Clean")
    ax_pr.plot(u, p_noisy[mid], color="#ff6d00", lw=1.0, alpha=0.85, label="Noisy")
    ax_pr.fill_between(u, p_clean[mid], p_noisy[mid],
                        alpha=0.15, color="#ff6d00")
    _styled(ax_pr, "Central Row Profile  (v = centre)", "u  (mm)", "∫μ dl")
    ax_pr.legend(framealpha=0.2, labelcolor="white", fontsize=8,
                 facecolor="#111111", edgecolor=EDGE)
    ax_pr.grid(color=EDGE, lw=0.4, alpha=0.6)

    # ── Row 3: sinograms + geometry ───────────────────────────────────────────
    row_ids   = [0, nr//4, nr//2, 3*nr//4]
    row_labels = ["Top", "Upper", "Centre", "Lower"]

    for col, (row, lbl) in enumerate(zip(row_ids, row_labels)):
        ax = fig.add_subplot(gs[3, col])
        sino = projections_noisy[:, row, :]
        ext  = [ang[0], ang[-1], -nc*geo.det_pixel_mm/2, nc*geo.det_pixel_mm/2]
        im   = ax.imshow(sino.T, cmap=CMAP_SN, aspect="auto",
                         extent=ext, origin="lower")
        _styled(ax, f"Sinogram — {lbl} row", "angle (°)", "u (mm)")
        _cbar(fig, ax, im)

    # Geometry schematic (simplified inline version)
    ax_geo = fig.add_subplot(gs[3, 4])
    ax_geo.set_facecolor(PANEL)
    ax_geo.set_aspect("equal")
    _styled(ax_geo, "Geometry (XY, top view)", "x", "y")

    r_src = geo.SID
    r_det = geo.SDD - geo.SID
    th    = np.linspace(0, 2*np.pi, 300)
    ax_geo.plot(-r_src*np.cos(th), -r_src*np.sin(th),
                "--", color="#ff6d00", lw=0.8, alpha=0.5)

    for phi in np.linspace(0, 2*np.pi, 12, endpoint=False):
        sx = -r_src*np.cos(phi); sy = -r_src*np.sin(phi)
        dx =  r_det*np.cos(phi); dy =  r_det*np.sin(phi)
        ax_geo.plot([sx, dx], [sy, dy], color="#4fc3f7", lw=0.5, alpha=0.4)
        ax_geo.plot(sx, sy, "o", color="#ff6d00", ms=3, zorder=5)
        dh = geo.det_cols*geo.det_pixel_mm*0.25
        ux = -np.sin(phi)*dh; uy = np.cos(phi)*dh
        ax_geo.plot([dx-ux, dx+ux], [dy-uy, dy+uy],
                    color="#aed581", lw=1.5, zorder=5)

    ax_geo.plot(0, 0, "w+", ms=10, mew=2, zorder=6)
    ax_geo.set_xlim(-r_src*1.1, (r_det)*1.3)
    ax_geo.set_ylim(-r_src*1.1, r_src*1.1)
    ax_geo.text(-r_src*0.9, r_src*0.85,
                f"SID={geo.SID:.0f}mm\nSDD={geo.SDD:.0f}mm\n"
                f"M={geo.magnification:.2f}×\n{geo.n_angles} angles",
                color=DIM, fontsize=6.5, fontfamily="monospace")

    return _save(fig, save_path)
