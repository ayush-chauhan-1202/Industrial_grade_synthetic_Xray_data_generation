#!/usr/bin/env python3
"""
Stage 1 — 3D Industrial Phantom + Cone-Beam Projections
========================================================
Generates the physics-aware 3D volume, runs the cone-beam forward projector,
applies the realistic noise pipeline, saves all outputs, and renders the
summary visualisation.

Usage
-----
    # Defaults  (128×128×64, 180 angles, medium noise)
    python run_stage1.py

    # Custom
    python run_stage1.py --Nx 128 --Ny 128 --Nz 64 \
                         --n-angles 180 --noise medium \
                         --SID 500 --SDD 900 \
                         --out-dir results --seed 42

Outputs (in --out-dir)
----------------------
    volume/phantom.npy          — 3D attenuation volume (Nz×Ny×Nx), float32
    volume/phantom_meta.json    — voxel size, μ values, geometry
    projections/clean.npy       — (n_angles × det_rows × det_cols), float32
    projections/noisy.npy       — same, with realistic noise applied
    projections/geometry.json   — scanner geometry parameters
    figures/orthogonal.png      — 3-plane orthogonal views
    figures/defect_panel.png    — axial slice evolution
    figures/projection.png      — single projection radiograph
    figures/sinogram_stack.png  — sinograms for representative rows
    figures/geometry.png        — scanner geometry schematic
    figures/stage1_summary.png  — ONE-PAGE full summary (main output)
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def _parse() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    # Volume
    p.add_argument("--Nx",       type=int,   default=128, help="Volume width  (voxels)")
    p.add_argument("--Ny",       type=int,   default=128, help="Volume height (voxels)")
    p.add_argument("--Nz",       type=int,   default=64,  help="Volume depth  (voxels)")
    p.add_argument("--vox-mm",   type=float, default=0.5, help="Voxel size (mm)")
    # Scanner
    p.add_argument("--SID",      type=float, default=500.0,  help="Source-isocenter distance (mm)")
    p.add_argument("--SDD",      type=float, default=900.0,  help="Source-detector distance (mm)")
    p.add_argument("--n-angles", type=int,   default=180,    help="Projection angles")
    p.add_argument("--det-rows", type=int,   default=80,     help="Detector rows")
    p.add_argument("--det-cols", type=int,   default=160,    help="Detector columns")
    p.add_argument("--det-px",   type=float, default=0.8,    help="Detector pixel pitch (mm)")
    # Noise
    p.add_argument("--noise",    choices=["noiseless","low","medium","high"],
                   default="medium")
    # Misc
    p.add_argument("--seed",     type=int,   default=42)
    p.add_argument("--out-dir",  default="results")
    p.add_argument("--no-figure",action="store_true", help="Skip all figure rendering")
    p.add_argument("--save-raw", action="store_true", help="Export RAW + ImageJ macros")
    p.add_argument("--fast",     action="store_true",
                   help="Reduce ray samples for faster (less accurate) projection")
    return p.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _banner(msg: str) -> None:
    w = 62
    print(f"\n{'─'*w}\n  {msg}\n{'─'*w}")


def _elapsed(t0: float) -> str:
    dt = time.perf_counter() - t0
    return f"{dt:.1f}s" if dt < 60 else f"{dt/60:.1f}min"


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    args = _parse()

    out      = Path(args.out_dir)
    vol_dir  = out / "volume"
    proj_dir = out / "projections"
    fig_dir  = out / "figures"
    for d in (vol_dir, proj_dir, fig_dir):
        d.mkdir(parents=True, exist_ok=True)

    # ── 1. Build 3D phantom ───────────────────────────────────────────────────
    _banner("1 / 4  Building 3D industrial phantom")
    from src.phantom.industrial_volume import IndustrialPhantom, PhantomConfig

    t0  = time.perf_counter()
    cfg = PhantomConfig(Nx=args.Nx, Ny=args.Ny, Nz=args.Nz,
                        voxel_size_mm=args.vox_mm, seed=args.seed)
    ph  = IndustrialPhantom(cfg)
    vol = ph.volume

    print(f"  Shape        : {vol.shape}  (Nz × Ny × Nx)")
    print(f"  Physical size: {args.Nz*args.vox_mm:.1f} × "
          f"{args.Ny*args.vox_mm:.1f} × {args.Nx*args.vox_mm:.1f}  mm")
    print(f"  μ range      : [{vol.min():.4f}, {vol.max():.4f}]  (normalised)")
    print(f"  Non-air voxels: {(vol > 0.05).sum():,} / {vol.size:,}")
    print(f"  Done in {_elapsed(t0)}")

    np.save(vol_dir / "phantom.npy", vol)
    with open(vol_dir / "phantom_meta.json", "w") as f:
        json.dump(ph.metadata, f, indent=2)
    print(f"  Saved → {vol_dir}/phantom.npy")

    # ── 2. Cone-beam forward projection ──────────────────────────────────────
    _banner("2 / 4  Cone-beam forward projection")
    from src.projector.conebeam import ConeBeamGeometry, ConeBeamProjector

    from src.projector.conebeam import auto_detector_size
    det_rows, det_cols = auto_detector_size(
        phantom_Nx=args.Nx, phantom_Nz=args.Nz,
        voxel_size_mm=args.vox_mm,
        SID=args.SID, SDD=args.SDD,
        det_pixel_mm=args.det_px,
    )
    # Allow CLI overrides
    if args.det_rows != 80:   det_rows = args.det_rows
    if args.det_cols != 160:  det_cols = args.det_cols

    geo  = ConeBeamGeometry(
        SID=args.SID, SDD=args.SDD,
        n_angles=args.n_angles,
        det_rows=det_rows, det_cols=det_cols,
        det_pixel_mm=args.det_px,
    )
    print(f"  Detector (auto-sized): {det_rows} rows × {det_cols} cols")
    print(f"  {geo.summary()}")

    n_samp = 128 if args.fast else None   # fast mode uses fewer ray samples
    proj   = ConeBeamProjector(geo, n_samples=n_samp, verbose=True)

    t0            = time.perf_counter()
    proj_clean    = proj.forward(vol, voxel_size_mm=args.vox_mm)
    print(f"  Done in {_elapsed(t0)}")
    print(f"  Projections shape: {proj_clean.shape}  "
          f"(angles × det_rows × det_cols)")
    print(f"  Value range: [{proj_clean.min():.4f}, {proj_clean.max():.4f}]")

    np.save(proj_dir / "clean.npy", proj_clean)
    geo_meta = dict(SID=geo.SID, SDD=geo.SDD, n_angles=geo.n_angles,
                    det_rows=geo.det_rows, det_cols=geo.det_cols,
                    det_pixel_mm=geo.det_pixel_mm,
                    magnification=geo.magnification,
                    voxel_size_mm=args.vox_mm)
    with open(proj_dir / "geometry.json", "w") as f:
        json.dump(geo_meta, f, indent=2)
    print(f"  Saved → {proj_dir}/clean.npy")

    # ── 3. Apply realistic noise ──────────────────────────────────────────────
    _banner("3 / 4  Applying realistic noise pipeline")
    from src.noise.noise_model import apply_noise, PRESETS

    noise_cfg  = PRESETS[args.noise]
    noise_cfg.seed = args.seed
    proj_noisy = apply_noise(proj_clean, noise_cfg)

    print(f"  Preset : {args.noise}")
    print(f"  I0     : {noise_cfg.photon_count:.0f} photons/pixel")
    print(f"  σ_e    : {noise_cfg.gaussian_sigma}")
    print(f"  rings  : {noise_cfg.ring_strength*100:.1f}%")
    print(f"  BH coeff: {noise_cfg.beam_hardening}")
    print(f"  Scatter : {noise_cfg.scatter_fraction*100:.1f}%")

    snr = float(proj_clean.mean() / (proj_noisy - proj_clean).std())
    print(f"  SNR (mean/σ_noise): {snr:.1f}")

    np.save(proj_dir / "noisy.npy", proj_noisy)
    print(f"  Saved → {proj_dir}/noisy.npy")

    # ── 4. Visualisations ─────────────────────────────────────────────────────
    if args.no_figure:
        _banner("4 / 4  Figures skipped (--no-figure)")
    else:
        _banner("4 / 4  Rendering visualisations")
        from src.visualization.visualize import (
            plot_volume_orthogonal, plot_defect_panel,
            plot_projection, plot_sinogram_stack,
            plot_geometry, plot_stage1_summary,
        )

        t0 = time.perf_counter()

        print("  [1/6] Orthogonal views …")
        plot_volume_orthogonal(vol, voxel_size_mm=args.vox_mm,
                               save_path=str(fig_dir/"orthogonal.png"))

        print("  [2/6] Defect panel …")
        plot_defect_panel(vol, voxel_size_mm=args.vox_mm,
                          save_path=str(fig_dir/"defect_panel.png"))

        print("  [3/6] Projection view …")
        plot_projection(proj_noisy, angle_idx=0,
                        angles_deg=geo.angles_deg,
                        det_pixel_mm=geo.det_pixel_mm,
                        save_path=str(fig_dir/"projection.png"))

        print("  [4/6] Sinogram stack …")
        plot_sinogram_stack(proj_noisy, angles_deg=geo.angles_deg,
                            det_pixel_mm=geo.det_pixel_mm,
                            save_path=str(fig_dir/"sinogram_stack.png"))

        print("  [5/6] Geometry schematic …")
        plot_geometry(geo, save_path=str(fig_dir/"geometry.png"))

        print("  [6/6] Stage 1 summary (main figure) …")
        plot_stage1_summary(vol, proj_clean, proj_noisy, geo,
                            voxel_size_mm=args.vox_mm,
                            noise_label=args.noise,
                            save_path=str(fig_dir/"stage1_summary.png"))

        print(f"  All figures rendered in {_elapsed(t0)}")
        print(f"  Figures → {fig_dir}/")

    # ── Done ──────────────────────────────────────────────────────────────────
    # ── Optional: export RAW + ImageJ macros ──────────────────────────────────
    if args.save_raw:
        _banner("5 / 5  Exporting RAW files for ImageJ")
        import subprocess, sys
        subprocess.run(
            [sys.executable, str(ROOT / "save_as_raw.py"),
             "--out-dir", str(out)],
            check=True
        )

    _banner("Stage 1 complete")
    print(f"  Outputs in: {out.resolve()}\n"
          f"  ├── volume/\n"
          f"  │   ├── phantom.npy          ({vol.nbytes//1024} KB)\n"
          f"  │   └── phantom_meta.json\n"
          f"  ├── projections/\n"
          f"  │   ├── clean.npy            ({proj_clean.nbytes//1024} KB)\n"
          f"  │   ├── noisy.npy            ({proj_noisy.nbytes//1024} KB)\n"
          f"  │   └── geometry.json\n"
          f"  └── figures/\n"
          f"      ├── stage1_summary.png   ← start here\n"
          f"      ├── orthogonal.png\n"
          f"      ├── defect_panel.png\n"
          f"      ├── projection.png\n"
          f"      ├── sinogram_stack.png\n"
          f"      └── geometry.png")
    print()


if __name__ == "__main__":
    main()
