"""
Stage 1 — Cone-Beam Forward Projector
======================================
Simulates a cone-beam CT acquisition on a circular source trajectory.

Geometry (IEC / industrial convention)
---------------------------------------
    Rotation axis  = Z (world, = volume axis 0)
    Source moves   in XY plane around the object
    SID            = source-to-isocenter distance  (mm)
    SDD            = source-to-detector distance   (mm)
    Magnification  = SDD / SID

Ray parametrisation
-------------------
Each ray is cast from the X-ray source through every detector pixel.
To avoid the source being far (SID ≫ volume size), rays are sampled
only through a symmetric window around the isocentre:

    t ∈ [SID - vol_halfwidth*1.5,  SID + vol_halfwidth*1.5]

This guarantees all rays that could intersect the volume are sampled.
Trilinear interpolation (map_coordinates) evaluates μ at each sample.
The line integral is approximated by a Riemann sum × step size.

Coordinate convention
---------------------
    World  (x, y, z)  →  Volume  (axis0=Nz, axis1=Ny, axis2=Nx)
    world x  →  axis 2  (Nx, transaxial left-right)
    world y  →  axis 1  (Ny, transaxial depth)
    world z  →  axis 0  (Nz, axial / rotation axis)

Usage
-----
    from src.projector.conebeam import ConeBeamGeometry, ConeBeamProjector
    geo  = ConeBeamGeometry(SID=500, SDD=900, n_angles=180)
    proj = ConeBeamProjector(geo)
    data = proj.forward(volume, voxel_size_mm=0.5)
    # data.shape → (n_angles, det_rows, det_cols)
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple
import numpy as np
from scipy.ndimage import map_coordinates


def auto_detector_size(phantom_Nx: int, phantom_Nz: int,
                        voxel_size_mm: float,
                        SID: float, SDD: float,
                        det_pixel_mm: float,
                        margin: float = 1.25) -> tuple[int, int]:
    """
    Compute detector rows/cols that fully cover the phantom with margin.

    Args:
        phantom_Nx:    Phantom width in voxels (transaxial).
        phantom_Nz:    Phantom depth in voxels (axial).
        voxel_size_mm: Voxel pitch (mm).
        SID, SDD:      Geometry distances (mm).
        det_pixel_mm:  Detector pixel pitch (mm).
        margin:        Coverage margin factor (default 1.25 = 25% extra).

    Returns:
        (det_rows, det_cols)
    """
    M        = SDD / SID
    cols = int(np.ceil(phantom_Nx * voxel_size_mm * M / det_pixel_mm * margin))
    rows = int(np.ceil(phantom_Nz * voxel_size_mm * M / det_pixel_mm * margin))
    # Round up to even numbers
    cols = cols + (cols % 2)
    rows = rows + (rows % 2)
    return rows, cols


@dataclass
class ConeBeamGeometry:
    """
    Cone-beam CT scanner geometry descriptor.

    Args:
        SID:           Source-to-isocenter distance (mm).
        SDD:           Source-to-detector distance (mm).
        n_angles:      Projection angles over [0, 360).
        det_rows:      Detector rows (v, along rotation axis).
        det_cols:      Detector columns (u, transaxial).
        det_pixel_mm:  Physical detector pixel pitch (mm).
    """
    SID:          float = 500.0
    SDD:          float = 900.0
    n_angles:     int   = 180
    det_rows:     int   = 80
    det_cols:     int   = 160
    det_pixel_mm: float = 0.8

    @property
    def magnification(self) -> float:
        return self.SDD / self.SID

    @property
    def angles_rad(self) -> np.ndarray:
        return np.linspace(0, 2 * np.pi, self.n_angles, endpoint=False)

    @property
    def angles_deg(self) -> np.ndarray:
        return np.degrees(self.angles_rad)

    def detector_coords(self) -> Tuple[np.ndarray, np.ndarray]:
        """Physical (u, v) coordinates of detector pixel centres (mm)."""
        u = (np.arange(self.det_cols) - (self.det_cols - 1) / 2.0) * self.det_pixel_mm
        v = (np.arange(self.det_rows) - (self.det_rows - 1) / 2.0) * self.det_pixel_mm
        return u, v

    def summary(self) -> str:
        return (f"ConeBeamGeometry | SID={self.SID}mm  SDD={self.SDD}mm  "
                f"M={self.magnification:.2f}x  {self.n_angles} angles  "
                f"detector {self.det_rows}x{self.det_cols}  "
                f"pixel {self.det_pixel_mm}mm")


class ConeBeamProjector:
    """
    Ray-driven cone-beam forward projector with trilinear interpolation.

    Args:
        geo:       ConeBeamGeometry.
        n_samples: Samples per ray. Defaults to 2x volume diagonal / voxel.
        verbose:   Print per-angle progress.
    """

    def __init__(self, geo: ConeBeamGeometry | None = None,
                 n_samples: int | None = None, verbose: bool = True):
        self.geo       = geo or ConeBeamGeometry()
        self.n_samples = n_samples
        self.verbose   = verbose

    def forward(self, volume: np.ndarray,
                voxel_size_mm: float = 0.5) -> np.ndarray:
        """
        Forward-project a 3D volume to a full cone-beam projection stack.

        Args:
            volume:        float32 array (Nz, Ny, Nx) — normalised μ.
            voxel_size_mm: Physical voxel pitch (mm).

        Returns:
            projections: float32 (n_angles, det_rows, det_cols).
                         Units: mm · cm⁻¹ (line integral).
        """
        geo = self.geo
        Nz, Ny, Nx = volume.shape
        vs  = voxel_size_mm

        # Volume centre in voxel coords
        cx, cy, cz = Nx / 2.0, Ny / 2.0, Nz / 2.0

        # Half-width of volume in mm (used to set ray sampling window)
        vol_hw = max(Nx, Ny, Nz) * vs / 2.0

        # Ray sampling window: from (SID - margin) to (SID + margin) from source
        margin   = vol_hw * 1.8
        t_start  = max(0.0, geo.SID - margin)
        t_end    = geo.SID + margin
        n_s      = self.n_samples or max(256, int(2.0 * margin / vs))
        step_mm  = (t_end - t_start) / n_s
        t        = np.linspace(t_start, t_end, n_s)   # (n_s,)

        u_coords, v_coords = geo.detector_coords()
        UU, VV = np.meshgrid(u_coords, v_coords)       # (det_rows, det_cols)

        projections = np.zeros(
            (geo.n_angles, geo.det_rows, geo.det_cols), dtype=np.float32)

        for i, phi in enumerate(geo.angles_rad):
            if self.verbose and (i % max(1, geo.n_angles // 10) == 0):
                print(f"  Projecting angle {i+1:3d}/{geo.n_angles}  "
                      f"({np.degrees(phi):6.1f}°)")

            # Source position in world mm
            src = np.array([-geo.SID * np.cos(phi),
                            -geo.SID * np.sin(phi),
                             0.0])                         # (3,)

            # Detector centre in world mm
            det_c = np.array([(geo.SDD - geo.SID) * np.cos(phi),
                               (geo.SDD - geo.SID) * np.sin(phi),
                               0.0])

            # Detector axes:
            #   u-dir: perpendicular to source→iso in XY
            #   v-dir: along rotation axis (world Z)
            u_dir = np.array([-np.sin(phi), np.cos(phi), 0.0])
            v_dir = np.array([ 0.0,          0.0,         1.0])

            # Detector pixel positions in world mm: (det_rows, det_cols, 3)
            pix = (det_c
                   + UU[:, :, None] * u_dir
                   + VV[:, :, None] * v_dir)

            # Unit ray direction: source → pixel
            ray_dir = pix - src                            # (rows, cols, 3)
            ray_len = np.linalg.norm(ray_dir, axis=-1, keepdims=True)
            ray_dir = ray_dir / (ray_len + 1e-12)

            # Sample points along each ray: (n_s, rows, cols, 3)
            pts = src + t[:, None, None, None] * ray_dir[None, :, :, :]

            # Convert world mm → voxel indices
            #   world x → volume axis 2 (Nx)
            #   world y → volume axis 1 (Ny)
            #   world z → volume axis 0 (Nz)
            vox0 = pts[..., 2] / vs + cz    # world-z → Nz axis
            vox1 = pts[..., 1] / vs + cy    # world-y → Ny axis
            vox2 = pts[..., 0] / vs + cx    # world-x → Nx axis

            R, C = geo.det_rows, geo.det_cols
            coords = np.array([vox0.ravel(), vox1.ravel(), vox2.ravel()])

            # Trilinear interpolation; outside volume = 0 (air)
            samples = map_coordinates(volume, coords, order=1,
                                      mode='constant', cval=0.0)
            samples = samples.reshape(n_s, R, C)

            projections[i] = (samples.sum(axis=0) * step_mm).astype(np.float32)

        return projections
