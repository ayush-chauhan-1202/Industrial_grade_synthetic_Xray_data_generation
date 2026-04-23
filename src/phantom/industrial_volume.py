"""
Stage 1 — Physics-Aware 3D Industrial Phantom
==============================================
Generates a fully volumetric (Nx × Ny × Nz) CT phantom of a cylindrical
industrial component with physically grounded X-ray attenuation coefficients
and defects that evolve realistically along the axial (z) axis.

Physical basis
--------------
X-ray attenuation follows Beer-Lambert:   I = I0 * exp(-∫ μ dl)

Attenuation coefficients at 100 keV (cm⁻¹), from NIST XCOM:
    Air             :  0.000 cm⁻¹  (negligible)
    CFRP polymer    :  0.180 cm⁻¹
    Aluminium 7075  :  0.461 cm⁻¹
    Stainless steel :  0.797 cm⁻¹
    Tungsten carbide:  1.270 cm⁻¹
    Lead (inclusion):  1.100 cm⁻¹

All values are stored normalised to WC=1.0 for numerical stability,
but the physical_mu property returns true cm⁻¹ values.

Defects modelled
----------------
1. Spherical gas pores     — voids at random positions, spherical geometry
2. Axial micro-porosity    — pore cluster that tracks the steel/Al interface
3. Planar crack            — tilted elliptical crack that tapers to nothing
4. Dense inclusion         — ellipsoidal lead fragment
5. Cylindrical delamination— thin air shell between polymer and Al, partial arc
6. Corrosion gradient      — radial μ drop in Al body, stronger near one end

Usage
-----
    from src.phantom.industrial_volume import IndustrialPhantom, PhantomConfig
    cfg     = PhantomConfig(Nx=128, Ny=128, Nz=64)
    phantom = IndustrialPhantom(cfg)
    vol     = phantom.volume          # shape (Nz, Ny, Nx), float32
    meta    = phantom.metadata        # dict with physical info
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Tuple
import numpy as np
from scipy.ndimage import gaussian_filter


# ---------------------------------------------------------------------------
# Physical attenuation coefficients at 100 keV  (cm⁻¹)
# ---------------------------------------------------------------------------
MU_PHYSICAL = {
    "air":      0.000,
    "polymer":  0.180,
    "aluminium":0.461,
    "steel":    0.797,
    "tungsten": 1.270,
    "lead":     1.100,
}
MU_NORM_FACTOR = MU_PHYSICAL["tungsten"]   # normalise so WC = 1.0

MU = {k: v / MU_NORM_FACTOR for k, v in MU_PHYSICAL.items()}


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class PhantomConfig:
    """All parameters controlling the 3D phantom."""

    # Volume dimensions (voxels)
    Nx: int = 128
    Ny: int = 128
    Nz: int = 64

    # Physical voxel size (mm)
    voxel_size_mm: float = 0.5

    # Layer radii as fraction of min(Nx,Ny)/2
    r_polymer:    float = 0.90
    r_aluminium:  float = 0.78
    r_steel:      float = 0.52
    r_tungsten:   float = 0.17

    # Corrosion: max μ reduction fraction at worst end (z=0)
    corrosion_strength: float = 0.1

    # Spherical pores: list of (x_frac, y_frac, z_frac, radius_mm)
    # fractions relative to volume centre
    pores: List[Tuple[float,float,float,float]] = field(default_factory=lambda: [
        (-0.32,  0.25, -0.10,  2.0),   # large pore
        ( 0.40, -0.18,  0.20,  1.4),   # medium pore
        (-0.12, -0.40,  0.35,  0.8),   # small pore
        ( 0.20,  0.38, -0.30,  0.6),   # tiny pore
    ])

    # Micro-porosity cluster
    n_micro_pores:       int   = 80
    micro_pore_r_mm:     float = 0.3    # max radius
    micro_pore_jitter:   float = 0.06   # radial jitter as fraction of R_steel

    # Crack: tilted elliptical disk
    crack_centre_frac:   Tuple[float,float,float] = (-0.28,  0.15,  0.0)
    crack_axes_mm:       Tuple[float,float,float] = (10.0,   6.0,   0.8)
    crack_tilt_deg:      float = 22.0   # tilt around y-axis

    # Dense inclusion (lead fragment)
    inclusion_centre_frac: Tuple[float,float,float] = (0.25, 0.30, 0.10)
    inclusion_axes_mm:     Tuple[float,float,float] = (2.5,  1.5,  1.0)

    # Delamination arc (angular extent in degrees)
    delam_arc_start_deg: float =  25.0
    delam_arc_end_deg:   float = 140.0
    delam_thickness_mm:  float =  0.5
    delam_z_frac:        Tuple[float,float] = (-0.5, 0.3)  # z extent

    # Partial-volume smoothing (voxels)
    smooth_sigma: float = 0.4

    # RNG seed
    seed: int = 42


# ---------------------------------------------------------------------------
# Builder
# ---------------------------------------------------------------------------

class IndustrialPhantom:
    """
    Builds and stores the 3D industrial CT phantom volume.

    Args:
        cfg: PhantomConfig instance.

    Attributes:
        volume   : float32 array (Nz, Ny, Nx) with normalised μ values.
        metadata : dict with voxel size, physical μ map, defect list.
    """

    def __init__(self, cfg: PhantomConfig | None = None):
        self.cfg = cfg or PhantomConfig()
        self._rng = np.random.default_rng(self.cfg.seed)
        self.volume, self.label_volume = self._build()

    # ------------------------------------------------------------------
    # Public helpers
    # ------------------------------------------------------------------

    @property
    def shape(self) -> Tuple[int,int,int]:
        return self.volume.shape   # (Nz, Ny, Nx)

    @property
    def physical_volume(self) -> np.ndarray:
        """Return volume in true cm⁻¹ units."""
        return (self.volume * MU_NORM_FACTOR).astype(np.float32)

    @property
    def metadata(self) -> dict:
        cfg = self.cfg
        return {
            "shape":          list(self.volume.shape),
            "voxel_size_mm":  cfg.voxel_size_mm,
            "mu_normalisation_factor": MU_NORM_FACTOR,
            "mu_physical_cm_inv": MU_PHYSICAL,
            "mu_normalised":  MU,
            "energy_keV":     100,
        }

    def get_slice(self, axis: str = "z", index: int | None = None) -> np.ndarray:
        """
        Extract a 2-D slice from the volume.

        Args:
            axis:  'x', 'y', or 'z' (axial).
            index: Slice index. Defaults to the central slice.

        Returns:
            2-D float32 slice.
        """
        Nz, Ny, Nx = self.volume.shape
        if axis == "z":
            idx = index if index is not None else Nz // 2
            return self.volume[idx]
        elif axis == "y":
            idx = index if index is not None else Ny // 2
            return self.volume[:, idx, :]
        elif axis == "x":
            idx = index if index is not None else Nx // 2
            return self.volume[:, :, idx]
        raise ValueError(f"axis must be 'x', 'y', or 'z', got {axis!r}")

    # ------------------------------------------------------------------
    # Internal build
    # ------------------------------------------------------------------

    def _build(self) -> Tuple[np.ndarray, np.ndarray]:
        cfg   = self.cfg
        Nz, Ny, Nx = cfg.Nz, cfg.Ny, cfg.Nx
        vs    = cfg.voxel_size_mm
        R_max = min(Nx, Ny) / 2.0            # voxels

        vol   = np.zeros((Nz, Ny, Nx), dtype=np.float32)
        label = np.zeros((Nz, Ny, Nx), dtype=np.uint8)   # 0=air,1=poly,...

        # coordinate grids (voxel units, centred)
        cz, cy, cx = Nz/2, Ny/2, Nx/2
        z_idx = np.arange(Nz)
        Z, Y, X = np.meshgrid(z_idx - cz,
                               np.arange(Ny) - cy,
                               np.arange(Nx) - cx,
                               indexing="ij")            # (Nz,Ny,Nx)
        R_xy  = np.sqrt(Y**2 + X**2)                    # radial in xy

        # ── Material layers (inside-out so outer overwrites) ───────────────
        layers = [
            ("polymer",    cfg.r_polymer),
            ("aluminium",  cfg.r_aluminium),
            ("steel",      cfg.r_steel),
            ("tungsten",   cfg.r_tungsten),
        ]
        label_ids = {"polymer":1, "aluminium":2, "steel":3, "tungsten":4}

        for mat, r_frac in layers:
            r_px = r_frac * R_max
            mask = R_xy <= r_px
            vol[mask]   = MU[mat]
            label[mask] = label_ids[mat]

        # ── Corrosion gradient in Al body (stronger at z=0 end) ───────────
        r_al_in  = cfg.r_steel      * R_max
        r_al_out = cfg.r_aluminium  * R_max
        al_mask  = (R_xy > r_al_in) & (R_xy <= r_al_out)

        # radial gradient (0 at inner → 1 at outer edge)
        r_grad  = np.clip((R_xy - r_al_in) / (r_al_out - r_al_in + 1e-8),
                          0, 1)
        # axial gradient (1 at z=0 → 0 at z=Nz)
        z_frac  = 1.0 - (Z + cz) / float(Nz)
        z_grad  = np.clip(z_frac, 0, 1)

        corr = cfg.corrosion_strength * r_grad * z_grad
        vol[al_mask] -= corr[al_mask].astype(np.float32)

        # ── Spherical gas pores ───────────────────────────────────────────
        for xf, yf, zf, r_mm in cfg.pores:
            r_px = r_mm / vs
            pc   = np.array([zf * Nz/2 + cz,
                              yf * Ny/2 + cy,
                              xf * Nx/2 + cx])
            dist = np.sqrt((Z - (pc[0]-cz))**2 +
                           (Y - (pc[1]-cy))**2 +
                           (X - (pc[2]-cx))**2)
            mask = dist <= r_px
            vol[mask]   = MU["air"]
            label[mask] = 0

        # ── Micro-porosity cluster at steel/Al interface ──────────────────
        r_iface = (cfg.r_steel * R_max)
        for _ in range(cfg.n_micro_pores):
            ang   = self._rng.uniform(0, 2*np.pi)
            jit   = self._rng.uniform(-cfg.micro_pore_jitter,
                                       cfg.micro_pore_jitter) * R_max
            ri    = r_iface + jit
            py_px = ri * np.sin(ang)
            px_px = ri * np.cos(ang)
            pz_px = self._rng.uniform(-cz*0.9, cz*0.9)
            r_px  = self._rng.uniform(0.4, cfg.micro_pore_r_mm / vs)
            dist  = np.sqrt((Z - pz_px)**2 +
                            (Y - py_px)**2 +
                            (X - px_px)**2)
            vol[dist <= r_px]   = MU["air"]
            label[dist <= r_px] = 0

        # ── Planar crack (tilted ellipsoidal disk) ────────────────────────
        cc = cfg.crack_centre_frac
        cp = np.array([cc[2]*Nz/2, cc[1]*Ny/2, cc[0]*Nx/2])   # (z,y,x) offsets
        ax, ay, az = [a / vs for a in cfg.crack_axes_mm]        # in voxels
        tilt  = np.deg2rad(cfg.crack_tilt_deg)
        cos_t, sin_t = np.cos(tilt), np.sin(tilt)

        # Rotate coordinates around y-axis by tilt
        dZ = Z - cp[0];  dY = Y - cp[1];  dX = X - cp[2]
        rZ =  cos_t*dZ + sin_t*dX
        rX = -sin_t*dZ + cos_t*dX

        crack_mask = ((rX/ax)**2 + (dY/ay)**2 + (rZ/az)**2) <= 1.0
        vol[crack_mask]   = MU["air"]
        label[crack_mask] = 0

        # ── Dense inclusion (lead) ────────────────────────────────────────
        ic = cfg.inclusion_centre_frac
        ip = np.array([ic[2]*Nz/2, ic[1]*Ny/2, ic[0]*Nx/2])
        iax, iay, iaz = [a / vs for a in cfg.inclusion_axes_mm]
        dZ = Z - ip[0];  dY = Y - ip[1];  dX = X - ip[2]
        inc_mask = ((dX/iax)**2 + (dY/iay)**2 + (dZ/iaz)**2) <= 1.0
        vol[inc_mask]   = MU["lead"]
        label[inc_mask] = 5

        # ── Delamination arc ──────────────────────────────────────────────
        r_delam  = (cfg.r_polymer * R_max + cfg.r_aluminium * R_max) / 2.0
        th_px    = (cfg.delam_thickness_mm / vs) / 2.0
        ang_vol  = np.arctan2(Y, X)   # (Nz,Ny,Nx)

        a_start  = np.deg2rad(cfg.delam_arc_start_deg)
        a_end    = np.deg2rad(cfg.delam_arc_end_deg)
        in_arc   = (ang_vol >= a_start) & (ang_vol <= a_end)

        z_lo = cfg.delam_z_frac[0] * Nz/2
        z_hi = cfg.delam_z_frac[1] * Nz/2
        in_z = (Z >= z_lo) & (Z <= z_hi)

        in_shell = np.abs(R_xy - r_delam) <= th_px
        delam_mask = in_arc & in_z & in_shell
        vol[delam_mask]   = MU["air"]
        label[delam_mask] = 0

        # ── Clip and partial-volume smooth ────────────────────────────────
        vol = np.clip(vol, 0.0, 1.0)
        vol = gaussian_filter(vol, sigma=cfg.smooth_sigma).astype(np.float32)

        return vol, label
