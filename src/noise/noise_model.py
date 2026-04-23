"""
Stage 1 — Realistic Cone-Beam Noise Pipeline
=============================================
Applies physically motivated noise to cone-beam projections.

Unit convention
---------------
The projector outputs line integrals in  mm · (normalised μ).
Normalised μ has units cm⁻¹ (max=1.27 cm⁻¹ for WC at 100 keV).
So the raw projection values are in  mm · cm⁻¹.

The Beer-Lambert Poisson model requires a dimensionless optical depth τ:
    τ = ∫ μ(x) dx   [cm⁻¹ · cm = dimensionless]

Conversion:   τ = proj_mm_per_cm  ×  0.1   (1 mm = 0.1 cm)

Pipeline order (physically correct)
------------------------------------
1. Unit convert   mm·cm⁻¹  →  dimensionless τ
2. Beam hardening τ → τ + coeff·τ²   (pre-detection, polychromatic effect)
3. Poisson noise  I = I0·exp(-τ);  Ĩ ~ Poisson(I);  τ̃ = -log(Ĩ/I0)
4. Gaussian noise additive electronic read-out noise on τ̃
5. Ring artefacts per-column gain variation on τ̃
6. Scatter        uniform additive floor on τ̃
7. Unit convert back  τ̃  →  mm·cm⁻¹  (÷ 0.1)

The returned array is in the same units as the input (mm·cm⁻¹) so
it stays compatible with the FDK reconstructor in Stage 2.

Noise presets (industrial flat-panel at 100 keV)
-------------------------------------------------
    noiseless : I0=10⁹, all other effects off
    low       : I0=50 000, σ_e=0.002, rings=0.5%, BH=1%
    medium    : I0=10 000, σ_e=0.005, rings=1.0%, BH=3%
    high      : I0= 2 000, σ_e=0.015, rings=2.0%, BH=6%

Usage
-----
    from src.noise.noise_model import NoiseConfig, apply_noise, PRESETS
    noisy = apply_noise(clean_projections, PRESETS["medium"])
"""

from __future__ import annotations
from dataclasses import dataclass
import numpy as np


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class NoiseConfig:
    """
    Parameters for the realistic cone-beam noise pipeline.

    Args:
        photon_count:     Blank-scan photon count I0 per detector pixel.
                          Lower → more Poisson noise.
                          Typical: 2000 (high dose) – 100000 (low dose).
        gaussian_sigma:   Std-dev of Gaussian electronic noise as a
                          fraction of the mean projection optical depth.
        ring_strength:    Std-dev of per-column gain variation (fraction).
        beam_hardening:   Quadratic BH coefficient on optical depth.
        scatter_fraction: Uniform scatter as fraction of mean τ.
        seed:             RNG seed.
    """
    photon_count:     float = 10_000.0
    gaussian_sigma:   float = 0.005
    ring_strength:    float = 0.010
    beam_hardening:   float = 0.030
    scatter_fraction: float = 0.005
    seed:             int   = 42


PRESETS: dict[str, NoiseConfig] = {
    "noiseless": NoiseConfig(photon_count=1e9,    gaussian_sigma=0.0,
                             ring_strength=0.0,   beam_hardening=0.0,
                             scatter_fraction=0.0),
    "low":       NoiseConfig(photon_count=50_000, gaussian_sigma=0.002,
                             ring_strength=0.005, beam_hardening=0.01,
                             scatter_fraction=0.002),
    "medium":    NoiseConfig(photon_count=10_000, gaussian_sigma=0.005,
                             ring_strength=0.010, beam_hardening=0.03,
                             scatter_fraction=0.005),
    "high":      NoiseConfig(photon_count=2_000,  gaussian_sigma=0.015,
                             ring_strength=0.020, beam_hardening=0.06,
                             scatter_fraction=0.010),
}


# ---------------------------------------------------------------------------
# Individual noise components  (all operate on dimensionless τ)
# ---------------------------------------------------------------------------

def _beam_hardening(tau: np.ndarray, coeff: float) -> np.ndarray:
    """τ → τ + coeff·τ²  (cupping, polychromatic effect, pre-detection)."""
    return (tau + coeff * tau * tau).astype(np.float32)


def _poisson(tau: np.ndarray, I0: float,
             rng: np.random.Generator) -> np.ndarray:
    """
    Photon-counting Poisson noise on dimensionless optical depth τ.

    Steps:
        I_transmitted = I0 · exp(-τ)          ideal detector signal
        Ĩ            ~ Poisson(I_transmitted)  quantum noise
        τ̃             = -log(Ĩ / I0)           estimated optical depth
    """
    tau64        = tau.astype(np.float64)
    transmitted  = I0 * np.exp(-tau64)
    # Clip to avoid Poisson overflow for very small τ regions
    transmitted  = np.clip(transmitted, 0.0, None)
    noisy_counts = rng.poisson(transmitted).astype(np.float64)
    # Protect log from zero counts
    noisy_tau    = -np.log(np.clip(noisy_counts / I0, 1e-9, None))
    return noisy_tau.astype(np.float32)


def _gaussian(tau: np.ndarray, sigma_frac: float,
              rng: np.random.Generator) -> np.ndarray:
    """
    Additive Gaussian electronic noise.
    sigma_frac is expressed as fraction of mean(τ) so it scales
    sensibly regardless of dose or path length.
    """
    sigma = sigma_frac * float(np.mean(tau)) + 1e-6
    return (tau + rng.normal(0.0, sigma, tau.shape)).astype(np.float32)


def _rings(tau: np.ndarray, strength: float,
           rng: np.random.Generator) -> np.ndarray:
    """
    Per-column multiplicative gain variation → ring artefacts in recon.
    Applied along the last axis (detector columns).
    """
    n_cols = tau.shape[-1]
    gains  = rng.normal(1.0, strength, size=(1, 1, n_cols)).astype(np.float32)
    return (tau * gains).astype(np.float32)


def _scatter(tau: np.ndarray, fraction: float) -> np.ndarray:
    """Uniform additive scatter floor (raises background, lowers contrast)."""
    return (tau + fraction * float(np.mean(tau))).astype(np.float32)


# ---------------------------------------------------------------------------
# Full pipeline
# ---------------------------------------------------------------------------

def apply_noise(projections: np.ndarray,
                cfg: NoiseConfig | None = None) -> np.ndarray:
    """
    Apply the full realistic cone-beam noise pipeline.

    Args:
        projections: Clean projection stack (n_angles, det_rows, det_cols)
                     in units of  mm · cm⁻¹  (raw line integrals from
                     the ConeBeamProjector).
        cfg:         NoiseConfig. Defaults to PRESETS["medium"].

    Returns:
        Noisy projections in the same units (mm · cm⁻¹), float32.
    """
    if cfg is None:
        cfg = PRESETS["medium"]

    rng  = np.random.default_rng(cfg.seed)
    proj = projections.astype(np.float32).copy()

    # ── Step 1: convert mm·cm⁻¹ → dimensionless optical depth τ ─────────────
    MM_TO_CM = 0.1
    tau = proj * MM_TO_CM      # now dimensionless: cm⁻¹·cm

    # ── Step 2: beam hardening (pre-detection) ────────────────────────────────
    if cfg.beam_hardening > 0:
        tau = _beam_hardening(tau, cfg.beam_hardening)

    # ── Step 3: Poisson quantum noise ─────────────────────────────────────────
    tau = _poisson(tau, cfg.photon_count, rng)

    # ── Step 4: Gaussian electronic noise ─────────────────────────────────────
    if cfg.gaussian_sigma > 0:
        tau = _gaussian(tau, cfg.gaussian_sigma, rng)

    # ── Step 5: ring artefacts ────────────────────────────────────────────────
    if cfg.ring_strength > 0:
        tau = _rings(tau, cfg.ring_strength, rng)

    # ── Step 6: scatter ────────────────────────────────────────────────────────
    if cfg.scatter_fraction > 0:
        tau = _scatter(tau, cfg.scatter_fraction)

    tau = np.clip(tau, 0.0, None)

    # ── Step 7: convert back τ → mm·cm⁻¹ ─────────────────────────────────────
    noisy_proj = tau / MM_TO_CM

    return noisy_proj.astype(np.float32)
