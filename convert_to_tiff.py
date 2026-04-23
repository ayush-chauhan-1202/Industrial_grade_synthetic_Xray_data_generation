#!/usr/bin/env python3
"""
Convert Stage 1 .npy outputs to ImageJ-compatible TIFF stacks.

Converts:
    projections/clean.npy   → projections/clean.tif   (n_angles-frame stack)
    projections/noisy.npy   → projections/noisy.tif
    volume/phantom.npy      → volume/phantom.tif       (Nz-frame stack)

Each .tif opens in ImageJ as a stack where you can scroll through
angles (for projections) or axial slices (for the volume).

Usage
-----
    python convert_to_tiff.py                        # default: results/
    python convert_to_tiff.py --out-dir my_results   # custom folder
    python convert_to_tiff.py --bit 8                # 8-bit (smaller files)
    python convert_to_tiff.py --bit 16               # 16-bit (default)
    python convert_to_tiff.py --bit 32               # 32-bit float (lossless)

ImageJ tips after opening
--------------------------
    Image > Adjust > Brightness/Contrast  → auto-scale display
    Image > Stacks > Z Project            → max/mean intensity projection
    Analyze > Plot Profile                → line profile through a row
    File > Save As > AVI                  → animate the projection stack
    Plugins > FIJI > 3D Viewer           → volume render the phantom.tif
"""

import argparse
import struct
import sys
import zlib
from pathlib import Path

import numpy as np


# ---------------------------------------------------------------------------
# Minimal TIFF writer (no external deps beyond numpy)
# ---------------------------------------------------------------------------

def _write_tiff_stack(array: np.ndarray, path: Path, bit_depth: int) -> None:
    """
    Write a 3-D numpy array as a multi-page grayscale TIFF.

    Uses tifffile if available (best quality), otherwise falls back to
    imageio, and finally to a minimal hand-rolled TIFF writer so the
    script works with zero extra dependencies.

    Args:
        array:     3-D float32 array (n_frames, height, width).
        path:      Output .tif path.
        bit_depth: 8, 16, or 32.
    """
    # ── Try tifffile (best) ────────────────────────────────────────────────
    try:
        import tifffile
        if bit_depth == 8:
            out = _to_uint8(array)
        elif bit_depth == 16:
            out = _to_uint16(array)
        else:
            out = array.astype(np.float32)
        tifffile.imwrite(str(path), out, imagej=True,
                         metadata={"axes": "ZYX"})
        return
    except ImportError:
        pass

    # ── Try imageio ────────────────────────────────────────────────────────
    try:
        import imageio
        if bit_depth == 8:
            out = _to_uint8(array)
        elif bit_depth == 16:
            out = _to_uint16(array)
        else:
            out = array.astype(np.float32)
        imageio.mimwrite(str(path), [out[i] for i in range(out.shape[0])])
        return
    except ImportError:
        pass

    # ── Fallback: minimal TIFF writer (8-bit or 16-bit only) ──────────────
    if bit_depth == 32:
        print("    ⚠ tifffile/imageio not found — falling back to 16-bit "
              "(install tifffile for 32-bit float TIFF)")
        bit_depth = 16

    if bit_depth == 8:
        out = _to_uint8(array)
        dtype_code, bps = np.uint8, 8
    else:
        out = _to_uint16(array)
        dtype_code, bps = np.uint16, 16

    _write_minimal_tiff(out, path, bps)


def _to_uint8(arr: np.ndarray) -> np.ndarray:
    lo, hi = arr.min(), arr.max()
    if hi == lo:
        return np.zeros_like(arr, dtype=np.uint8)
    return ((arr - lo) / (hi - lo) * 255).clip(0, 255).astype(np.uint8)


def _to_uint16(arr: np.ndarray) -> np.ndarray:
    lo, hi = arr.min(), arr.max()
    if hi == lo:
        return np.zeros_like(arr, dtype=np.uint16)
    return ((arr - lo) / (hi - lo) * 65535).clip(0, 65535).astype(np.uint16)


def _write_minimal_tiff(stack: np.ndarray, path: Path, bps: int) -> None:
    """
    Hand-rolled multi-page TIFF writer (little-endian, uncompressed).
    Sufficient for ImageJ to read as a stack.
    """
    n_frames, H, W = stack.shape
    bytes_per_sample = bps // 8
    strip_bytes = H * W * bytes_per_sample

    # We'll write IFDs + image data sequentially.
    # Each IFD is 2 + 12*n_tags + 4 bytes.
    n_tags    = 11
    ifd_size  = 2 + 12 * n_tags + 4

    # Layout: [IFDs (all frames)] [image data (all frames)]
    ifd_block  = n_frames * ifd_size
    data_start = 8 + ifd_block   # 8-byte TIFF header

    def ifd_entry(tag, dtype, count, value_or_offset):
        # dtype: 3=SHORT, 4=LONG
        if dtype == 3:   # SHORT — value fits in 4 bytes
            return struct.pack("<HHHI", tag, dtype, count, value_or_offset)
        else:            # LONG
            return struct.pack("<HHII", tag, dtype, count, value_or_offset)

    buf = bytearray()

    # TIFF header (little-endian, magic 42)
    buf += b"II"                            # byte order
    buf += struct.pack("<H", 42)            # magic
    buf += struct.pack("<I", 8)             # offset to first IFD

    for f in range(n_frames):
        image_offset  = data_start + f * strip_bytes
        next_ifd      = (8 + (f + 1) * ifd_size) if f < n_frames - 1 else 0

        buf += struct.pack("<H", n_tags)
        buf += ifd_entry(256, 4, 1, W)              # ImageWidth
        buf += ifd_entry(257, 4, 1, H)              # ImageLength
        buf += ifd_entry(258, 3, 1, bps)            # BitsPerSample
        buf += ifd_entry(259, 3, 1, 1)              # Compression=None
        buf += ifd_entry(262, 3, 1, 1)              # PhotometricInterp=BlackIsZero
        buf += ifd_entry(278, 4, 1, H)              # RowsPerStrip
        buf += ifd_entry(279, 4, 1, strip_bytes)    # StripByteCounts
        buf += ifd_entry(273, 4, 1, image_offset)   # StripOffsets
        buf += ifd_entry(282, 4, 1, 72)             # XResolution (dummy)
        buf += ifd_entry(283, 4, 1, 72)             # YResolution (dummy)
        buf += ifd_entry(296, 3, 1, 2)              # ResolutionUnit=inch
        buf += struct.pack("<I", next_ifd)

    # Image data
    for f in range(n_frames):
        buf += stack[f].tobytes()

    path.write_bytes(bytes(buf))


# ---------------------------------------------------------------------------
# Conversion pipeline
# ---------------------------------------------------------------------------

def convert_file(npy_path: Path, out_path: Path,
                 bit_depth: int, label: str) -> None:
    """Load a .npy file, normalise, and save as TIFF stack."""
    print(f"\n  {label}")
    print(f"    Loading  {npy_path} …")

    arr = np.load(npy_path).astype(np.float32)
    print(f"    Shape    : {arr.shape}")
    print(f"    Range    : [{arr.min():.4f}, {arr.max():.4f}]")

    # Ensure 3-D  (frame, height, width)
    if arr.ndim == 2:
        arr = arr[np.newaxis]   # single slice → 1-frame stack

    n_frames, H, W = arr.shape
    print(f"    Frames   : {n_frames}  ({H}×{W} pixels each)")
    print(f"    Bit depth: {bit_depth}")

    _write_tiff_stack(arr, out_path, bit_depth)

    size_mb = out_path.stat().st_size / 1_048_576
    print(f"    Saved  → {out_path}  ({size_mb:.1f} MB)")


def main() -> None:
    p = argparse.ArgumentParser(
        description="Convert Stage 1 .npy outputs to ImageJ TIFF stacks",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--out-dir", default="results",
                   help="Stage 1 output directory (default: results)")
    p.add_argument("--bit", type=int, choices=[8, 16, 32], default=16,
                   help="Output bit depth (default: 16)")
    p.add_argument("--no-volume", action="store_true",
                   help="Skip phantom volume conversion (large file)")
    args = p.parse_args()

    out = Path(args.out_dir)

    targets = [
        (out / "projections" / "clean.npy",
         out / "projections" / "clean.tif",
         "Projections — clean  (one frame per angle)"),
        (out / "projections" / "noisy.npy",
         out / "projections" / "noisy.tif",
         "Projections — noisy"),
    ]
    if not args.no_volume:
        targets.append((
            out / "volume" / "phantom.npy",
            out / "volume"  / "phantom.tif",
            "Volume — phantom  (one frame per axial slice)",
        ))

    print("=" * 60)
    print("  NPY → TIFF converter for ImageJ")
    print("=" * 60)

    converted = 0
    for npy, tif, label in targets:
        if not npy.exists():
            print(f"\n  ⚠  Not found, skipping: {npy}")
            continue
        convert_file(npy, tif, args.bit, label)
        converted += 1

    print(f"\n{'='*60}")
    print(f"  Done — {converted} file(s) converted.")
    print()
    print("  Open in ImageJ:")
    print("    File > Open  →  select the .tif file")
    print("    Use the slider at the bottom to scroll through frames.")
    print()
    print("  Useful ImageJ commands:")
    print("    Image > Adjust > Brightness/Contrast  (Ctrl+Shift+C)")
    print("    Image > Stacks > Z Project")
    print("    Analyze > Plot Profile")
    print("    Image > Stacks > Start Animation (\\)")


if __name__ == "__main__":
    main()
