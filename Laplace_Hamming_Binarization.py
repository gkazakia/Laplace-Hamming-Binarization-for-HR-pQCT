"""
laplace_hamming_binarization.py
================================
Laplace-Hamming frequency-domain binarization of Scanco HR-pQCT AIM files.

This is a clean, calibration-free reimplementation of Scanco IPL's
fft_laplace_hamming + norm_max + threshold pipeline.  Unlike the previous
implementation, there is **no IPL <-> Python cross-calibration**: every
parameter and unit convention is recovered directly from the IPL processing
log and the AIM file headers.

Pipeline (per IPL EVAL_LH_EFF log)
----------------------------------
1.  Read grayscale AIM in *native* int16 units (linear attenuation x mu_scaling)
2.  Apply IPL preprocessing:
        bounding_box_cut border=1 1 1
        offset_add        1 1 1
        fill_offset_duplicate            (pads edges by 1 voxel with duplicate values)
3.  Mirror-pad x/y/z to the next power of 2 (IPL: D3P_FFT_AdjustDimensionsMirror)
4.  Apply fft_laplace_hamming with parameters from the log:
        laplace_eps      = 0.45
        lp_cut_off_freq  = 0.30
        hamming_amp      = 1.0    (-> Hann window; see note below)
5.  Crop back to (Nz+2, Ny+2, Nx+2)
6.  norm_max max=200000 type_out=short   (float -> int16, clip to +/-32767)
7.  threshold lower=475 upper=1000 (permille of int16 range)
        -> bone where  475/1000 * 32767 <= value <= 32767  (i.e. >= 15564)
8.  Crop +1 boundary, mask with periosteal contour (peel_iter=0)
9.  Optional: connected-component cleanup (cl_nr_extract min=70)

Filter formula
--------------
    H(k) = (2*pi)^2 * [ (1 - eps) + eps * |k|^2 ] * W(|k|)

where
    |k|         = radial frequency in physical units (cycles / mm)
    W(|k|)      = 0.5 + 0.5 * cos(pi * |k| / k_lp)   for |k| < k_lp
                = 0                                   otherwise
    k_lp        = lp_cut_off_freq * 2 * k_Nyquist
                = lp_cut_off_freq / voxel_size_mm     ( = 4.942 lp/mm @ 0.0607 mm)

Note on the "Hamming" name
--------------------------
Despite the IPL command being named "fft_laplace_hamming", the actual cosine
window kernel that reproduces IPL's output is the **Hann window**
(0.5 + 0.5*cos), not the textbook Hamming window (0.54 + 0.46*cos).  This was
verified empirically on PFJ021/L0003481 where Hann variant gives Dice > 0.997.

A consistent interpretation of IPL's `hamming_amp` parameter is:
    W = (1 - hamming_amp/2) + (hamming_amp/2) * cos(pi*|k|/k_lp)
With the log's default `hamming_amp=1.0` this gives Hann (0.5/0.5).
With `hamming_amp ~ 0.92` it would give textbook Hamming (0.54/0.46).

The (2*pi)^2 factor is the eigenvalue prefactor of (-Laplacian) in continuous
Fourier; including it recovers the absolute scale of IPL's float output
*without* any per-scanner calibration.

Authors
-------
    Yihua Zhu - Kazakia Lab (BQRL), UCSF Musculoskeletal Quantitative Imaging
"""

import os
import struct
import argparse
import numpy as np
import SimpleITK as sitk
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.fft import fftn, ifftn
from scipy.ndimage import label


# ============================================================================
# USER CONFIGURATION  (used when run in IDE mode with no CLI args)
# ============================================================================
# Default paths used by the original Kazakia Lab PFJOA pipeline.  Override on
# the command line, or edit here for Spyder use.

INPUT_AIM_PATH        = r"R:\Research\Kazakia_Lab\PFJOA\XCT_masks\PFJ016_R\C0002398_version1.AIM"
OUTPUT_DIR            = r"R:\Research\Kazakia_Lab\PFJOA\XCT_masks\PFJ016_R\ormir outputs"
FILLED_BONE_MASK_PATH = r"R:\Research\Kazakia_Lab\PFJOA\XCT_masks\PFJ016_R\ormir outputs\C0002398_version1_PRX_mask.nii.gz"
IPL_GT_PATH           = None     # set to an AIM path to compute Dice on a known sample


# ============================================================================
# IPL FILTER PARAMETERS (parsed directly from EVAL_LH_EFF log)
# ----------------------------------------------------------------------------
# Do NOT modify these without verifying against the corresponding log entry.
# ============================================================================
LAPLACE_EPS     = 0.45        # fft.laplace.eps
LP_CUT_OFF_FREQ = 0.30        # fft.lp_cut_off_freq (units of Nyquist/2)
HAMMING_AMP     = 1.0         # fft.laplace_hamming.amplitude
NORM_MAX        = 200000.0    # norm_max max value
LOWER_PERMILLE  = 475.0       # threshold lower (permille of int16 range)
UPPER_PERMILLE  = 1000.0      # threshold upper (permille of int16 range)
INT16_MAX       = 32767

# Post-processing (cl_nr_extract per cort_seg / trab_seg in log)
CC_MIN_VOXELS_CORT = 35       # cort cl_nr_extract -min_number
CC_MIN_VOXELS_TRAB = 70       # trab cl_nr_extract -min_number
CC_STRUCT_6   = np.array(
    [[[0, 0, 0], [0, 1, 0], [0, 0, 0]],
     [[0, 1, 0], [1, 1, 1], [0, 1, 0]],
     [[0, 0, 0], [0, 1, 0], [0, 0, 0]]], dtype=np.int32)


# ============================================================================
# 1. AIM v020 reader (direct, no ITK rescaling quirks)
# ============================================================================
def read_aim_v020(path):
    """
    Read a Scanco AIM v020 file directly from disk.

    AIM v020 layout:
        [0:20]         preheader (5 int32):
                          [0] format version (=16 for v020)
                          [1] image structure size (bytes)
                          [2] processing log size (bytes)
                          [3] image data size (bytes)
                          [4] (unused)
        [20:20+S]      image structure (S = preheader[1])
        [20+S:20+S+L]  processing log (ASCII, L = preheader[2])
        [20+S+L:...]   image data (uncompressed)

    Within the image structure, fields are 32-bit ints:
        idx 6, 7,  8: global position (voxel units)        - pos.x, pos.y, pos.z
        idx 9,10, 11: dimensions (voxel units)             - dim.x, dim.y, dim.z

    Element size is parsed from the processing log
    ('Orig-ISQ-Dim-um' / 'Orig-ISQ-Dim-p').

    Data type is determined by image_data_size / (dim.x * dim.y * dim.z):
        1 byte/voxel -> uint8  (binary / char AIMs)
        2 byte/voxel -> int16  (grayscale AIMs)
    """
    with open(path, 'rb') as f:
        buf = f.read()

    # AIM v020 preheader: 5 int32 values.
    #   pre[0] = 20   (preheader length itself, in bytes)
    #   pre[1] = 140  (image structure length, in bytes)
    #   pre[2] = N    (processing log length, in bytes)
    #   pre[3] = N    (image data length, in bytes)
    #   pre[4] = 0
    pre = struct.unpack('5i', buf[:20])
    header_size, log_size, data_size = pre[1], pre[2], pre[3]
    if pre[0] != 20:
        raise ValueError(f'{path}: not an AIM v020 (preheader[0] = {pre[0]}, expected 20)')

    # --- image structure: pos at idx 6..8, dim at idx 9..11 ---
    n_ints = header_size // 4
    fields = struct.unpack(f'{n_ints}i', buf[20:20 + header_size])
    pos_vx = (fields[6], fields[7], fields[8])
    dim    = (fields[9], fields[10], fields[11])
    dx, dy, dz = dim
    bpv = data_size // (dx * dy * dz)
    if dx * dy * dz * bpv != data_size:
        raise ValueError(f'{path}: dim {dim} inconsistent with data_size {data_size}')

    # --- processing log: ASCII ---
    log = buf[20 + header_size : 20 + header_size + log_size].decode('latin-1',
                                                                     errors='replace')

    # --- parse element size from log ---
    spacing_mm = _parse_element_size_mm(log)

    # --- image data ---
    data_start = 20 + header_size + log_size
    raw = buf[data_start : data_start + data_size]
    if bpv == 1:
        arr = np.frombuffer(raw, dtype=np.uint8).reshape(dz, dy, dx).copy()
    elif bpv == 2:
        arr = np.frombuffer(raw, dtype=np.int16).reshape(dz, dy, dx).copy()
    else:
        raise ValueError(f'{path}: unsupported bytes/voxel = {bpv}')

    # --- origin in mm = global voxel position * element size ---
    origin_mm = np.array(pos_vx) * spacing_mm   # (x, y, z) mm

    # --- parse Mu_Scaling, mu_water, density slope/intercept from log ---
    calibration = _parse_calibration(log)

    return {
        'arr':        arr,                  # (z, y, x), native units
        'dim_xyz':    dim,
        'pos_vx_xyz': np.array(pos_vx, dtype=np.int64),
        'origin_mm':  origin_mm,
        'spacing_mm': spacing_mm,
        'bpv':        bpv,
        'log':        log,
        **calibration,
    }


def _parse_element_size_mm(proc_log):
    """
    Parse element size from the AIM proc log.  Looks for either of:
        Orig-ISQ-Dim-p          2304   2304    168
        Orig-ISQ-Dim-um       139852 139852  10197
    or, for derived (GOBJ-based) AIMs:
        Orig-GOBJ-Dim-p         2304   2304    168
        Orig-GOBJ-Dim-um      139852 139852  10197

    Returns an (x, y, z) spacing in mm.
    """
    p_line  = None
    um_line = None
    for line in proc_log.splitlines():
        if 'Orig-ISQ-Dim-p' in line or 'Orig-GOBJ-Dim-p' in line:
            p_line = line
        elif 'Orig-ISQ-Dim-um' in line or 'Orig-GOBJ-Dim-um' in line:
            um_line = line
    if not (p_line and um_line):
        raise ValueError('AIM proc log missing Orig-{ISQ,GOBJ}-Dim-{p,um} lines')

    p_vals  = [int(s)   for s in p_line.split()  if s.lstrip('-').isdigit()]
    um_vals = [int(s)   for s in um_line.split() if s.lstrip('-').isdigit()]
    if len(p_vals) < 3 or len(um_vals) < 3:
        raise ValueError(f'Could not parse ISQ-Dim lines: {p_line} | {um_line}')

    # element size [um] = ISQ um / ISQ pixels;  convert um -> mm
    el_um = np.array(um_vals[-3:], dtype=np.float64) / np.array(p_vals[-3:], dtype=np.float64)
    return el_um * 1e-3   # mm


def _parse_calibration(proc_log):
    """Extract Mu_Scaling, HU mu_water, density slope/intercept from the log."""
    out = {'mu_scaling': None, 'mu_water': None,
           'density_slope': None, 'density_intercept': None}
    for line in proc_log.splitlines():
        s = line.strip()
        if s.startswith('Mu_Scaling'):
            out['mu_scaling'] = float(s.split()[-1])
        elif s.startswith('HU: mu water'):
            out['mu_water'] = float(s.split()[-1])
        elif s.startswith('Density: slope'):
            out['density_slope'] = float(s.split()[-1])
        elif s.startswith('Density: intercept'):
            out['density_intercept'] = float(s.split()[-1])
    return out


# ============================================================================
# 2. Mirror-pad to next power of 2 (IPL: D3P_FFT_AdjustDimensionsMirror)
# ============================================================================
def next_power_of_two(n):
    return 1 if n <= 1 else 2 ** int(np.ceil(np.log2(n)))


def mirror_pad_to_pow2(arr):
    """
    Mirror-pad each dimension of `arr` (z, y, x) up to the next power of 2,
    matching IPL's D3P_FFT_AdjustDimensionsMirror behavior.  Padding is split
    symmetrically before and after the original volume.

    Returns the padded array and the (z, y, x) slice that recovers the
    original region.
    """
    pad_widths = []
    slices = []
    for n in arr.shape:
        n_target = next_power_of_two(n)
        pad_total = n_target - n
        pad_lo = pad_total // 2
        pad_hi = pad_total - pad_lo
        pad_widths.append((pad_lo, pad_hi))
        slices.append(slice(pad_lo, pad_lo + n))
    padded = np.pad(arr, pad_widths, mode='reflect')
    return padded, tuple(slices)


# ============================================================================
# 3. FFT Laplace-Hamming filter (matches IPL's D3P_FFT_LaplaceHamming_M)
# ============================================================================
def fft_laplace_hamming(vol, voxel_size_mm,
                        laplace_eps=LAPLACE_EPS,
                        lp_cut_off_freq=LP_CUT_OFF_FREQ,
                        hamming_amp=HAMMING_AMP,
                        dtype=np.float32):
    """
    Apply the IPL Laplace-Hamming filter to `vol`.

    Filter (in 3D Fourier domain):
        H(k) = (2*pi)^2 * [ (1 - eps) + eps * |k|^2 ] * W(|k|)

    with |k| in cycles/mm and cosine low-pass window
        W(|k|)  = (1 - amp/2) + (amp/2) * cos(pi * |k| / k_lp)   for |k| < k_lp
                = 0                                                otherwise

    where amp = hamming_amp.  With amp=1.0 (IPL default) this is a Hann window
    (0.5 + 0.5*cos), which experimentally reproduces IPL's binarization output
    far more accurately than the textbook Hamming window (0.54/0.46).
    See module docstring for details.

    Cutoff:
        k_lp = lp_cut_off_freq * 2 * k_Nyquist
             = lp_cut_off_freq / voxel_size_mm
             ( = 4.9426 lp/mm for lp_cut_off_freq=0.30, voxel=0.0607 mm)

    The (2*pi)^2 prefactor is the continuous Laplacian eigenvalue; including it
    recovers the absolute scale of IPL's float output without per-scanner
    calibration.

    Parameters
    ----------
    vol : (Nz, Ny, Nx) array
        Real input.  Should already be mirror-padded to power-of-2 dimensions.
    voxel_size_mm : float or (3,) array
        Isotropic or per-axis voxel size in mm.
    dtype : numpy dtype
        Working precision (IPL uses float32; default here matches).

    Returns
    -------
    out : (Nz, Ny, Nx) array of `dtype`
        Filtered image.
    """
    vol = np.asarray(vol, dtype=dtype)
    if np.isscalar(voxel_size_mm):
        sp = np.array([voxel_size_mm] * 3, dtype=dtype)
    else:
        sp = np.asarray(voxel_size_mm, dtype=dtype)
        if sp.size != 3:
            raise ValueError('voxel_size_mm must be a scalar or 3-element array')

    nz, ny, nx = vol.shape
    # Physical frequency coordinates (cycles per mm)
    kx = np.fft.fftfreq(nx, d=float(sp[0])).astype(dtype)
    ky = np.fft.fftfreq(ny, d=float(sp[1])).astype(dtype)
    kz = np.fft.fftfreq(nz, d=float(sp[2])).astype(dtype)
    KZ, KY, KX = np.meshgrid(kz, ky, kx, indexing='ij')
    K2   = KX * KX + KY * KY + KZ * KZ            # cycles^2 / mm^2
    Kmag = np.sqrt(K2)

    # Low-pass cutoff (cycles/mm)
    k_nyq_min = dtype(1.0) / (dtype(2.0) * np.min(sp))
    k_lp      = dtype(lp_cut_off_freq) * dtype(2.0) * k_nyq_min

    # Generalized cosine window:
    #     W = (1 - amp/2) + (amp/2)*cos(pi*|k|/k_lp)
    # amp = 1.0 -> Hann (0.5 + 0.5*cos), reproduces IPL.
    # amp = 0.92 -> textbook Hamming (0.54 + 0.46*cos).
    half_amp = dtype(hamming_amp) * dtype(0.5)
    win = np.where(
        Kmag < k_lp,
        (dtype(1.0) - half_amp) + half_amp * np.cos(np.pi * Kmag / k_lp),
        dtype(0.0),
    ).astype(dtype)

    # Filter:  (2*pi)^2 * [ (1-eps) + eps * |k|^2 ] * W
    TPI2 = dtype((2.0 * np.pi) ** 2)
    eps  = dtype(laplace_eps)
    H = TPI2 * ((dtype(1.0) - eps) + eps * K2) * win

    # workers=-1 parallelises the transform across all cores (pocketfft gives
    # bit-identical results regardless of worker count). The padded volume is
    # power-of-2 and large, so this is a free, exact speedup.
    F = fftn(vol, workers=-1)
    F *= H
    out = np.real(ifftn(F, workers=-1)).astype(dtype)
    return out


# ============================================================================
# 4. norm_max (float -> int16)
# ============================================================================
def norm_max(arr, max_val=NORM_MAX, type_out_max=INT16_MAX):
    """
    IPL's norm_max:  scale so that `max_val` maps to `type_out_max`
    (and -max_val to -type_out_max).  Clip outside the range and cast to int16.

    out = clip(arr * (type_out_max / max_val), -type_out_max, type_out_max)
    """
    scale  = float(type_out_max) / float(max_val)   # = 0.16384 for defaults
    scaled = arr * scale
    np.clip(scaled, -type_out_max, type_out_max, out=scaled)
    return np.rint(scaled).astype(np.int16)


# ============================================================================
# 5. Threshold (permille of int16 range)
# ============================================================================
def threshold_permille(arr, lower_perm=LOWER_PERMILLE,
                       upper_perm=UPPER_PERMILLE, int16_max=INT16_MAX):
    """IPL's /threshold with -unit 0 (raw permille of int16)."""
    lo = lower_perm / 1000.0 * int16_max
    hi = upper_perm / 1000.0 * int16_max
    return ((arr >= lo) & (arr <= hi)).astype(np.uint8)


# ============================================================================
# 6. Connected-component cleanup (IPL cl_nr_extract -min_number=N)
# ============================================================================
def cl_nr_extract(binary, min_voxels=CC_MIN_VOXELS_TRAB):
    """Drop connected components smaller than `min_voxels` (6-connectivity)."""
    labeled, _ = label(binary, structure=CC_STRUCT_6)
    sizes      = np.bincount(labeled.ravel())
    small      = np.where(sizes < min_voxels)[0]
    small      = small[small != 0]
    return binary & ~np.isin(labeled, small)


# ============================================================================
# 7. Spatial alignment helpers
# ============================================================================
def voxel_offset(child_origin_mm, parent_origin_mm, spacing_mm):
    """
    Integer (dz, dy, dx) voxel offset of `child` relative to `parent`.

    AIM origin is stored in (x, y, z) order; numpy arrays are (z, y, x).
    """
    off_mm = np.asarray(child_origin_mm) - np.asarray(parent_origin_mm)
    off_vx = off_mm / np.asarray(spacing_mm)
    off_i  = np.rint(off_vx).astype(np.int64)
    if not np.allclose(off_vx, off_i, atol=1e-3):
        print(f'  WARNING: non-integer voxel offset {off_vx} -> {off_i}')
    return off_i[::-1]   # xyz -> zyx


def embed_in_frame(child, child_origin_mm, parent_origin_mm, spacing_mm,
                   parent_shape, fill=0):
    """Place `child` into a `parent_shape`-sized array at its proper offset."""
    dz, dy, dx = voxel_offset(child_origin_mm, parent_origin_mm, spacing_mm)
    full = np.full(parent_shape, fill, dtype=child.dtype)
    cz, cy, cx = child.shape
    # Safe clipping if child extends outside parent
    z0, y0, x0 = max(0, dz), max(0, dy), max(0, dx)
    z1, y1, x1 = min(parent_shape[0], dz + cz), \
                 min(parent_shape[1], dy + cy), \
                 min(parent_shape[2], dx + cx)
    sz, sy, sx = z0 - dz, y0 - dy, x0 - dx
    full[z0:z1, y0:y1, x0:x1] = child[sz:sz + (z1 - z0),
                                       sy:sy + (y1 - y0),
                                       sx:sx + (x1 - x0)]
    return full


def dice(a, b):
    a = a.astype(bool); b = b.astype(bool)
    inter = np.logical_and(a, b).sum()
    s = a.sum() + b.sum()
    return float(2.0 * inter / s) if s > 0 else 1.0


# ============================================================================
# 8. Save outputs
# ============================================================================
def save_nifti(arr_zyx, spacing_mm_xyz, origin_mm_xyz, path):
    img = sitk.GetImageFromArray(arr_zyx.astype(np.uint8))
    img.SetSpacing([float(s) for s in spacing_mm_xyz])
    img.SetOrigin([float(o) for o in origin_mm_xyz])
    sitk.WriteImage(img, path)


# ============================================================================
# 9. Pipeline
# ============================================================================
def run_lh_binarization(input_aim_path,
                        output_dir,
                        filled_bone_mask_path,
                        ipl_gt_path=None,
                        cc_cleanup=False,
                        save_preview=True,
                        verbose=True):
    """
    Run the calibration-free Laplace-Hamming binarization pipeline.

    Parameters
    ----------
    input_aim_path        : str   Grayscale AIM (Scanco native int16)
    output_dir            : str   Directory for output files
    filled_bone_mask_path : str   Periosteal contour: either a binary NIfTI in
                                  grayscale-image coordinates, or an AIM with
                                  its own origin (auto-aligned).
    ipl_gt_path           : str   (optional) IPL SEG.AIM ground truth for Dice
    cc_cleanup            : bool  Apply connected-component cleanup (min 70 vox)
    save_preview          : bool  Write a 2-panel PNG preview
    verbose               : bool  Print pipeline diagnostics

    Returns
    -------
    dict with keys:
        'lh_binary'         : (Nz, Ny, Nx) uint8, in grayscale-image frame
        'lh_float'          : padded float LH output (for QC)
        'lh_int16'          : padded int16 (post-norm_max)
        'dice'              : Dice against IPL GT (None if no GT given)
    """
    os.makedirs(output_dir, exist_ok=True)
    basename     = os.path.splitext(os.path.basename(input_aim_path))[0]
    out_binary   = os.path.join(output_dir, basename + '_LH_Binary.nii.gz')
    out_png      = os.path.join(output_dir, basename + '_LH_preview.png')

    log = print if verbose else (lambda *a, **kw: None)

    # ---- Step 1: load grayscale AIM ----------------------------------------
    log('[1/8] Loading grayscale AIM:', input_aim_path)
    gray = read_aim_v020(input_aim_path)
    native = gray['arr']                              # (z, y, x), native int16
    sp = gray['spacing_mm']                            # (x, y, z) mm
    log(f'    shape    : {native.shape}  (z,y,x)')
    log(f'    spacing  : {sp} mm (xyz)')
    log(f'    origin   : {gray["origin_mm"]} mm')
    log(f'    native   : min={int(native.min())}, max={int(native.max())}, '
        f'mean={native.mean():.2f}, std={native.std():.2f}')

    # ---- Step 2: IPL pre-FFT preprocessing ---------------------------------
    # bounding_box_cut -border 1 1 1 + offset_add 1 1 1 + fill_offset_duplicate.
    # Together these extend the image by 1 voxel in every direction with the
    # edge values duplicated.  Shape grows from (Nz, Ny, Nx) to (Nz+2, Ny+2, Nx+2).
    log('[2/8] IPL pre-FFT preprocessing: bbox_cut(border=1) + fill_offset_duplicate')
    extended = np.pad(native, ((1, 1), (1, 1), (1, 1)),
                      mode='edge').astype(np.float32)
    log(f'    extended shape: {extended.shape}')

    # ---- Step 3: mirror-pad to next power of 2 -----------------------------
    log('[3/8] Mirror-padding to next power of 2 (D3P_FFT_AdjustDimensionsMirror)')
    padded, ext_slc = mirror_pad_to_pow2(extended)
    log(f'    padded shape: {padded.shape}')

    # ---- Step 4: FFT Laplace-Hamming ---------------------------------------
    log('[4/8] FFT Laplace-Hamming filter')
    log(f'    eps        = {LAPLACE_EPS}')
    log(f'    lp_cut_off = {LP_CUT_OFF_FREQ}')
    log(f'    ham_amp    = {HAMMING_AMP}    (-> Hann window @ amp=1.0)')
    lh_float = fft_laplace_hamming(
        padded, voxel_size_mm=sp,
        laplace_eps=LAPLACE_EPS,
        lp_cut_off_freq=LP_CUT_OFF_FREQ,
        hamming_amp=HAMMING_AMP,
        dtype=np.float32,
    )
    log(f'    LH float stats (full padded volume):')
    log(f'      mean={lh_float.mean():.2f}, std={lh_float.std():.2f}, '
        f'min={lh_float.min():.0f}, max={lh_float.max():.0f}')

    # ---- Step 5: norm_max --------------------------------------------------
    log('[5/8] norm_max(200000) -> int16')
    lh_int16 = norm_max(lh_float, max_val=NORM_MAX)
    log(f'    int16 stats: mean={lh_int16.mean():.2f}, std={lh_int16.std():.2f}')

    # ---- Step 6: threshold + crop ------------------------------------------
    log(f'[6/8] threshold {LOWER_PERMILLE}/1000 - {UPPER_PERMILLE}/1000 permille')
    seg_padded = threshold_permille(lh_int16, LOWER_PERMILLE, UPPER_PERMILLE)
    seg_ext    = seg_padded[ext_slc]                   # back to (Nz+2, Ny+2, Nx+2)
    # Undo the +1 border to return to original (Nz, Ny, Nx)
    seg = seg_ext[1:-1, 1:-1, 1:-1]
    log(f'    pre-mask bone voxels (gray frame): {int(seg.sum()):,}')

    # ---- Step 7: mask with periosteal contour ------------------------------
    log('[7/8] Apply periosteal (filled bone) mask')
    bone_mask  = _load_bone_mask(filled_bone_mask_path, gray)
    seg_masked = (seg.astype(bool) & bone_mask.astype(bool)).astype(np.uint8)
    log(f'    post-mask bone voxels: {int(seg_masked.sum()):,}')

    if cc_cleanup:
        log(f'    cl_nr_extract: removing components < {CC_MIN_VOXELS_TRAB} voxels')
        before = int(seg_masked.sum())
        seg_masked = cl_nr_extract(seg_masked.astype(bool),
                                   min_voxels=CC_MIN_VOXELS_TRAB).astype(np.uint8)
        log(f'    removed: {before - int(seg_masked.sum()):,} '
            f'-> remaining: {int(seg_masked.sum()):,}')

    # ---- Save NIfTI --------------------------------------------------------
    save_nifti(seg_masked, sp, gray['origin_mm'], out_binary)
    log(f'    -> {out_binary}')

    # ---- Step 8: optional Dice vs IPL ground truth -------------------------
    dice_val = None
    gt_in_gray = None
    if ipl_gt_path is not None:
        log('[8/8] Compare with IPL ground truth')
        gt = read_aim_v020(ipl_gt_path)
        log(f'    GT shape: {gt["arr"].shape}, '
            f'unique values: {np.unique(gt["arr"])}')
        gt_bin = (gt['arr'] > 0).astype(np.uint8)
        gt_in_gray = embed_in_frame(
            gt_bin, gt['origin_mm'], gray['origin_mm'], sp,
            seg_masked.shape, fill=0,
        )
        dice_val = dice(seg_masked, gt_in_gray)
        log(f'    IPL bone voxels (in gray frame): {int(gt_in_gray.sum()):,}')
        log(f'    Our bone voxels (in gray frame): {int(seg_masked.sum()):,}')
        log(f'    Dice                             = {dice_val:.4f}')

    if save_preview:
        _save_preview(native, seg_masked, gt_in_gray, basename, out_png)

    return {
        'lh_binary':  seg_masked,
        'lh_float':   lh_float,
        'lh_int16':   lh_int16,
        'dice':       dice_val,
        'gray':       gray,
    }


# ============================================================================
# Internal helpers
# ============================================================================
def _load_bone_mask(path, gray_meta):
    """
    Load a periosteal-contour bone mask.  Supports two formats:
        - .AIM (Scanco char/binary) -- auto-aligned via header origin
        - .nii / .nii.gz NIfTI       -- must already be in grayscale frame
    Returns a uint8 array in the grayscale-image frame.
    """
    if path.lower().endswith('.aim'):
        m = read_aim_v020(path)
        bin_ = (m['arr'] > 0).astype(np.uint8)
        return embed_in_frame(
            bin_, m['origin_mm'], gray_meta['origin_mm'],
            gray_meta['spacing_mm'], gray_meta['arr'].shape, fill=0,
        )
    else:
        arr = sitk.GetArrayFromImage(sitk.ReadImage(path)).astype(np.uint8)
        if arr.shape != gray_meta['arr'].shape:
            raise ValueError(
                f'Bone mask shape {arr.shape} != grayscale shape '
                f'{gray_meta["arr"].shape}; cannot align without AIM origin'
            )
        return arr


def _save_preview(gray, lh_binary, gt_binary, basename, out_png):
    counts = lh_binary.sum(axis=(1, 2))
    z = int(np.argmax(counts)) if counts.max() > 0 else gray.shape[0] // 2

    pos = gray[gray > 0]
    vmin = float(np.percentile(pos, 1)) if pos.size else 0.0
    vmax = float(np.percentile(pos, 99)) if pos.size else float(gray.max())

    def overlay(g, m, rgb, alpha=0.5):
        g01 = np.clip((g - vmin) / (vmax - vmin + 1e-9), 0, 1)
        out = np.stack([g01, g01, g01], axis=-1)
        for c, v in enumerate(rgb):
            out[m.astype(bool), c] = out[m.astype(bool), c] * (1 - alpha) + v * alpha
        return out

    ncols = 3 if gt_binary is not None else 2
    fig, ax = plt.subplots(1, ncols, figsize=(5 * ncols, 5))
    fig.suptitle(f'{basename}  --  axial slice {z} / {gray.shape[0]}', fontsize=13)

    ax[0].imshow(gray[z], cmap='gray', vmin=vmin, vmax=vmax)
    ax[0].set_title('Grayscale')
    ax[0].axis('off')

    ax[1].imshow(overlay(gray[z], lh_binary[z], (1.0, 0.25, 0.0)))
    ax[1].set_title('LH binary (ours)')
    ax[1].axis('off')

    if gt_binary is not None:
        # difference overlay: green = both, red = ours only, blue = IPL only
        both     = lh_binary[z].astype(bool) & gt_binary[z].astype(bool)
        ours_only = lh_binary[z].astype(bool) & ~gt_binary[z].astype(bool)
        ipl_only = ~lh_binary[z].astype(bool) & gt_binary[z].astype(bool)
        g01 = np.clip((gray[z] - vmin) / (vmax - vmin + 1e-9), 0, 1)
        rgb = np.stack([g01, g01, g01], axis=-1)
        rgb[both,      :] = (0.0, 1.0, 0.0)
        rgb[ours_only, :] = (1.0, 0.0, 0.0)
        rgb[ipl_only,  :] = (0.0, 0.3, 1.0)
        ax[2].imshow(rgb)
        ax[2].set_title('green=both, red=ours only, blue=IPL only')
        ax[2].axis('off')

    plt.tight_layout()
    plt.savefig(out_png, dpi=150, bbox_inches='tight')
    plt.close(fig)


# ============================================================================
# CLI
# ============================================================================
def _parse_args():
    p = argparse.ArgumentParser(
        description='Calibration-free Laplace-Hamming binarization of Scanco HR-pQCT AIM files'
    )
    p.add_argument('aim', help='Input grayscale AIM file')
    p.add_argument('output_dir', help='Output directory')
    p.add_argument('bone_mask', help='Periosteal (filled bone) mask (NIfTI or AIM)')
    p.add_argument('--ipl-gt', help='Optional IPL SEG.AIM ground truth for Dice')
    p.add_argument('--cc-cleanup', action='store_true',
                   help='Apply connected-component cleanup (min 70 voxels)')
    p.add_argument('--no-preview', action='store_true')
    return p.parse_args()


if __name__ == '__main__':
    import sys
    if len(sys.argv) > 1:
        args = _parse_args()
        run_lh_binarization(
            input_aim_path        = args.aim,
            output_dir            = args.output_dir,
            filled_bone_mask_path = args.bone_mask,
            ipl_gt_path           = args.ipl_gt,
            cc_cleanup            = args.cc_cleanup,
            save_preview          = not args.no_preview,
        )
    else:
        run_lh_binarization(
            input_aim_path        = INPUT_AIM_PATH,
            output_dir            = OUTPUT_DIR,
            filled_bone_mask_path = FILLED_BONE_MASK_PATH,
            ipl_gt_path           = IPL_GT_PATH,
            cc_cleanup            = False,
        )
