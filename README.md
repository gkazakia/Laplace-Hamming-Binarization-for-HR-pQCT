# Laplace-Hamming Binarization for HR-pQCT (Scanco XtremeCT II)

A clean, **calibration-free** Python reimplementation of Scanco IPL's
`fft_laplace_hamming` + `norm_max` + `threshold` pipeline for binarizing
HR-pQCT AIM volumes.

Validated on a variety of samples (n > 80) with different scanning protocol against IPL ground truth:
**Dice = 0.999** .

## What it does

Takes a Scanco grayscale `.AIM` file and a periosteal (filled bone) contour,
and produces a binary bone mask numerically equivalent to IPL's LH workflow.
Every parameter is recovered directly from the IPL processing log and the
AIM file headers — there are **no per-scanner or per-sample calibration
constants**.

## Pipeline

Per the IPL `EVAL_LH_EFF` log:

1. Read grayscale AIM in **native int16** units (linear attenuation × Mu_Scaling)
2. IPL pre-FFT preprocessing: `bounding_box_cut border=1` + `offset_add 1 1 1` + `fill_offset_duplicate`
3. Mirror-pad x/y/z to the next power of 2 (`D3P_FFT_AdjustDimensionsMirror`)
4. Apply `fft_laplace_hamming` with `laplace_eps=0.45`, `lp_cut_off_freq=0.30`, `hamming_amp=1.0`
5. Crop back to the extended frame
6. `norm_max max=200000 type_out=short` (float → int16, clip to ±32767)
7. `threshold lower=475 upper=1000` permille (i.e. int16 ≥ 15564)
8. Undo +1 boundary, mask with periosteal contour
9. *(Optional)* `cl_nr_extract min_number=70` connected-component cleanup

## Filter formula

```
H(k) = (2π)² · [(1 − ε) + ε · |k|²] · W(|k|)

where
  |k|    = radial frequency in physical units (cycles / mm)
  W(|k|) = 0.5 + 0.5 · cos(π · |k| / k_lp)     for |k| < k_lp
         = 0                                    otherwise
  k_lp   = lp_cut_off_freq / voxel_size_mm     ( = 4.943 lp/mm @ 0.0607 mm)
```

The `(2π)²` prefactor is the continuous Laplacian eigenvalue and recovers
the absolute scale of IPL's float output without any per-scanner calibration.

### Note on the "Hamming" name

Despite IPL's command being named `fft_laplace_hamming`, the actual cosine
window kernel that reproduces IPL's binary output is the **Hann window**
(`0.5 + 0.5·cos`), not the textbook Hamming window (`0.54 + 0.46·cos`).

A consistent interpretation of IPL's `hamming_amp` parameter is:
```
W = (1 − amp/2) + (amp/2) · cos(π · |k| / k_lp)
```
With the log's default `hamming_amp=1.0` this gives Hann (0.5/0.5).
With `hamming_amp ≈ 0.92` it would give textbook Hamming (0.54/0.46).

## Usage

### IDE (Spyder) mode

Edit the `USER CONFIGURATION` block at the top of
`Laplace_Hamming_Binarization.py`:

```python
INPUT_AIM_PATH        = r"D:\Research\...\C0002398_version1.AIM"
OUTPUT_DIR            = r"D:\Research\...\ormir outputs"
FILLED_BONE_MASK_PATH = r"D:\Research\...\C0002398_version1_PRX_mask.nii.gz"
IPL_GT_PATH           = None   # set to an AIM path to compute Dice
```

Then run the file.

### Command line

```bash
python Laplace_Hamming_Binarization.py \
    scan.AIM \
    output_dir/ \
    bone_mask.{aim,nii.gz} \
    --ipl-gt ipl_seg.AIM \
    --cc-cleanup
```

The periosteal mask can be either a Scanco `.AIM` (auto-aligned via the AIM
origin) or a NIfTI already in the grayscale frame.

## Outputs

For input `scan.AIM` in `output_dir/`:

- `scan_LH_Binary.nii.gz` — binarized bone mask (within periosteal ROI)
- `scan_LH_preview.png` — axial slice preview (3-panel if `--ipl-gt` given)

## Dependencies

```
numpy
scipy
SimpleITK
matplotlib
```

No `itk` / `ScancoImageIO` requirement — AIM files are parsed directly,
which avoids ITK's HU rescaling and the spurious ±24 char-offset issues.

## References

- **LH method**: Laib A., Rüegsegger P. (1999). Calibration of trabecular bone
  structure measurements of in vivo three-dimensional peripheral quantitative
  computed tomography with 28-µm-resolution microcomputed tomography.
  *Bone* 24(1), 35–39. PMID 10227372.
- **Application to second-gen HR-pQCT**: Sadoughi S., Subramanian A., Ramil G.,
  Burghardt A.J., Kazakia G.J. (2023). A Laplace-Hamming Binarization
  Approach for Second-Generation HR-pQCT Rescues Fine Feature Segmentation.
  *JBMR* 38(7), 1006–1014.

## Authors

Yihua Zhu — Kazakia Lab (BQRL), UCSF Musculoskeletal Quantitative Imaging
Research Group.
