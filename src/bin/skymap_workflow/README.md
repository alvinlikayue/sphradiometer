# Harmonic Sphradiometer Skymap Workflow

This directory contains a stripped workflow for Kipp's harmonic
sphradiometer localization backend.

It adds workflow wrappers only.  The C library and Python
`RapidLocalization` implementation are unchanged from the Kipp source tree.

## Precompute

Precompute one set of harmonic localization objects for a PSD:

```bash
sph_skymap_precompute_pipe.py example_precompute.ini
condor_submit_dag precompute_skymap.dag
```

## Single Event

Create one skymap from a `gstlal_inspiral_calc_snr` output directory:

```bash
sph_skymap_from_calc_snr.py outputs/calc_snr/sim0 \
  --output-fits outputs/skymap/sim0.fits \
  --precalc-dir outputs/precalc/H1L1V1_asd_438/H1L1V1 \
  --bank-number 0 \
  --row-number 0 \
  --center-time 1243540000.0
```

## Manifest DAG

Create a harmonic-only skymap DAG from a manifest:

```bash
sph_skymap_injection_pipe.py example_injections.ini
condor_submit_dag injection_skymaps.dag
```

The manifest must contain at least:

```text
simulation_id,center_time,bank_number,row_number,calc_snr_dir
```

If `calc_snr_dir` is omitted, the DAG assumes
`outputs/calc_snr/sim<simulation_id>` relative to the workflow output root.
