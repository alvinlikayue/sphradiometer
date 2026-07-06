#!/usr/bin/env python3
"""Create one harmonic sphradiometer skymap from calc_snr XML dumps."""

import argparse
import glob
import os
import re
import time
from contextlib import contextmanager

import healpy as hp
import lal
import ligolw
import ligolw.utils
import numpy as np
from ligo.skymap import io as skymap_io
from sphradiometer.RapidLocalization import RapidLocalization, RapidLocalization_, rapidskyloc_io
from sphradiometer import healpix as sph_healpix
from sphradiometer import sphradiometer as sph

from sph_skymap_common import load_relative_psd_weights


class StageTimer:
	def __init__(self, enabled=False):
		self.enabled = enabled
		self.records = []
		self.start = time.perf_counter()

	@contextmanager
	def stage(self, label):
		if not self.enabled:
			yield
			return
		t0 = time.perf_counter()
		try:
			yield
		finally:
			self.records.append((label, time.perf_counter() - t0))

	def report(self):
		if not self.enabled:
			return
		print("timing total %.6f s" % (time.perf_counter() - self.start))
		for label, elapsed in self.records:
			print("timing %-24s %.6f s" % (label, elapsed))


def filename_gps_start(path):
	match = re.search(r"-(\d+)-\d+\.xml(?:\.gz)?$", os.path.basename(path))
	if not match:
		raise ValueError(f"could not parse GPS start time from SNR filename: {path}")
	return float(match.group(1))


def read_calc_snr(calc_snr_dir, bank_number, row_number):
	files = {
		os.path.basename(path).split("-")[0]: path
		for path in glob.glob(os.path.join(calc_snr_dir, "*SNR*.xml.gz"))
	}
	if not files:
		raise FileNotFoundError(f"no *SNR*.xml.gz files found in {calc_snr_dir}")

	full = {}
	aut = {}
	for ifo, path in sorted(files.items()):
		doc = ligolw.utils.load_filename(path)
		arrays = {array.Name: array.array for array in doc.getElementsByTagName(ligolw.Array.tagName)}
		snr_name = f"{ifo}_{bank_number}_{row_number}"
		if snr_name not in arrays:
			raise KeyError(f"{snr_name} not found in {path}")
		snr_arr = arrays[snr_name]
		times = np.asarray(snr_arr[0], dtype=float)
		if times.size and np.nanmax(np.abs(times)) < 1e6:
			times = times + filename_gps_start(path)
		ac_arr = arrays["autocorrelation_bank"]
		full[ifo] = (times, np.asarray(snr_arr[1] + 1j * snr_arr[2], dtype=np.complex128))
		vector = lal.CreateCOMPLEX16Vector(len(ac_arr))
		vector.data = np.asarray(ac_arr, dtype=np.complex128)
		aut[ifo] = vector
		doc.unlink()
	return full, aut


def build_common_window(full, aut, center_time, pre_trigger, start_idx_override=None):
	dt = float(full[sorted(full)[0]][0][1] - full[sorted(full)[0]][0][0])
	input_len = len(next(iter(aut.values())).data)
	ref_ifo = sorted(full)[0]
	t0 = float(full[ref_ifo][0][0])
	if start_idx_override is None:
		start_idx = int(round((center_time - pre_trigger - t0) / dt))
	else:
		start_idx = start_idx_override
	if start_idx < 0 or start_idx + input_len > len(full[ref_ifo][1]):
		raise ValueError(
			f"common window [{start_idx}, {start_idx + input_len}) outside SNR array length {len(full[ref_ifo][1])}"
		)

	snr = {}
	for ifo, (times, z) in full.items():
		data = np.asarray(z[start_idx:start_idx + input_len], dtype=np.complex128)
		series = lal.CreateCOMPLEX16TimeSeries(
			ifo,
			lal.LIGOTimeGPS(float(times[start_idx])),
			0.0,
			dt,
			lal.DimensionlessUnit,
			input_len,
		)
		series.data.data = data
		snr[ifo] = series
		peak_i = int(np.argmax(np.abs(data)))
		print(ifo, "peak_abs", float(abs(data[peak_i])), "peak_time", float(times[start_idx]) + peak_i * dt)
	return snr, dt


def coeff_series_to_prob(series, nside=None):
	alms, l_max, m_max = sph_healpix.sh_series_to_healpy_alm(series.get())
	if nside is None:
		_, _, log_prob = sph_healpix.healpy_alm_to_map(alms, l_max, m_max)
	else:
		log_prob = hp.alm2map(alms, nside, l_max, m_max)
	prob = np.exp(log_prob - log_prob.max())
	prob /= np.sum(prob)
	return prob


def coeff_pair_to_prob(series_p, series_n, nside=None):
	prob_p = coeff_series_to_prob(series_p, nside=nside)
	prob_n = coeff_series_to_prob(series_n, nside=nside)
	return (prob_p + prob_n) / 2.0


def write_prob_fits(prob, path):
	os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
	skymap_io.write_sky_map(path, prob)
	print("wrote", path, "npix", len(prob), "sum", float(np.sum(prob)))


def default_coeff_paths(output_fits):
	root, _ = os.path.splitext(output_fits)
	if root.endswith(".fits"):
		root = root[:-5]
	return root + "_coeffp.fits", root + "_coeffn.fits"


def main():
	parser = argparse.ArgumentParser()
	parser.add_argument("calc_snr_dir")
	parser.add_argument("--output-fits", required=True)
	parser.add_argument("--output-coeff-p")
	parser.add_argument("--output-coeff-n")
	parser.add_argument("--output-fits-p")
	parser.add_argument("--output-fits-n")
	parser.add_argument("--no-output-coefficients", action="store_true")
	parser.add_argument("--precalc-dir")
	parser.add_argument("--psd-xml")
	parser.add_argument("--mode", choices=("flat", "asd", "psd"), default="asd")
	parser.add_argument("--instruments", help="comma-separated detector list to use from the calc_snr dump")
	parser.add_argument("--bank-number", required=True)
	parser.add_argument("--row-number", required=True)
	parser.add_argument("--coinc-event-id", type=int)
	parser.add_argument("--center-time", type=float, required=True)
	parser.add_argument("--pre-trigger", type=float, default=0.12)
	parser.add_argument("--start-idx", type=int)
	parser.add_argument("--precalc-len", type=int, default=438)
	parser.add_argument("--sample-rate", type=float, default=2048.0)
	parser.add_argument("--effective-sample-rate", type=int, default=512)
	parser.add_argument("--scale-power", type=float, default=0.6)
	parser.add_argument("--scale-overall", type=float, default=2.7)
	parser.add_argument("--output-nside", type=int, help="HEALPix nside for the output map; default is derived from l_max")
	parser.add_argument("--include-whitening", action="store_true")
	parser.add_argument(
		"--no-derotate",
		action="store_true",
		help="deprecated no-op; Kipp's harmonic backend already returns celestial coordinates",
	)
	parser.add_argument("--timing", action="store_true")
	args = parser.parse_args()

	if not args.no_output_coefficients:
		default_coeff_p, default_coeff_n = default_coeff_paths(args.output_fits)
		args.output_coeff_p = args.output_coeff_p or default_coeff_p
		args.output_coeff_n = args.output_coeff_n or default_coeff_n

	timer = StageTimer(args.timing)
	with timer.stage("read_calc_snr"):
		full, aut = read_calc_snr(args.calc_snr_dir, args.bank_number, args.row_number)
	if args.instruments:
		selected = [ifo.strip() for ifo in args.instruments.split(",") if ifo.strip()]
		missing = sorted(set(selected) - set(full))
		if missing:
			raise ValueError(f"requested instruments not present in SNR dump: {', '.join(missing)}")
		full = {ifo: full[ifo] for ifo in selected}
		aut = {ifo: aut[ifo] for ifo in selected}
	with timer.stage("build_common_window"):
		snr, dt = build_common_window(full, aut, args.center_time, args.pre_trigger, args.start_idx)

	instruments = sorted(snr)
	if args.precalc_dir:
		loc = RapidLocalization.read(args.precalc_dir)
	else:
		if not args.psd_xml:
			raise ValueError("--psd-xml is required when --precalc-dir is not provided")
		psds = load_relative_psd_weights(args.psd_xml, instruments, args.precalc_len, args.sample_rate, args.mode)
		loc = RapidLocalization_(psds, args.precalc_len, dt, effective_sample_rate=args.effective_sample_rate)

	with timer.stage("harmonic_sphcoeff"):
		skyp, skyn = loc.sphcoeff(
			snr,
			aut,
			power=args.scale_power,
			overall=args.scale_overall,
		)
	sky = rapidskyloc_io(skyp, skyn, coinc_event_id=args.coinc_event_id)
	if args.output_nside:
		sky.prob = coeff_pair_to_prob(skyp, skyn, nside=args.output_nside)
	os.makedirs(os.path.dirname(os.path.abspath(args.output_fits)), exist_ok=True)
	if args.output_coeff_p:
		os.makedirs(os.path.dirname(os.path.abspath(args.output_coeff_p)), exist_ok=True)
		if sph.sh_series_write_healpix_alm(skyp.get(), args.output_coeff_p):
			raise RuntimeError(f"failed to write {args.output_coeff_p}")
		print("wrote", args.output_coeff_p)
	if args.output_coeff_n:
		os.makedirs(os.path.dirname(os.path.abspath(args.output_coeff_n)), exist_ok=True)
		if sph.sh_series_write_healpix_alm(skyn.get(), args.output_coeff_n):
			raise RuntimeError(f"failed to write {args.output_coeff_n}")
		print("wrote", args.output_coeff_n)
	with open(args.output_fits, "wb") as f:
		f.write(sky.to_fits_buffer(fmt="bayestar"))
	print("wrote", args.output_fits, "npix", len(sky.prob), "sum", float(np.sum(sky.prob)))

	if args.output_fits_p:
		write_prob_fits(coeff_series_to_prob(skyp, nside=args.output_nside), args.output_fits_p)
	if args.output_fits_n:
		write_prob_fits(coeff_series_to_prob(skyn, nside=args.output_nside), args.output_fits_n)
	timer.report()


if __name__ == "__main__":
	main()
