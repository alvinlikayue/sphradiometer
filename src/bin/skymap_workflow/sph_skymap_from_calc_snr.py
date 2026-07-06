#!/usr/bin/env python3
"""Create one skymap from gstlal_inspiral_calc_snr XML dumps."""

import argparse
import glob
import json
import os
import re
import subprocess
import time
from contextlib import contextmanager
from numpy.polynomial import legendre

import lal
import ligolw
import ligolw.utils
import healpy as hp
import numpy as np
from astropy.io import fits
from astropy_healpix import level_ipix_to_uniq, nside_to_level
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
		total = time.perf_counter() - self.start
		print("timing total %.6f s" % total)
		for label, elapsed in self.records:
			print("timing %-28s %.6f s" % (label, elapsed))


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


def build_common_window(full, aut, center_time, pre_trigger, start_idx_override=None, time_gate_center=None, time_gate_half_width=None):
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
		if time_gate_center is not None and time_gate_half_width is not None:
			window_times = np.asarray(times[start_idx:start_idx + input_len], dtype=float)
			data = data.copy()
			data[np.abs(window_times - time_gate_center) > time_gate_half_width] = 0.0
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


def coeff_series_to_prob(series):
	_, _, log_prob = sph_healpix.healpy_alm_to_map(*sph_healpix.sh_series_to_healpy_alm(series.get()))
	prob = np.exp(log_prob - log_prob.max())
	prob /= np.sum(prob)
	return prob


def derotate_to_celestial(series, gmst):
	source = series.get()
	rotated = sph.sh_series_new_zero(source.l_max, source.polar)
	sph.sh_series_rotate_z(rotated, source, -gmst)
	if series.ndet > 1:
		sph.sh_seriespp_assign(series.coeff, rotated)
	else:
		sph.sh_seriesp_assign(series.coeff, rotated)


def write_prob_fits(prob, path):
	os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
	skymap_io.write_sky_map(path, prob)
	print("wrote", path, "npix", len(prob), "sum", float(np.sum(prob)))


def apply_isotropic_mixture(prob, fraction):
	fraction = float(fraction)
	if fraction < 0.0 or fraction >= 1.0:
		raise ValueError("--isotropic-mixture must be in [0, 1)")
	if fraction == 0.0:
		return prob
	prob = np.asarray(prob, dtype=float)
	npix = prob.size
	mixed = (1.0 - fraction) * prob + fraction / npix
	mixed /= np.sum(mixed)
	return mixed


def apply_probability_mixture(prob, prior, fraction, name):
	fraction = float(fraction)
	if fraction < 0.0 or fraction >= 1.0:
		raise ValueError("--%s must be in [0, 1)" % name)
	if fraction == 0.0:
		return prob
	prob = np.asarray(prob, dtype=float)
	prior = np.asarray(prior, dtype=float)
	if prob.shape != prior.shape:
		raise ValueError("%s prior shape does not match probability map" % name)
	prior = np.maximum(prior, 0.0)
	prior /= np.sum(prior)
	mixed = (1.0 - fraction) * prob + fraction * prior
	mixed /= np.sum(mixed)
	return mixed


def write_moc_fits(nside, ipix, prob, path):
	os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
	nside = np.asarray(nside, dtype=np.int64)
	ipix = np.asarray(ipix, dtype=np.int64)
	prob = np.asarray(prob, dtype=float)
	if not (len(nside) == len(ipix) == len(prob)):
		raise ValueError("adaptive MOC arrays have inconsistent lengths")
	if np.any(prob < 0.0) or not np.all(np.isfinite(prob)):
		raise ValueError("adaptive MOC probabilities must be finite and non-negative")
	prob = prob / np.sum(prob)
	area = 4.0 * np.pi / (12.0 * nside.astype(float) ** 2)
	uniq = level_ipix_to_uniq(nside_to_level(nside), ipix)
	order = np.argsort(uniq)
	table = fits.BinTableHDU.from_columns([
		fits.Column(name="UNIQ", format="K", array=uniq[order]),
		fits.Column(name="PROBDENSITY", format="D", unit="sr-1", array=(prob / area)[order]),
	])
	table.header["PIXTYPE"] = "HEALPIX"
	table.header["ORDERING"] = "NUNIQ"
	table.header["COORDSYS"] = "C"
	table.header["INDXSCHM"] = "EXPLICIT"
	table.header["MOCORDER"] = int(np.max(nside_to_level(nside)))
	table.header["OBJECT"] = "coherent_likelihood"
	table.writeto(path, overwrite=True)
	print("wrote", path, "moc_pixels", len(prob), "prob_sum", float(np.sum(prob)), "nside_min", int(np.min(nside)), "nside_max", int(np.max(nside)))


def default_coeff_paths(output_fits):
	root, _ = os.path.splitext(output_fits)
	if root.endswith(".fits"):
		root = root[:-5]
	return root + "_coeffp.fits", root + "_coeffn.fits"


def apply_containment_calibration(args):
	if not args.containment_calibration:
		return
	if args.network_snr is None:
		raise ValueError("--network-snr is required with --containment-calibration")
	if args.high_snr_threshold <= args.low_snr_threshold:
		raise ValueError("--high-snr-threshold must be greater than --low-snr-threshold")
	if args.network_snr <= args.low_snr_threshold:
		weight = 0.0
		regime = "low_snr"
	elif args.network_snr >= args.high_snr_threshold:
		weight = 1.0
		regime = "high_snr"
	else:
		x = (args.network_snr - args.low_snr_threshold) / (args.high_snr_threshold - args.low_snr_threshold)
		weight = x * x * (3.0 - 2.0 * x)
		regime = "intermediate_snr"
	args.direct_score_scale = (1.0 - weight) * args.low_snr_score_scale + weight * args.high_snr_score_scale
	args.direct_smoothing_deg = (1.0 - weight) * args.low_snr_smoothing_deg + weight * args.high_snr_smoothing_deg
	args.isotropic_mixture = (1.0 - weight) * args.low_snr_isotropic_mixture + weight * args.high_snr_isotropic_mixture
	args.antenna_mixture = (1.0 - weight) * args.low_snr_antenna_mixture + weight * args.high_snr_antenna_mixture
	print(
		"containment_calibration",
		"regime", regime,
		"network_snr", args.network_snr,
		"weight", weight,
		"score_scale", args.direct_score_scale,
		"smoothing_deg", args.direct_smoothing_deg,
		"isotropic_mixture", args.isotropic_mixture,
		"antenna_mixture", args.antenna_mixture,
		flush=True,
	)


def resolve_direct_score_scale(args, length):
	try:
		args.direct_score_scale = float(args.direct_score_scale)
	except ValueError:
		mode = str(args.direct_score_scale).strip().lower()
		if mode not in ("snr_baseline_ramp", "snr-baseline-ramp"):
			raise ValueError("--direct-score-scale must be a number, <=0 for auto, or 'snr_baseline_ramp'")
		if args.network_snr is None:
			raise ValueError("--network-snr is required with --direct-score-scale snr_baseline_ramp")
		base = 1.0 / float(length)
		n_baselines = max(1, len(args.active_instruments) * (len(args.active_instruments) - 1) // 2)
		if n_baselines == 1:
			cap = args.snr_ramp_two_ifo_cap
		else:
			cap = args.snr_ramp_multi_ifo_cap
		if args.snr_ramp_high_snr <= args.snr_ramp_low_snr:
			raise ValueError("--snr-ramp-high-snr must be greater than --snr-ramp-low-snr")
		if args.network_snr <= args.snr_ramp_low_snr:
			weight = 0.0
		elif args.network_snr >= args.snr_ramp_high_snr:
			weight = 1.0
		else:
			x = (args.network_snr - args.snr_ramp_low_snr) / (args.snr_ramp_high_snr - args.snr_ramp_low_snr)
			weight = x * x * (3.0 - 2.0 * x)
		args.direct_score_scale = base + weight * (cap - base)
		print(
			"direct_score_scale snr_baseline_ramp",
			"length", length,
			"network_snr", args.network_snr,
			"n_baselines", n_baselines,
			"base", base,
			"cap", cap,
			"weight", weight,
			"score_scale", args.direct_score_scale,
			flush=True,
		)
		return
	if args.direct_score_scale > 0.0:
		return
	args.direct_score_scale = 1.0 / float(length)
	print(
		"direct_score_scale auto",
		"length", length,
		"score_scale", args.direct_score_scale,
		flush=True,
	)


def write_prob_alm(prob, path, lmax=None):
	os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
	nside = hp.npix2nside(len(prob))
	if lmax is None:
		lmax = 3 * nside - 1
	alm = hp.map2alm(prob, lmax=lmax)
	hp.fitsfunc.write_alm(path, alm, overwrite=True)
	print("wrote", path, "lmax", lmax, "nalm", len(alm))


def split_combo(combo):
	return [combo[i:i + 2] for i in range(0, len(combo), 2)]


def precalc_metadata_path(precalc_dir, instruments):
	combo = "".join(sorted(instruments))
	return os.path.abspath(precalc_dir).rstrip(os.sep) + ".metadata/" + combo + ".json"


def read_precalc_metadata(precalc_dir, instruments):
	path = precalc_metadata_path(precalc_dir, instruments)
	if not os.path.exists(path):
		return {}
	with open(path) as f:
		metadata = json.load(f)
	metadata["_metadata_path"] = path
	return metadata


def write_logprob_mesh_sh_series(mesh_values, path, lmax, shift=None, log_floor=None):
	os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
	ntheta = nphi = 2 * (lmax + 1)
	if mesh_values.shape != (ntheta, nphi):
		raise ValueError(f"log-likelihood mesh has shape {mesh_values.shape}, expected {(ntheta, nphi)}")
	if shift is None:
		shift = float(np.nanmax(mesh_values))
	mesh_values = np.asarray(mesh_values - shift, dtype=float)
	if log_floor is not None:
		if log_floor <= 0.0:
			raise ValueError("log_floor must be positive")
		mesh_values = np.maximum(mesh_values, -float(log_floor))
	if not np.all(np.isfinite(mesh_values)):
		raise ValueError("non-finite values in spherical-harmonic log-likelihood mesh")
	mesh = sph.new_double_array(ntheta * nphi)
	series = sph.sh_series_new_zero(lmax, 0)
	try:
		flat = np.ravel(mesh_values)
		for i, value in enumerate(flat):
			sph.double_array_setitem(mesh, i, float(value))
		if not sph.sh_series_from_realmesh(series, mesh):
			raise RuntimeError("sh_series_from_realmesh() failed")
		if sph.sh_series_write_healpix_alm(series, path):
			raise RuntimeError(f"failed to write {path}")
	finally:
		sph.delete_double_array(mesh)
		sph.sh_series_free(series)
	print("wrote", path, "lmax", lmax, "mesh", f"{ntheta}x{nphi}", "log_shift", shift, "log_floor", log_floor)
	return shift


def write_logprob_sh_series(log_prob_func, path, lmax):
	ntheta = nphi = 2 * (lmax + 1)
	cos_theta, _ = legendre.leggauss(ntheta)
	cos_theta = cos_theta[::-1]
	theta = np.arccos(cos_theta)
	phi = np.arange(nphi, dtype=float) * (2.0 * np.pi / nphi)
	return write_logprob_mesh_sh_series(log_prob_func(theta, phi), path, lmax)


def write_skymap_png(fits_path, png_path, ra_deg=None, dec_deg=None, contour=None):
	cmd = [
		"ligo-skymap-plot",
		fits_path,
		"-o",
		png_path,
		"--annotate",
		"--contour",
		] + [str(value) for value in (contour or ["50", "90"])]
	if ra_deg is not None and dec_deg is not None:
		cmd += ["--radec", str(ra_deg), str(dec_deg)]
	subprocess.check_call(cmd)
	print("wrote", png_path)


def detector_by_prefix(instruments):
	detectors = {}
	for detector in lal.CachedDetectors:
		prefix = detector.frDetector.prefix
		if prefix in instruments:
			detectors[prefix] = detector
	missing = set(instruments) - set(detectors)
	if missing:
		raise ValueError(f"missing LAL detector definitions for: {', '.join(sorted(missing))}")
	return detectors


def sample_complex_timeseries(times, values, sample_times):
	real = np.interp(sample_times, times, values.real, left=0.0, right=0.0)
	imag = np.interp(sample_times, times, values.imag, left=0.0, right=0.0)
	return real + 1j * imag


def direct_detector_weights(args, instruments, full):
	if not args.psd_xml:
		return {ifo: 1.0 for ifo in instruments}
	dt = float(full[instruments[0]][0][1] - full[instruments[0]][0][0])
	length = args.precalc_len
	psds = load_relative_psd_weights(args.psd_xml, instruments, length, 1.0 / dt, args.mode)
	weights = {}
	for ifo in instruments:
		values = np.array([sph.double_array_getitem(psds[ifo].psd, i) for i in range(length)], dtype=float)
		good = np.isfinite(values) & (values > 0.0)
		weights[ifo] = float(np.median(values[good])) if np.any(good) else 1.0
	return weights


def detector_response_and_delay(detector, ra, dec, gmst):
	"""Vectorized equivalent of LAL's delay and psi=0 antenna response."""
	ra = np.asarray(ra, dtype=float)
	dec = np.asarray(dec, dtype=float)
	alpha = ra - gmst
	cos_dec = np.cos(dec)
	sin_dec = np.sin(dec)
	cos_alpha = np.cos(alpha)
	sin_alpha = np.sin(alpha)

	source = np.stack((
		cos_dec * cos_alpha,
		cos_dec * sin_alpha,
		sin_dec,
	), axis=0)
	delay = -np.tensordot(np.asarray(detector.location, dtype=float), source, axes=(0, 0)) / lal.C_SI

	p = np.stack((sin_alpha, -cos_alpha, np.zeros_like(alpha)), axis=0)
	q = np.stack((
		-sin_dec * cos_alpha,
		-sin_dec * sin_alpha,
		cos_dec,
	), axis=0)
	response = np.asarray(detector.response, dtype=float)
	fplus = np.einsum("ij,i...,j...->...", response, p, p) - np.einsum("ij,i...,j...->...", response, q, q)
	fcross = np.einsum("ij,i...,j...->...", response, p, q) + np.einsum("ij,i...,j...->...", response, q, p)
	return delay, fplus, fcross


def snr_peak_info(full):
	info = {}
	for ifo, (times, values) in full.items():
		abs_values = np.asarray(np.abs(values), dtype=float)
		index = int(np.argmax(abs_values))
		peak_abs = float(abs_values[index])
		floor = float(np.median(abs_values))
		scale = float(np.percentile(abs_values, 95.0) - floor)
		prominence = (peak_abs - floor) / max(scale, 1e-12)
		info[ifo] = {
			"time": float(times[index]),
			"abs": peak_abs,
			"floor": floor,
			"prominence": prominence,
		}
	return info


def peak_time_quality_passes(context, instruments, args):
	if args.peak_time_min_peak_abs <= 0.0 and args.peak_time_min_prominence <= 0.0:
		return True
	for ifo in instruments:
		info = context["peak_info"][ifo]
		if args.peak_time_min_peak_abs > 0.0 and info["abs"] < args.peak_time_min_peak_abs:
			return False
		if args.peak_time_min_prominence > 0.0 and info["prominence"] < args.peak_time_min_prominence:
			return False
	return True


def peak_time_pair_sigma(context, args, ifo_i, ifo_j):
	sigma = float(args.peak_time_sigma)
	if args.peak_time_uncertainty_model == "fixed":
		return sigma
	info_i = context["peak_info"][ifo_i]
	info_j = context["peak_info"][ifo_j]
	min_peak_abs = min(float(info_i["abs"]), float(info_j["abs"]))
	if min_peak_abs <= 0.0:
		return float(args.peak_time_max_sigma) if args.peak_time_max_sigma > 0.0 else sigma
	scale = max(1.0, (float(args.peak_time_reference_peak_abs) / min_peak_abs) ** float(args.peak_time_uncertainty_power))
	pair_sigma = sigma * scale
	if args.peak_time_max_sigma > 0.0:
		pair_sigma = min(pair_sigma, float(args.peak_time_max_sigma))
	return pair_sigma


def peak_time_consistency_logprob(context, instruments, args, ra, dec):
	if args.peak_time_sigma <= 0.0 or len(instruments) < 2:
		return 0.0
	if args.peak_time_min_snr > 0.0 and args.network_snr is not None and args.network_snr < args.peak_time_min_snr:
		return 0.0
	if not peak_time_quality_passes(context, instruments, args):
		return 0.0
	ra = np.asarray(ra, dtype=float)
	dec = np.asarray(dec, dtype=float)
	shape = np.broadcast_shapes(ra.shape, dec.shape)
	ra = np.broadcast_to(ra, shape)
	dec = np.broadcast_to(dec, shape)
	flat_ra = np.ravel(ra)
	flat_dec = np.ravel(dec)
	gps = lal.LIGOTimeGPS(float(args.time_prior_center or args.center_time))
	gmst = lal.GreenwichMeanSiderealTime(gps)
	delays = {}
	for ifo in instruments:
		delays[ifo], _, _ = detector_response_and_delay(context["detectors"][ifo], flat_ra, flat_dec, gmst)
	log_prob = np.zeros(flat_ra.size, dtype=float)
	npairs = 0
	for i, ifo_i in enumerate(instruments):
		for ifo_j in instruments[:i]:
			observed = context["peak_info"][ifo_i]["time"] - context["peak_info"][ifo_j]["time"]
			predicted = delays[ifo_i] - delays[ifo_j]
			residual = observed - predicted
			pair_sigma = peak_time_pair_sigma(context, args, ifo_i, ifo_j)
			log_prob -= 0.5 * (residual / pair_sigma) ** 2
			npairs += 1
	if npairs:
		log_prob /= npairs
	return log_prob.reshape(shape)


def apply_two_ifo_timing_defaults(args, instruments):
	if len(instruments) != 2 or not args.two_ifo_timing_prior or args.peak_time_sigma > 0.0:
		return
	args.peak_time_sigma = args.two_ifo_peak_time_sigma
	args.peak_time_min_peak_abs = args.two_ifo_peak_time_min_peak_abs
	args.peak_time_uncertainty_model = "peak_abs"
	args.peak_time_reference_peak_abs = args.two_ifo_peak_time_reference_peak_abs
	args.peak_time_uncertainty_power = args.two_ifo_peak_time_uncertainty_power
	args.peak_time_max_sigma = args.two_ifo_peak_time_max_sigma
	print(
		"two_ifo_timing_prior",
		"enabled", True,
		"sigma", args.peak_time_sigma,
		"min_peak_abs", args.peak_time_min_peak_abs,
		"reference_peak_abs", args.peak_time_reference_peak_abs,
		"uncertainty_power", args.peak_time_uncertainty_power,
		"max_sigma", args.peak_time_max_sigma,
	)


def single_detector_antenna_logprob(instrument, args, ra, dec):
	detector = detector_by_prefix([instrument])[instrument]
	gps = lal.LIGOTimeGPS(float(args.time_prior_center or args.center_time))
	gmst = lal.GreenwichMeanSiderealTime(gps)
	_, fplus, fcross = detector_response_and_delay(detector, ra, dec, gmst)
	power = np.asarray(fplus * fplus + fcross * fcross, dtype=float)
	floor = max(float(np.nanmax(power)) * 1e-12, 1e-300)
	return np.log(np.maximum(power, floor))


def network_antenna_probability(context, instruments, args, ra, dec):
	gps = lal.LIGOTimeGPS(float(args.time_prior_center or args.center_time))
	gmst = lal.GreenwichMeanSiderealTime(gps)
	shape = np.broadcast_shapes(np.shape(ra), np.shape(dec))
	ra = np.broadcast_to(np.asarray(ra, dtype=float), shape)
	dec = np.broadcast_to(np.asarray(dec, dtype=float), shape)
	power = np.zeros(shape, dtype=float)
	for ifo in instruments:
		_, fplus, fcross = detector_response_and_delay(context["detectors"][ifo], ra, dec, gmst)
		power += np.asarray(fplus * fplus + fcross * fcross, dtype=float)
	power = np.maximum(power, 0.0)
	if not np.any(power):
		power = np.ones_like(power)
	prob = power / np.sum(power)
	return prob


def single_detector_antenna_map(instrument, args, timer=None):
	if args.direct_nside <= 0:
		raise ValueError("--direct-nside must be positive")
	timer = timer or StageTimer(False)
	with timer.stage("antenna_map_grid"):
		npix = hp.nside2npix(args.direct_nside)
		theta, phi = hp.pix2ang(args.direct_nside, np.arange(npix))
	with timer.stage("antenna_map_likelihood"):
		log_prob = single_detector_antenna_logprob(instrument, args, phi, np.pi / 2.0 - theta)
	with timer.stage("antenna_map_normalize"):
		prob = np.exp(log_prob - np.max(log_prob))
		prob /= np.sum(prob)
	print("single_detector_antenna", "ifo", instrument, "nside", args.direct_nside)
	return prob, prob.copy(), prob.copy()


def write_single_detector_antenna_sh_series(instrument, args, output_coeff_p, output_coeff_n, timer=None):
	timer = timer or StageTimer(False)
	lmax = args.direct_coeff_lmax
	if lmax is None:
		lmax = 3 * args.direct_nside - 1
	if lmax < 0:
		raise ValueError("--direct-coeff-lmax must be non-negative")
	with timer.stage("antenna_coeff_grid"):
		ntheta = nphi = 2 * (lmax + 1)
		cos_theta, _ = legendre.leggauss(ntheta)
		cos_theta = cos_theta[::-1]
		theta = np.arccos(cos_theta)
		phi = np.arange(nphi, dtype=float) * (2.0 * np.pi / nphi)
		theta_grid, phi_grid = np.meshgrid(theta, phi, indexing="ij")
	with timer.stage("antenna_coeff_likelihood"):
		log_prob = single_detector_antenna_logprob(instrument, args, phi_grid, np.pi / 2.0 - theta_grid)
	shift = None
	if output_coeff_p and output_coeff_n:
		shift = float(np.nanmax(log_prob))
	if output_coeff_p:
		with timer.stage("write_coeff_p"):
			shift = write_logprob_mesh_sh_series(log_prob, output_coeff_p, lmax, shift=shift, log_floor=args.direct_coeff_log_floor)
	if output_coeff_n:
		with timer.stage("write_coeff_n"):
			write_logprob_mesh_sh_series(log_prob, output_coeff_n, lmax, shift=shift, log_floor=args.direct_coeff_log_floor)


def direct_time_prior_map(full, instruments, args):
	if args.time_prior_center is None:
		raise ValueError("--time-prior-center is required with --direct-time-prior")
	if args.time_prior_half_width is None:
		raise ValueError("--time-prior-half-width is required with --direct-time-prior")
	if args.direct_nside <= 0:
		raise ValueError("--direct-nside must be positive")
	detectors = detector_by_prefix(instruments)
	weights = direct_detector_weights(args, instruments, full)
	ref_dt = float(full[instruments[0]][0][1] - full[instruments[0]][0][0])
	resolve_direct_score_scale(args, len(full[instruments[0]][1]))
	time_step = args.time_prior_step or ref_dt
	times_geo = np.arange(
		args.time_prior_center - args.time_prior_half_width,
		args.time_prior_center + args.time_prior_half_width + 0.5 * time_step,
		time_step,
	)
	if times_geo.size == 0:
		raise ValueError("empty geocentric time prior grid")

	npix = hp.nside2npix(args.direct_nside)
	log_prob_p = np.empty(npix)
	log_prob_n = np.empty(npix)
	for pix in range(npix):
		theta, phi = hp.pix2ang(args.direct_nside, pix)
		ra = phi
		dec = np.pi / 2.0 - theta
		score = {+1: np.empty(times_geo.size), -1: np.empty(times_geo.size)}
		for it, tg in enumerate(times_geo):
			gps = lal.LIGOTimeGPS(float(tg))
			gmst = lal.GreenwichMeanSiderealTime(gps)
			z = []
			fplus = []
			fcross = []
			for ifo in instruments:
				delay = lal.TimeDelayFromEarthCenter(detectors[ifo].location, ra, dec, gps)
				sample = sample_complex_timeseries(full[ifo][0], full[ifo][1], np.array([tg + delay]))[0]
				fp, fc = lal.ComputeDetAMResponse(detectors[ifo].response, ra, dec, 0.0, gmst)
				weight = weights[ifo]
				z.append(sample)
				fplus.append(fp / weight)
				fcross.append(fc / weight)
			z = np.asarray(z, dtype=np.complex128)
			fplus = np.asarray(fplus, dtype=float)
			fcross = np.asarray(fcross, dtype=float)
			for beta in (+1, -1):
				response = fplus + 1j * beta * fcross
				norm = float(np.vdot(response, response).real)
				score[beta][it] = 0.0 if norm <= 0.0 else abs(np.vdot(response / np.sqrt(norm), z)) ** 2
		log_prob_p[pix] = np.logaddexp.reduce(args.direct_score_scale * score[+1]) - np.log(times_geo.size)
		log_prob_n[pix] = np.logaddexp.reduce(args.direct_score_scale * score[-1]) - np.log(times_geo.size)

	if args.likelihood_branch == "p":
		log_prob = log_prob_p
	elif args.likelihood_branch == "n":
		log_prob = log_prob_n
	else:
		log_prob = np.logaddexp(log_prob_p, log_prob_n) - np.log(2.0)
	prob = np.exp(log_prob - np.max(log_prob))
	prob /= np.sum(prob)
	prob_p = np.exp(log_prob_p - np.max(log_prob_p))
	prob_p /= np.sum(prob_p)
	prob_n = np.exp(log_prob_n - np.max(log_prob_n))
	prob_n /= np.sum(prob_n)
	if args.direct_smoothing_deg > 0.0:
		prob = hp.smoothing(prob, sigma=np.deg2rad(args.direct_smoothing_deg))
		prob = np.maximum(prob, 0.0)
		prob /= np.sum(prob)
		prob_p = hp.smoothing(prob_p, sigma=np.deg2rad(args.direct_smoothing_deg))
		prob_p = np.maximum(prob_p, 0.0)
		prob_p /= np.sum(prob_p)
		prob_n = hp.smoothing(prob_n, sigma=np.deg2rad(args.direct_smoothing_deg))
		prob_n = np.maximum(prob_n, 0.0)
		prob_n /= np.sum(prob_n)
	print(
		"direct_time_prior",
		"nside", args.direct_nside,
		"ntimes", len(times_geo),
		"t0", float(times_geo[0]),
		"t1", float(times_geo[-1]),
		"score_scale", args.direct_score_scale,
		"smoothing_deg", args.direct_smoothing_deg,
		"weights", ",".join(f"{ifo}:{weights[ifo]:.6g}" for ifo in instruments),
	)
	return prob, prob_p, prob_n


def _regularized_autocorr_spectrum(aut_values, length, floor_fraction, shift_zero_lag=False):
	values = np.asarray(aut_values, dtype=np.complex128)
	if shift_zero_lag:
		values = np.fft.ifftshift(values)
	noise = np.abs(np.fft.fft(values, n=length))
	good = np.isfinite(noise) & (noise > 0.0)
	if not np.any(good):
		raise ValueError("autocorrelation spectrum has no positive finite samples")
	floor = floor_fraction * float(np.median(noise[good]))
	noise[~good] = floor
	noise = np.maximum(noise, floor)
	return noise


def _autocorr_template_spectrum(aut_values, length):
	values = np.asarray(aut_values, dtype=np.complex128)
	return np.fft.fft(np.fft.ifftshift(values), n=length)


def centered_pad(values, length):
	values = np.asarray(values, dtype=np.complex128)
	if len(values) == length:
		return values.copy()
	if len(values) > length:
		start = (len(values) - length) // 2
		return values[start:start + length].copy()
	padded = np.zeros(length, dtype=np.complex128)
	start = (length - len(values)) // 2
	padded[start:start + len(values)] = values
	return padded


def tukey_window(length, alpha):
	if alpha < 0.0 or alpha > 1.0:
		raise ValueError("--paper-tukey-alpha must be in [0, 1]")
	if length < 1:
		raise ValueError("window length must be positive")
	if alpha == 0.0:
		return np.ones(length, dtype=float)
	if alpha == 1.0:
		return np.hanning(length)
	n = np.arange(length, dtype=float)
	window = np.ones(length, dtype=float)
	edge = alpha * (length - 1) / 2.0
	first = n < edge
	last = n >= (length - 1) * (1.0 - alpha / 2.0)
	window[first] = 0.5 * (1.0 + np.cos(np.pi * (2.0 * n[first] / (alpha * (length - 1)) - 1.0)))
	window[last] = 0.5 * (1.0 + np.cos(np.pi * (2.0 * n[last] / (alpha * (length - 1)) - 2.0 / alpha + 1.0)))
	return window


def prepare_coherent_likelihood_context(full, aut, instruments, args):
	if args.time_prior_center is None:
		raise ValueError("--time-prior-center is required with --coherent-likelihood")
	if args.time_prior_half_width is None:
		raise ValueError("--time-prior-half-width is required with --coherent-likelihood")
	if args.paper_autocorr_floor <= 0.0:
		raise ValueError("--paper-autocorr-floor must be positive")

	detectors = detector_by_prefix(instruments)
	ref_ifo = instruments[0]
	dt = float(full[ref_ifo][0][1] - full[ref_ifo][0][0])
	if args.paper_snr_window == "full":
		length = min(len(values) for _, values in full.values())
		start_idx = 0
	else:
		length = len(next(iter(aut.values())).data)
		start_idx = int(round((args.center_time - args.pre_trigger - float(full[ref_ifo][0][0])) / dt))
	if start_idx < 0:
		raise ValueError("coherent likelihood SNR window starts before available data")
	freqs = np.fft.fftfreq(length, d=dt)
	freq_mask = np.ones(length, dtype=bool)
	if args.paper_fmax > 0.0:
		freq_mask &= np.abs(freqs) <= args.paper_fmax
	if args.paper_fmin > 0.0:
		freq_mask &= np.abs(freqs) >= args.paper_fmin
	if not np.any(freq_mask):
		raise ValueError("coherent likelihood frequency mask is empty")
	freqs = freqs[freq_mask]
	time_step = args.time_prior_step or dt
	times_geo = np.arange(
		args.time_prior_center - args.time_prior_half_width,
		args.time_prior_center + args.time_prior_half_width + 0.5 * time_step,
		time_step,
	)
	if times_geo.size == 0:
		raise ValueError("empty geocentric time prior grid")

	fft_snr = {}
	inv_noise = {}
	aut_template = {}
	epochs = {}
	taper = tukey_window(length, args.paper_tukey_alpha)
	for ifo in instruments:
		times, values = full[ifo]
		if start_idx + length > len(values):
			raise ValueError(f"{ifo} coherent likelihood window exceeds SNR array")
		window_values = np.asarray(values[start_idx:start_idx + length], dtype=np.complex128) * taper
		aut_values = centered_pad(aut[ifo].data, length) * taper
		fft_snr[ifo] = np.fft.fft(window_values)[freq_mask]
		if args.paper_flat_noise:
			inv_noise[ifo] = np.ones(np.count_nonzero(freq_mask), dtype=float)
		else:
			inv_noise[ifo] = 1.0 / _regularized_autocorr_spectrum(aut_values, length, args.paper_autocorr_floor, args.paper_shift_autocorr)[freq_mask]
		aut_template[ifo] = _autocorr_template_spectrum(aut_values, length)[freq_mask]
		epochs[ifo] = float(times[start_idx])

	context = {
		"detectors": detectors,
		"dt": dt,
		"length": length,
		"freq_mask_count": int(np.count_nonzero(freq_mask)),
		"freqs": freqs,
		"times_geo": times_geo,
		"fft_snr": fft_snr,
		"inv_noise": inv_noise,
		"aut_template": aut_template,
		"epochs": epochs,
		"peak_info": snr_peak_info({ifo: full[ifo] for ifo in instruments}),
	}
	if hasattr(sph, "direct_paper_logprob_for_radec_c"):
		data = []
		den_weight = []
		for ifo in instruments:
			if args.amplitude_model == "paper_cross" and args.paper_cross_weighting == "raw":
				template = 1.0
				weight = 1.0
			elif args.paper_autocorr_template or (args.amplitude_model == "paper_cross" and args.paper_cross_weighting == "autocorr_template"):
				template = aut_template[ifo]
				weight = inv_noise[ifo]
			else:
				template = 1.0
				weight = inv_noise[ifo]
			data.append(fft_snr[ifo] * np.conj(template) * weight)
			den_weight.append(float(np.sum(np.abs(template) ** 2 * weight)))
		context["c_inputs"] = {
			"locations": np.ascontiguousarray([detectors[ifo].location for ifo in instruments], dtype=np.float64),
			"responses": np.ascontiguousarray([detectors[ifo].response for ifo in instruments], dtype=np.float64),
			"freqs": np.ascontiguousarray(freqs, dtype=np.float64),
			"times_geo": np.ascontiguousarray(times_geo, dtype=np.float64),
			"epochs": np.ascontiguousarray([epochs[ifo] for ifo in instruments], dtype=np.float64),
			"data": np.ascontiguousarray(data, dtype=np.complex128),
			"den_weight": np.ascontiguousarray(den_weight, dtype=np.float64),
		}
	return context


def inclination_grid(samples):
	if samples < 1 or samples > 33:
		raise ValueError("--inclination-samples must be in [1, 33]")
	if samples == 1:
		return np.array([1.0], dtype=float)
	return np.linspace(-1.0, 1.0, samples, dtype=float)


def polarization_grid(samples):
	if samples < 1 or samples > 16:
		raise ValueError("--polarization-samples must be in [1, 16]")
	return np.linspace(0.0, np.pi, samples, endpoint=False, dtype=float)


def coherent_likelihood_logprob_at_radec(context, instruments, args, ra, dec):
	if args.amplitude_model == "cbc":
		cosi_grid = inclination_grid(args.inclination_samples)
		psi_grid = polarization_grid(args.polarization_samples)
		score = np.empty((context["times_geo"].size, cosi_grid.size, psi_grid.size), dtype=float)
		for it, tg in enumerate(context["times_geo"]):
			gps = lal.LIGOTimeGPS(float(tg))
			gmst = lal.GreenwichMeanSiderealTime(gps)
			shifted = {}
			fplus = {}
			fcross = {}
			for ifo in instruments:
				delay = lal.TimeDelayFromEarthCenter(context["detectors"][ifo].location, ra, dec, gps)
				phase = np.exp(2j * np.pi * args.paper_phase_sign * context["freqs"] * (tg + delay - context["epochs"][ifo]))
				shifted[ifo] = context["fft_snr"][ifo] * phase
				fp, fc = lal.ComputeDetAMResponse(context["detectors"][ifo].response, ra, dec, 0.0, gmst)
				fplus[ifo] = fp
				fcross[ifo] = fc
			for inc, cosi in enumerate(cosi_grid):
				num = 0.0j
				den = 0.0
				aplus = 0.5 * (1.0 + cosi * cosi)
				across = cosi
				for pol, psi in enumerate(psi_grid):
					num = 0.0j
					den = 0.0
					c2p = np.cos(2.0 * psi)
					s2p = np.sin(2.0 * psi)
					for ifo in instruments:
						fplus_psi = fplus[ifo] * c2p + fcross[ifo] * s2p
						fcross_psi = -fplus[ifo] * s2p + fcross[ifo] * c2p
						response = aplus * fplus_psi + 1j * across * fcross_psi
						weight = context["inv_noise"][ifo]
						if args.paper_autocorr_template:
							template = context["aut_template"][ifo]
						else:
							template = 1.0
						response_factor = np.conj(response) if args.paper_conj_response else response
						num += np.sum(response_factor * np.conj(template) * shifted[ifo] * weight)
						den += float(np.sum(abs(response) ** 2 * np.abs(template) ** 2 * weight))
					score[it, inc, pol] = 0.0 if den <= 0.0 else abs(num) ** 2 / den
		log_prob = np.logaddexp.reduce(args.direct_score_scale * score.ravel()) - np.log(score.size)
		return log_prob, log_prob
	score = {+1: np.empty(context["times_geo"].size), -1: np.empty(context["times_geo"].size)}
	for it, tg in enumerate(context["times_geo"]):
		gps = lal.LIGOTimeGPS(float(tg))
		gmst = lal.GreenwichMeanSiderealTime(gps)
		shifted = {}
		fplus = {}
		fcross = {}
		for ifo in instruments:
			delay = lal.TimeDelayFromEarthCenter(context["detectors"][ifo].location, ra, dec, gps)
			phase = np.exp(2j * np.pi * args.paper_phase_sign * context["freqs"] * (tg + delay - context["epochs"][ifo]))
			shifted[ifo] = context["fft_snr"][ifo] * phase
			fp, fc = lal.ComputeDetAMResponse(context["detectors"][ifo].response, ra, dec, 0.0, gmst)
			fplus[ifo] = fp
			fcross[ifo] = fc
		for beta in (+1, -1):
			num = 0.0j
			den = 0.0
			for ifo in instruments:
				response = fplus[ifo] + 1j * beta * fcross[ifo]
				weight = context["inv_noise"][ifo]
				if args.paper_autocorr_template:
					template = context["aut_template"][ifo]
				else:
					template = 1.0
				response_factor = np.conj(response) if args.paper_conj_response else response
				num += np.sum(response_factor * np.conj(template) * shifted[ifo] * weight)
				den += float(np.sum(abs(response) ** 2 * np.abs(template) ** 2 * weight))
			score[beta][it] = 0.0 if den <= 0.0 else abs(num) ** 2 / den
	log_prob_p = np.logaddexp.reduce(args.direct_score_scale * score[+1]) - np.log(context["times_geo"].size)
	log_prob_n = np.logaddexp.reduce(args.direct_score_scale * score[-1]) - np.log(context["times_geo"].size)
	return log_prob_p, log_prob_n


def coherent_likelihood_logprob_for_radec(context, instruments, args, ra, dec):
	ra = np.asarray(ra, dtype=float)
	dec = np.asarray(dec, dtype=float)
	shape = np.broadcast_shapes(ra.shape, dec.shape)
	ra = np.broadcast_to(ra, shape)
	dec = np.broadcast_to(dec, shape)
	flat_ra = np.ravel(ra)
	flat_dec = np.ravel(dec)
	if hasattr(sph, "direct_paper_logprob_for_radec_c"):
		c_inputs = context.get("c_inputs")
		if c_inputs is None:
			raise RuntimeError("missing prepared C inputs in coherent-likelihood context")
		out_p = np.empty(flat_ra.size, dtype=np.float64)
		out_n = np.empty(flat_ra.size, dtype=np.float64)
		amplitude_model_id = {
			"circular": 0,
			"two_pol": 1,
			"cbc": 2,
			"paper_cross": 3,
		}[args.amplitude_model]
		sph.direct_paper_logprob_for_radec_c(
			c_inputs["locations"],
			c_inputs["responses"],
			c_inputs["freqs"],
			c_inputs["times_geo"],
			c_inputs["epochs"],
			c_inputs["data"],
			c_inputs["den_weight"],
			np.ascontiguousarray(flat_ra, dtype=np.float64),
			np.ascontiguousarray(flat_dec, dtype=np.float64),
				float(args.paper_phase_sign),
				float(args.direct_score_scale),
				int(args.paper_conj_response),
				int(amplitude_model_id),
				int(args.inclination_samples),
				int(args.polarization_samples),
				out_p,
				out_n,
			)
		if args.peak_time_sigma > 0.0:
			peak_log_prob = np.ravel(peak_time_consistency_logprob(context, instruments, args, flat_ra, flat_dec))
			out_p += peak_log_prob
			out_n += peak_log_prob
		return out_p.reshape(shape), out_n.reshape(shape)
	score_p = np.empty((context["times_geo"].size, flat_ra.size), dtype=float)
	score_n = np.empty_like(score_p)
	two_pi_i = 2j * np.pi * args.paper_phase_sign
	data = {}
	den_weight = {}
	for ifo in instruments:
		if args.paper_autocorr_template:
			template = context["aut_template"][ifo]
		else:
			template = 1.0
		data[ifo] = context["fft_snr"][ifo] * np.conj(template) * context["inv_noise"][ifo]
		den_weight[ifo] = float(np.sum(np.abs(template) ** 2 * context["inv_noise"][ifo]))
	cosi_grid = inclination_grid(args.inclination_samples)
	psi_grid = polarization_grid(args.polarization_samples)

	for it, tg in enumerate(context["times_geo"]):
			gps = lal.LIGOTimeGPS(float(tg))
			gmst = lal.GreenwichMeanSiderealTime(gps)
			num_p = np.zeros(flat_ra.size, dtype=np.complex128)
			num_n = np.zeros(flat_ra.size, dtype=np.complex128)
			den_p = np.zeros(flat_ra.size, dtype=float)
			den_n = np.zeros(flat_ra.size, dtype=float)
			b_plus = np.zeros(flat_ra.size, dtype=np.complex128)
			b_cross = np.zeros(flat_ra.size, dtype=np.complex128)
			g_pp = np.zeros(flat_ra.size, dtype=float)
			g_pc = np.zeros(flat_ra.size, dtype=float)
			g_cc = np.zeros(flat_ra.size, dtype=float)
			num_cbc = np.zeros((cosi_grid.size, psi_grid.size, flat_ra.size), dtype=np.complex128)
			den_cbc = np.zeros((cosi_grid.size, psi_grid.size, flat_ra.size), dtype=float)
			for ifo in instruments:
				delay, fplus, fcross = detector_response_and_delay(context["detectors"][ifo], flat_ra, flat_dec, gmst)
				phase_arg = tg + delay - context["epochs"][ifo]
				phase = np.exp(two_pi_i * phase_arg[:, None] * context["freqs"][None, :])
				shifted_sum = phase @ data[ifo]
				if args.amplitude_model == "cbc":
					for inc, cosi in enumerate(cosi_grid):
						aplus = 0.5 * (1.0 + cosi * cosi)
						across = cosi
						for pol, psi in enumerate(psi_grid):
							c2p = np.cos(2.0 * psi)
							s2p = np.sin(2.0 * psi)
							fplus_psi = fplus * c2p + fcross * s2p
							fcross_psi = -fplus * s2p + fcross * c2p
							response = aplus * fplus_psi + 1j * across * fcross_psi
							response_factor = np.conj(response) if args.paper_conj_response else response
							num_cbc[inc, pol] += response_factor * shifted_sum
							den_cbc[inc, pol] += np.abs(response) ** 2 * den_weight[ifo]
				elif args.amplitude_model == "two_pol":
					b_plus += fplus * shifted_sum
					b_cross += fcross * shifted_sum
					g_pp += fplus * fplus * den_weight[ifo]
					g_pc += fplus * fcross * den_weight[ifo]
					g_cc += fcross * fcross * den_weight[ifo]
				else:
					response_p = fplus + 1j * fcross
					response_n = fplus - 1j * fcross
					if args.paper_conj_response:
						response_factor_p = np.conj(response_p)
						response_factor_n = np.conj(response_n)
					else:
						response_factor_p = response_p
						response_factor_n = response_n
					num_p += response_factor_p * shifted_sum
					num_n += response_factor_n * shifted_sum
					den_p += np.abs(response_p) ** 2 * den_weight[ifo]
					den_n += np.abs(response_n) ** 2 * den_weight[ifo]
			if args.amplitude_model == "cbc":
				score = np.where(den_cbc > 0.0, np.abs(num_cbc) ** 2 / den_cbc, 0.0)
				log_prob = np.logaddexp.reduce(args.direct_score_scale * score.reshape(-1, flat_ra.size), axis=0) - np.log(cosi_grid.size * psi_grid.size)
				score_p[it] = log_prob
				score_n[it] = score_p[it]
			elif args.amplitude_model == "two_pol":
				det = g_pp * g_cc - g_pc * g_pc
				good = det > 1e-300
				score = np.zeros(flat_ra.size, dtype=float)
				score[good] = (
					g_cc[good] * np.abs(b_plus[good]) ** 2 +
					g_pp[good] * np.abs(b_cross[good]) ** 2 -
					2.0 * g_pc[good] * np.real(np.conj(b_plus[good]) * b_cross[good])
				) / det[good]
				score_p[it] = score
				score_n[it] = score
			else:
				score_p[it] = np.where(den_p > 0.0, np.abs(num_p) ** 2 / den_p, 0.0)
				score_n[it] = np.where(den_n > 0.0, np.abs(num_n) ** 2 / den_n, 0.0)

	if args.amplitude_model == "cbc":
		log_prob_p = np.logaddexp.reduce(score_p, axis=0) - np.log(context["times_geo"].size)
		log_prob_n = log_prob_p
	else:
		log_prob_p = np.logaddexp.reduce(args.direct_score_scale * score_p, axis=0) - np.log(context["times_geo"].size)
		log_prob_n = np.logaddexp.reduce(args.direct_score_scale * score_n, axis=0) - np.log(context["times_geo"].size)
	if args.peak_time_sigma > 0.0:
		peak_log_prob = np.ravel(peak_time_consistency_logprob(context, instruments, args, flat_ra, flat_dec))
		log_prob_p += peak_log_prob
		log_prob_n += peak_log_prob
	return log_prob_p.reshape(shape), log_prob_n.reshape(shape)


def coherent_likelihood_logprob_on_mesh(context, instruments, args, theta, phi):
	if args.verbose_pixels:
		print("coherent_likelihood_sh_series", len(theta), "x", len(phi), flush=True)
	theta_grid, phi_grid = np.meshgrid(theta, phi, indexing="ij")
	return coherent_likelihood_logprob_for_radec(context, instruments, args, phi_grid, np.pi / 2.0 - theta_grid)


def write_coherent_likelihood_sh_series(context, instruments, args, output_coeff_p, output_coeff_n, timer=None):
	timer = timer or StageTimer(False)
	lmax = args.direct_coeff_lmax
	if lmax is None:
		lmax = 3 * args.direct_nside - 1
	if lmax < 0:
		raise ValueError("--direct-coeff-lmax must be non-negative")
	with timer.stage("direct_coeff_grid"):
		ntheta = nphi = 2 * (lmax + 1)
		cos_theta, _ = legendre.leggauss(ntheta)
		cos_theta = cos_theta[::-1]
		theta = np.arccos(cos_theta)
		phi = np.arange(nphi, dtype=float) * (2.0 * np.pi / nphi)
	with timer.stage("direct_coeff_likelihood"):
		log_prob_p, log_prob_n = coherent_likelihood_logprob_on_mesh(context, instruments, args, theta, phi)
	if output_coeff_p and output_coeff_n:
		shift = float(max(np.nanmax(log_prob_p), np.nanmax(log_prob_n)))
	else:
		shift = None
	if output_coeff_p:
		with timer.stage("write_coeff_p"):
			write_logprob_mesh_sh_series(log_prob_p, output_coeff_p, lmax, shift=shift, log_floor=args.direct_coeff_log_floor)
	if output_coeff_n:
		with timer.stage("write_coeff_n"):
			write_logprob_mesh_sh_series(log_prob_n, output_coeff_n, lmax, shift=shift, log_floor=args.direct_coeff_log_floor)


def coherent_likelihood_map(full, aut, instruments, args, timer=None):
	"""Direct pixel/time/frequency implementation of the whitened CBC likelihood.

	This is the validated coherent likelihood path for CBC SNR dumps.  It
	evaluates a coherent frequency-domain score using the SNR autocorrelation
	spectrum as the SNR noise spectrum.
	"""

	if args.direct_nside <= 0:
		raise ValueError("--direct-nside must be positive")
	timer = timer or StageTimer(False)
	with timer.stage("direct_context"):
		context = prepare_coherent_likelihood_context(full, aut, instruments, args)
	resolve_direct_score_scale(args, context["length"])
	with timer.stage("direct_map_grid"):
		npix = hp.nside2npix(args.direct_nside)
		if args.verbose_pixels:
			print("coherent_likelihood", npix, "pixels", flush=True)
		theta, phi = hp.pix2ang(args.direct_nside, np.arange(npix))
	with timer.stage("direct_map_likelihood"):
		log_prob_p, log_prob_n = coherent_likelihood_logprob_for_radec(context, instruments, args, phi, np.pi / 2.0 - theta)

	with timer.stage("direct_map_normalize"):
		if args.likelihood_branch == "p":
			log_prob = log_prob_p
		elif args.likelihood_branch == "n":
			log_prob = log_prob_n
		else:
			log_prob = np.logaddexp(log_prob_p, log_prob_n) - np.log(2.0)
		prob = np.exp(log_prob - np.max(log_prob))
		prob /= np.sum(prob)
		prob_p = np.exp(log_prob_p - np.max(log_prob_p))
		prob_p /= np.sum(prob_p)
		prob_n = np.exp(log_prob_n - np.max(log_prob_n))
		prob_n /= np.sum(prob_n)
		if args.direct_smoothing_deg > 0.0:
			sigma = np.deg2rad(args.direct_smoothing_deg)
			prob = hp.smoothing(prob, sigma=sigma)
			prob = np.maximum(prob, 0.0)
			prob /= np.sum(prob)
			prob_p = hp.smoothing(prob_p, sigma=sigma)
			prob_p = np.maximum(prob_p, 0.0)
			prob_p /= np.sum(prob_p)
			prob_n = hp.smoothing(prob_n, sigma=sigma)
			prob_n = np.maximum(prob_n, 0.0)
			prob_n /= np.sum(prob_n)
		if args.isotropic_mixture:
			prob = apply_isotropic_mixture(prob, args.isotropic_mixture)
			prob_p = apply_isotropic_mixture(prob_p, args.isotropic_mixture)
			prob_n = apply_isotropic_mixture(prob_n, args.isotropic_mixture)
		if args.antenna_mixture:
			antenna_prob = network_antenna_probability(context, instruments, args, phi, np.pi / 2.0 - theta)
			prob = apply_probability_mixture(prob, antenna_prob, args.antenna_mixture, "antenna-mixture")
			prob_p = apply_probability_mixture(prob_p, antenna_prob, args.antenna_mixture, "antenna-mixture")
			prob_n = apply_probability_mixture(prob_n, antenna_prob, args.antenna_mixture, "antenna-mixture")
	print(
			"coherent_likelihood",
			"nside", args.direct_nside,
			"ntimes", len(context["times_geo"]),
			"length", context["length"],
				"nfreq", context["freq_mask_count"],
				"score_scale", args.direct_score_scale,
				"snr_window", args.paper_snr_window,
				"fmin", args.paper_fmin,
				"fmax", args.paper_fmax,
				"autocorr_floor", args.paper_autocorr_floor,
				"shift_autocorr", args.paper_shift_autocorr,
				"autocorr_template", args.paper_autocorr_template,
	)
	if args.peak_time_sigma > 0.0:
		print(
			"peak_time_quality",
				"enabled", peak_time_quality_passes(context, instruments, args),
				"min_peak_abs", args.peak_time_min_peak_abs,
				"min_prominence", args.peak_time_min_prominence,
				"uncertainty_model", args.peak_time_uncertainty_model,
				"reference_peak_abs", args.peak_time_reference_peak_abs,
				"uncertainty_power", args.peak_time_uncertainty_power,
				"max_sigma", args.peak_time_max_sigma,
				"peaks", ",".join(
					f"{ifo}:abs={context['peak_info'][ifo]['abs']:.6g}:prom={context['peak_info'][ifo]['prominence']:.6g}"
					for ifo in instruments
				),
			flush=True,
		)
	return prob, prob_p, prob_n, log_prob_p, log_prob_n, context


def _combined_log_prob(log_prob_p, log_prob_n, branch):
	if branch == "p":
		return log_prob_p
	if branch == "n":
		return log_prob_n
	return np.logaddexp(log_prob_p, log_prob_n) - np.log(2.0)


def _evaluate_coherent_pixels(context, instruments, args, nside, ipix):
	ipix = np.asarray(ipix, dtype=np.int64)
	theta, phi = hp.pix2ang(int(nside), ipix, nest=True)
	log_prob_p, log_prob_n = coherent_likelihood_logprob_for_radec(context, instruments, args, phi, np.pi / 2.0 - theta)
	return np.asarray(log_prob_p, dtype=float), np.asarray(log_prob_n, dtype=float)


def _select_refine_pixels(nside, ipix, log_density, top_pixels, probability, include_neighbors):
	ipix = np.asarray(ipix, dtype=np.int64)
	log_density = np.asarray(log_density, dtype=float)
	order = np.argsort(log_density)[::-1]
	selected = set(int(x) for x in ipix[order[:min(top_pixels, len(order))]])
	if probability > 0.0:
		shift = float(np.max(log_density))
		prob = np.exp(log_density - shift)
		prob /= np.sum(prob)
		csum = np.cumsum(prob[order])
		limit = int(np.searchsorted(csum, min(probability, 1.0), side="left") + 1)
		selected.update(int(x) for x in ipix[order[:limit]])
	if include_neighbors and selected:
		present = set(int(x) for x in ipix)
		for pix in list(selected):
			for neigh in hp.get_all_neighbours(int(nside), pix, nest=True):
				if neigh >= 0 and int(neigh) in present:
					selected.add(int(neigh))
	return np.asarray(sorted(selected), dtype=np.int64)


def coherent_likelihood_adaptive_map(full, aut, instruments, args, timer=None):
	if args.direct_nside <= 0:
		raise ValueError("--direct-nside must be positive")
	if args.adaptive_nside_stop < args.direct_nside:
		raise ValueError("--adaptive-nside-stop must be >= --direct-nside")
	if args.adaptive_refine_top_pixels <= 0:
		raise ValueError("--adaptive-refine-top-pixels must be positive")
	timer = timer or StageTimer(False)
	with timer.stage("direct_context"):
		context = prepare_coherent_likelihood_context(full, aut, instruments, args)
	resolve_direct_score_scale(args, context["length"])

	leaf_nside = np.full(hp.nside2npix(args.direct_nside), args.direct_nside, dtype=np.int64)
	leaf_ipix = np.arange(len(leaf_nside), dtype=np.int64)
	with timer.stage("adaptive_initial_likelihood"):
		leaf_log_p, leaf_log_n = _evaluate_coherent_pixels(context, instruments, args, args.direct_nside, leaf_ipix)
	evals = len(leaf_ipix)
	current_nside = args.direct_nside

	while current_nside < args.adaptive_nside_stop:
		at_level = leaf_nside == current_nside
		if not np.any(at_level):
			current_nside *= 2
			continue
		log_density = _combined_log_prob(leaf_log_p[at_level], leaf_log_n[at_level], args.likelihood_branch)
		refine_ipix = _select_refine_pixels(
			current_nside,
			leaf_ipix[at_level],
			log_density,
			args.adaptive_refine_top_pixels,
			args.adaptive_refine_probability,
			args.adaptive_refine_neighbors,
		)
		if refine_ipix.size == 0:
			break
		remaining = args.adaptive_max_evals - evals
		if remaining <= 0:
			break
		max_parents = remaining // 4
		if max_parents <= 0:
			break
		if refine_ipix.size > max_parents:
			refine_set = set(int(x) for x in refine_ipix)
			level_ipix = leaf_ipix[at_level]
			level_log = log_density
			order = np.argsort(level_log)[::-1]
			refine_ipix = np.asarray([int(x) for x in level_ipix[order] if int(x) in refine_set][:max_parents], dtype=np.int64)
		children = (refine_ipix[:, None] * 4 + np.arange(4, dtype=np.int64)[None, :]).ravel()
		child_nside = current_nside * 2
		with timer.stage(f"adaptive_likelihood_nside_{child_nside}"):
			child_log_p, child_log_n = _evaluate_coherent_pixels(context, instruments, args, child_nside, children)
		evals += len(children)
		remove = at_level & np.isin(leaf_ipix, refine_ipix)
		keep = ~remove
		leaf_nside = np.concatenate([leaf_nside[keep], np.full(len(children), child_nside, dtype=np.int64)])
		leaf_ipix = np.concatenate([leaf_ipix[keep], children])
		leaf_log_p = np.concatenate([leaf_log_p[keep], child_log_p])
		leaf_log_n = np.concatenate([leaf_log_n[keep], child_log_n])
		current_nside = child_nside

	with timer.stage("adaptive_normalize"):
		log_density = _combined_log_prob(leaf_log_p, leaf_log_n, args.likelihood_branch)
		area = 4.0 * np.pi / (12.0 * leaf_nside.astype(float) ** 2)
		log_mass = log_density + np.log(area)
		shift = float(np.max(log_mass))
		prob = np.exp(log_mass - shift)
		prob /= np.sum(prob)
	print(
		"coherent_likelihood_adaptive",
		"nside_start", args.direct_nside,
		"nside_stop", args.adaptive_nside_stop,
		"leaf_pixels", len(prob),
		"evals", evals,
		"top_pixels", args.adaptive_refine_top_pixels,
		"refine_probability", args.adaptive_refine_probability,
		"neighbors", args.adaptive_refine_neighbors,
	)
	return leaf_nside, leaf_ipix, prob, context


def main():
	parser = argparse.ArgumentParser()
	parser.add_argument("--calc-snr-dir", required=True)
	parser.add_argument("--output-fits", required=True)
	parser.add_argument("--output-fits-p")
	parser.add_argument("--output-fits-n")
	parser.add_argument("--output-coeff-p")
	parser.add_argument("--output-coeff-n")
	parser.add_argument("--no-output-coefficients", action="store_true", help="do not write default p/n spherical-harmonic coefficient FITS files")
	parser.add_argument("--output-png")
	parser.add_argument("--output-png-p")
	parser.add_argument("--output-png-n")
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
	parser.add_argument("--time-gate-center", type=float)
	parser.add_argument("--time-gate-half-width", type=float)
	parser.add_argument("--backend", choices=("coherent_likelihood", "harmonic", "direct_time_prior"), default="coherent_likelihood", help="skymap likelihood backend")
	parser.add_argument("--direct-time-prior", action="store_true")
	parser.add_argument("--coherent-likelihood", action="store_true")
	parser.add_argument("--direct-nside", type=int, default=32)
	parser.add_argument("--adaptive", action="store_true", help="write an adaptive NUNIQ/MOC coherent-likelihood skymap")
	parser.add_argument("--adaptive-nside-stop", type=int, default=256)
	parser.add_argument("--adaptive-refine-top-pixels", type=int, default=128)
	parser.add_argument("--adaptive-refine-probability", type=float, default=0.0)
	parser.add_argument("--adaptive-refine-neighbors", action="store_true")
	parser.add_argument("--adaptive-max-evals", type=int, default=20000)
	parser.add_argument("--direct-coeff-lmax", type=int)
	parser.add_argument("--direct-coeff-log-floor", type=float, default=50.0)
	parser.add_argument("--direct-score-scale", default="-1", help="coherent-likelihood score scale; <=0 uses 1 / SNR-window length; 'snr_baseline_ramp' ramps from that baseline to an SNR/baseline-count cap")
	parser.add_argument("--snr-ramp-low-snr", type=float, default=10.0)
	parser.add_argument("--snr-ramp-high-snr", type=float, default=50.0)
	parser.add_argument("--snr-ramp-two-ifo-cap", type=float, default=2e-4)
	parser.add_argument("--snr-ramp-multi-ifo-cap", type=float, default=2.5e-4)
	parser.add_argument("--isotropic-mixture", type=float, default=0.0, help="mix this fraction of isotropic probability into fixed-grid coherent-likelihood maps")
	parser.add_argument("--antenna-mixture", type=float, default=0.0, help="mix this fraction of network antenna-response probability into fixed-grid coherent-likelihood maps")
	parser.add_argument("--network-snr", type=float, help="candidate network SNR used by containment calibration")
	parser.add_argument("--containment-calibration", action="store_true", help="apply the built-in SNR-dependent containment calibration")
	parser.add_argument("--low-snr-threshold", type=float, default=8.0)
	parser.add_argument("--low-snr-score-scale", type=float, default=1e-4)
	parser.add_argument("--low-snr-smoothing-deg", type=float, default=8.0)
	parser.add_argument("--low-snr-isotropic-mixture", type=float, default=0.0)
	parser.add_argument("--low-snr-antenna-mixture", type=float, default=0.0)
	parser.add_argument("--high-snr-threshold", type=float, default=50.0)
	parser.add_argument("--high-snr-score-scale", type=float, default=2e-4)
	parser.add_argument("--high-snr-smoothing-deg", type=float, default=0.0)
	parser.add_argument("--high-snr-isotropic-mixture", type=float, default=0.0)
	parser.add_argument("--high-snr-antenna-mixture", type=float, default=0.0)
	parser.add_argument("--direct-smoothing-deg", type=float, default=0.0)
	parser.add_argument("--paper-autocorr-floor", type=float, default=1e-6)
	parser.add_argument("--paper-tukey-alpha", type=float, default=0.1, help="Tukey taper applied to the SNR and autocorrelation windows before the direct coherent FFT, matching the native sphradiometer preprocessing")
	parser.add_argument("--paper-snr-window", choices=("full", "trigger"), default="full", help="SNR samples used by the coherent paper-cross FFT: full dumped time series or the historical short trigger-centered autocorrelation-length window")
	parser.add_argument("--paper-fmin", type=float, default=0.0)
	parser.add_argument("--paper-fmax", type=float, default=512.0)
	parser.add_argument("--paper-flat-noise", action="store_true")
	parser.add_argument("--paper-no-shift-autocorr", dest="paper_shift_autocorr", action="store_false")
	parser.add_argument("--paper-delta-template", dest="paper_autocorr_template", action="store_false")
	parser.add_argument("--paper-phase-sign", type=float, choices=(-1.0, 1.0), default=1.0)
	parser.add_argument("--paper-conj-response", action="store_true")
	parser.add_argument("--amplitude-model", choices=("paper_cross", "cbc", "two_pol", "circular"), default="paper_cross", help="coherent amplitude model: paper cross-correlation statistic, CBC inclination grid, arbitrary complex plus/cross amplitudes, or pure circular branches")
	parser.add_argument("--paper-cross-weighting", choices=("raw", "inv_noise", "autocorr_template"), default="raw", help="frequency weighting used by --amplitude-model paper_cross before forming detector-pair cross-products")
	parser.add_argument("--peak-time-sigma", type=float, default=0.0, help="optional Gaussian sigma, in seconds, for a lightweight detector peak-time consistency term")
	parser.add_argument("--peak-time-min-snr", type=float, default=0.0, help="legacy network-SNR gate for --peak-time-sigma; disabled when <=0")
	parser.add_argument("--peak-time-min-peak-abs", type=float, default=8.0, help="only apply --peak-time-sigma when every detector SNR time series has at least this peak absolute SNR; set <=0 to disable")
	parser.add_argument("--peak-time-min-prominence", type=float, default=0.0, help="only apply --peak-time-sigma when every detector peak passes this local prominence threshold; set <=0 to disable")
	parser.add_argument("--peak-time-uncertainty-model", choices=("fixed", "peak_abs"), default="fixed", help="model used to broaden the detector peak-time consistency term for weak SNR peaks")
	parser.add_argument("--peak-time-reference-peak-abs", type=float, default=20.0, help="reference single-detector peak abs SNR for --peak-time-uncertainty-model peak_abs")
	parser.add_argument("--peak-time-uncertainty-power", type=float, default=2.0, help="power-law broadening exponent for --peak-time-uncertainty-model peak_abs")
	parser.add_argument("--peak-time-max-sigma", type=float, default=0.008, help="maximum pair timing sigma for --peak-time-uncertainty-model peak_abs; set <=0 for no cap")
	parser.add_argument("--no-two-ifo-timing-prior", dest="two_ifo_timing_prior", action="store_false", help="disable the default soft peak-time consistency term for two-detector coherent-likelihood maps")
	parser.add_argument("--two-ifo-peak-time-sigma", type=float, default=0.00025, help="base Gaussian timing sigma, in seconds, for the default two-detector timing prior")
	parser.add_argument("--two-ifo-peak-time-min-peak-abs", type=float, default=14.0, help="minimum per-detector SNR time-series peak required to apply the default two-detector timing prior")
	parser.add_argument("--two-ifo-peak-time-reference-peak-abs", type=float, default=20.0, help="reference per-detector peak SNR used to broaden the default two-detector timing prior")
	parser.add_argument("--two-ifo-peak-time-uncertainty-power", type=float, default=2.0, help="power-law broadening exponent for the default two-detector timing prior")
	parser.add_argument("--two-ifo-peak-time-max-sigma", type=float, default=0.008, help="maximum timing sigma for the default two-detector timing prior")
	parser.add_argument("--inclination-samples", type=int, default=5, help="number of cos(inclination) samples for --amplitude-model cbc")
	parser.add_argument("--polarization-samples", type=int, default=4, help="number of polarization-angle samples for --amplitude-model cbc")
	parser.add_argument("--verbose-pixels", action="store_true")
	parser.add_argument("--timing", action="store_true", help="print internal stage timings")
	parser.add_argument("--time-prior-center", type=float)
	parser.add_argument("--time-prior-half-width", type=float)
	parser.add_argument("--time-prior-step", type=float)
	parser.add_argument("--precalc-len", type=int, default=438)
	parser.add_argument("--sample-rate", type=float, default=2048.0)
	parser.add_argument("--effective-sample-rate", type=int, default=512)
	parser.add_argument("--scale-power", type=float, default=0.6)
	parser.add_argument("--scale-overall", type=float, default=2.7)
	parser.add_argument("--likelihood-branch", choices=("both", "p", "n"), default="both")
	parser.add_argument("--include-auto-terms", action="store_true")
	parser.add_argument("--auto-term-scale", type=float)
	parser.add_argument("--include-whitening", action="store_true")
	parser.add_argument("--regulator-scale", type=float, default=-1.0, help="C likelihood regulator scale; negative preserves the historical n_baselines * 4 default")
	parser.add_argument("--conjugate-delay-product", action="store_true")
	parser.add_argument("--swap-frequency-product", action="store_true")
	parser.add_argument("--runtime-projection-gmst", action="store_true")
	parser.add_argument("--runtime-projection-frame-gmst-scale", type=float, default=1.0)
	parser.add_argument("--runtime-frame-gmst-only", action="store_true")
	parser.add_argument("--no-autocorr-template", dest="use_autocorrelation_template", action="store_false")
	parser.add_argument("--no-derotate", action="store_true")
	parser.add_argument("--ra-deg", type=float)
	parser.add_argument("--dec-deg", type=float)
	parser.add_argument("--contour", nargs="+", default=["50", "90"])
	parser.set_defaults(paper_shift_autocorr=True, paper_autocorr_template=True, use_autocorrelation_template=True)
	args = parser.parse_args()
	if args.direct_time_prior:
		args.backend = "direct_time_prior"
	elif args.coherent_likelihood:
		args.backend = "coherent_likelihood"
	if args.backend == "direct_time_prior":
		args.direct_time_prior = True
	elif args.backend == "coherent_likelihood":
		args.coherent_likelihood = True
		if args.time_prior_center is None:
			args.time_prior_center = args.center_time
		if args.time_prior_half_width is None:
			args.time_prior_half_width = 0.008
	elif args.backend == "harmonic":
		print("warning: harmonic uses the native fast sphradiometer cross-power statistic; coherent_likelihood is the validated CBC likelihood backend")
	apply_containment_calibration(args)
	if not args.no_output_coefficients:
		default_coeff_p, default_coeff_n = default_coeff_paths(args.output_fits)
		if not args.output_coeff_p:
			args.output_coeff_p = default_coeff_p
		if not args.output_coeff_n:
			args.output_coeff_n = default_coeff_n
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
		snr, dt = build_common_window(full, aut, args.center_time, args.pre_trigger, args.start_idx, args.time_gate_center, args.time_gate_half_width)
	instruments = sorted(snr)
	args.active_instruments = instruments
	apply_two_ifo_timing_defaults(args, instruments)

	if args.direct_time_prior:
		with timer.stage("direct_time_prior"):
			prob, prob_p, prob_n = direct_time_prior_map(full, instruments, args)
		with timer.stage("write_map_fits"):
			write_prob_fits(prob, args.output_fits)
		if args.output_coeff_p:
			with timer.stage("write_coeff_p"):
				write_prob_alm(prob_p, args.output_coeff_p)
		if args.output_coeff_n:
			with timer.stage("write_coeff_n"):
				write_prob_alm(prob_n, args.output_coeff_n)
		if args.output_fits_p or args.output_png_p:
			output_fits_p = args.output_fits_p or os.path.splitext(args.output_png_p)[0] + ".fits"
			with timer.stage("write_fits_p"):
				write_prob_fits(prob_p, output_fits_p)
			if args.output_png_p:
				with timer.stage("plot_png_p"):
					write_skymap_png(output_fits_p, args.output_png_p, args.ra_deg, args.dec_deg, args.contour)
		if args.output_fits_n or args.output_png_n:
			output_fits_n = args.output_fits_n or os.path.splitext(args.output_png_n)[0] + ".fits"
			with timer.stage("write_fits_n"):
				write_prob_fits(prob_n, output_fits_n)
			if args.output_png_n:
				with timer.stage("plot_png_n"):
					write_skymap_png(output_fits_n, args.output_png_n, args.ra_deg, args.dec_deg, args.contour)
		if args.output_png:
			with timer.stage("plot_png"):
				write_skymap_png(args.output_fits, args.output_png, args.ra_deg, args.dec_deg, args.contour)
		timer.report()
		return

	if args.coherent_likelihood:
		if len(instruments) == 1:
			prob, prob_p, prob_n = single_detector_antenna_map(instruments[0], args, timer=timer)
			context = None
		else:
			if args.adaptive:
				leaf_nside, leaf_ipix, leaf_prob, context = coherent_likelihood_adaptive_map(full, aut, instruments, args, timer=timer)
				with timer.stage("write_map_fits"):
					write_moc_fits(leaf_nside, leaf_ipix, leaf_prob, args.output_fits)
				if args.output_png:
					with timer.stage("plot_png"):
						write_skymap_png(args.output_fits, args.output_png, args.ra_deg, args.dec_deg, args.contour)
				timer.report()
				return
			prob, prob_p, prob_n, _, _, context = coherent_likelihood_map(full, aut, instruments, args, timer=timer)
		with timer.stage("write_map_fits"):
			write_prob_fits(prob, args.output_fits)
		if args.output_coeff_p or args.output_coeff_n:
			if len(instruments) == 1:
				write_single_detector_antenna_sh_series(instruments[0], args, args.output_coeff_p, args.output_coeff_n, timer=timer)
			else:
				write_coherent_likelihood_sh_series(context, instruments, args, args.output_coeff_p, args.output_coeff_n, timer=timer)
		if args.output_fits_p or args.output_png_p:
			output_fits_p = args.output_fits_p or os.path.splitext(args.output_png_p)[0] + ".fits"
			with timer.stage("write_fits_p"):
				write_prob_fits(prob_p, output_fits_p)
			if args.output_png_p:
				with timer.stage("plot_png_p"):
					write_skymap_png(output_fits_p, args.output_png_p, args.ra_deg, args.dec_deg, args.contour)
		if args.output_fits_n or args.output_png_n:
			output_fits_n = args.output_fits_n or os.path.splitext(args.output_png_n)[0] + ".fits"
			with timer.stage("write_fits_n"):
				write_prob_fits(prob_n, output_fits_n)
			if args.output_png_n:
				with timer.stage("plot_png_n"):
					write_skymap_png(output_fits_n, args.output_png_n, args.ra_deg, args.dec_deg, args.contour)
		if args.output_png:
			with timer.stage("plot_png"):
				write_skymap_png(args.output_fits, args.output_png, args.ra_deg, args.dec_deg, args.contour)
		timer.report()
		return

	precalc_metadata = {}
	runtime_psds = None
	if args.precalc_dir:
		loc = RapidLocalization.read(args.precalc_dir, instruments=instruments)
		precalc_metadata = read_precalc_metadata(args.precalc_dir, instruments)
		if precalc_metadata:
			print("read precompute metadata", precalc_metadata["_metadata_path"])
	else:
		if not args.psd_xml:
			raise ValueError("--psd-xml is required when --precalc-dir is not provided")
		psds = load_relative_psd_weights(args.psd_xml, instruments, args.precalc_len, args.sample_rate, args.mode)
		loc = RapidLocalization_(psds, args.precalc_len, dt, effective_sample_rate=args.effective_sample_rate)
		runtime_psds = psds

	runtime_projection_gmst = None
	runtime_projection_frame_gmst = 0.0
	precalc_has_projection = precalc_metadata.get("apply_projection", True)
	if args.precalc_dir and not precalc_has_projection and not args.runtime_projection_gmst:
		print("delay-only precompute detected; enabling full runtime GMST projection")
		args.runtime_projection_gmst = True
	if args.runtime_projection_gmst:
		runtime_projection_gmst = lal.GreenwichMeanSiderealTime(lal.LIGOTimeGPS(args.center_time))
		runtime_projection_frame_gmst = args.runtime_projection_frame_gmst_scale * runtime_projection_gmst
		if args.precalc_dir and not args.runtime_frame_gmst_only:
			psd_xml = args.psd_xml or precalc_metadata.get("psd_xml")
			mode = args.mode or precalc_metadata.get("mode", "asd")
			precalc_len = int(precalc_metadata.get("precalc_len", args.precalc_len))
			sample_rate = float(precalc_metadata.get("sample_rate", args.sample_rate))
			if not psd_xml:
				raise ValueError("--psd-xml is required for full runtime harmonic projection when precompute metadata is missing")
			runtime_psds = load_relative_psd_weights(psd_xml, instruments, precalc_len, sample_rate, mode)
	skyp, skyn = loc.sphcoeff(
		snr,
		aut,
		power=args.scale_power,
		overall=args.scale_overall,
		auto_term_scale=args.auto_term_scale if args.auto_term_scale is not None else float(args.include_auto_terms),
		include_whitening=args.include_whitening,
		regulator_scale=args.regulator_scale,
		conjugate_delay_product=args.conjugate_delay_product,
		swap_frequency_product=args.swap_frequency_product,
		time_prior_center=args.time_prior_center or 0.0,
		time_prior_half_width=args.time_prior_half_width if args.time_prior_half_width is not None else -1.0,
		use_autocorrelation_template=args.use_autocorrelation_template,
		runtime_projection_gmst=runtime_projection_gmst,
		runtime_projection_frame_gmst=runtime_projection_frame_gmst,
		runtime_frame_gmst_only=args.runtime_frame_gmst_only,
		runtime_psds=runtime_psds,
	)
	if not args.no_derotate:
		gmst = lal.GreenwichMeanSiderealTime(lal.LIGOTimeGPS(args.center_time))
		derotate_to_celestial(skyp, gmst)
		derotate_to_celestial(skyn, gmst)
	sky = rapidskyloc_io(skyp, skyn, coinc_event_id=args.coinc_event_id)
	if args.likelihood_branch == "p":
		sky.prob = coeff_series_to_prob(skyp)
	elif args.likelihood_branch == "n":
		sky.prob = coeff_series_to_prob(skyn)

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

	if args.output_fits_p or args.output_png_p:
		output_fits_p = args.output_fits_p or os.path.splitext(args.output_png_p)[0] + ".fits"
		write_prob_fits(coeff_series_to_prob(skyp), output_fits_p)
		if args.output_png_p:
			write_skymap_png(output_fits_p, args.output_png_p, args.ra_deg, args.dec_deg, args.contour)
	if args.output_fits_n or args.output_png_n:
		output_fits_n = args.output_fits_n or os.path.splitext(args.output_png_n)[0] + ".fits"
		write_prob_fits(coeff_series_to_prob(skyn), output_fits_n)
		if args.output_png_n:
			write_skymap_png(output_fits_n, args.output_png_n, args.ra_deg, args.dec_deg, args.contour)

	if args.output_png:
		write_skymap_png(args.output_fits, args.output_png, args.ra_deg, args.dec_deg, args.contour)


if __name__ == "__main__":
	main()
