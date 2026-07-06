#!/usr/bin/env python3

import argparse
import os
import subprocess

os.environ.setdefault("MPLCONFIGDIR", os.path.join("/tmp", "sphradiometer-matplotlib"))


def parse_args():
	parser = argparse.ArgumentParser(description = "Run BAYESTAR for one GstLAL coinc_event_id using gstlal_inspiral_calc_snr time series.")
	parser.add_argument("--input-sqlite", required = True)
	parser.add_argument("--calc-snr-dir", required = True)
	parser.add_argument("--coinc-event-id", required = True)
	parser.add_argument("--output-fits", required = True)
	parser.add_argument("--output-png", default = "")
	parser.add_argument("--psd-xml", default = "")
	parser.add_argument("--f-low", default = "")
	parser.add_argument("--f-high-truncate", default = "")
	parser.add_argument("--waveform", default = "")
	parser.add_argument("--min-distance", default = "")
	parser.add_argument("--max-distance", default = "")
	parser.add_argument("--prior-distance-power", default = "")
	parser.add_argument("--rescale-loglikelihood", default = "")
	parser.add_argument("--omp-num-threads", default = "")
	parser.add_argument("--disable-snr-series", action = "store_true")
	parser.add_argument("--keep-going", action = "store_true")
	parser.add_argument("--loglevel", default = "")
	parser.add_argument("--snr-series-half-width", type = float, default = 0.1)
	return parser.parse_args()


def value_or_none(value, cast = float):
	if value == "":
		return None
	return cast(value)


def load_calc_snr_series(calc_snr_dir):
	import glob
	import ligolw
	import ligolw.utils
	from lalmetaio import series as lalseries

	out = {}
	for path in sorted(glob.glob(os.path.join(calc_snr_dir, "*-LLOID_bank_SNR_*.xml*"))):
		ifo = os.path.basename(path).split("-", 1)[0]
		doc = ligolw.utils.load_filename(path)
		try:
			series = [
				lalseries.parse_COMPLEX8TimeSeries(elem)
				for elem in doc.getElementsByTagName(ligolw.LIGO_LW.tagName)
				if elem.hasAttribute("Name") and elem.Name == "COMPLEX8TimeSeries"
			]
			if not series:
				raise ValueError(f"{path}: no COMPLEX8TimeSeries found")
			if len(series) > 1:
				raise ValueError(f"{path}: expected one COMPLEX8TimeSeries, found {len(series)}")
			out[ifo] = series[0]
		finally:
			doc.unlink()
	if not out:
		raise FileNotFoundError(f"no calc-SNR XML files found in {calc_snr_dir}")
	return out


def centered_odd_cut(series, center_time, half_width):
	import lal
	import numpy

	delta_t = float(series.deltaT)
	nsamples_half = max(1, int(round(float(half_width) / delta_t)))
	length = 2 * nsamples_half + 1
	start_time = float(series.epoch)
	center_index = int(round((float(center_time) - start_time) / delta_t))
	first = center_index - nsamples_half
	if first < 0 or first + length > len(series.data.data):
		raise ValueError(
			f"SNR series centered at {center_time} with half-width {half_width} s "
			f"is outside [{start_time}, {start_time + (len(series.data.data) - 1) * delta_t}]"
		)
	cut = lal.CutCOMPLEX8TimeSeries(series, first, length)
	# Force the epoch to the exact trigger-centered value expected by BAYESTAR,
	# avoiding nanosecond-level roundoff from the nearest-sample calculation.
	cut.epoch = lal.LIGOTimeGPS(float(center_time) - nsamples_half * delta_t)
	if not numpy.isfinite(cut.data.data).all():
		raise ValueError("SNR series contains non-finite samples")
	return cut


def main():
	args = parse_args()
	env = os.environ.copy()
	if args.omp_num_threads:
		env["OMP_NUM_THREADS"] = str(args.omp_num_threads)
	os.makedirs(os.path.dirname(os.path.abspath(args.output_fits)), exist_ok = True)
	if args.output_png:
		os.makedirs(os.path.dirname(os.path.abspath(args.output_png)), exist_ok = True)

	import logging
	if args.loglevel:
		logging.basicConfig(level = getattr(logging, args.loglevel.upper()))

	from ligo.skymap.bayestar import localize
	from ligo.skymap.io import events, fits

	with open(args.input_sqlite, "rb") as f:
		event_source = events.open(f)
		event = event_source[int(args.coinc_event_id)]

	if args.psd_xml:
		for single in event.singles:
			single._psd_file = args.psd_xml

	if not args.disable_snr_series:
		series_by_ifo = load_calc_snr_series(args.calc_snr_dir)
		for single in event.singles:
			try:
				series = series_by_ifo[single.detector]
			except KeyError as exc:
				raise KeyError(f"missing calc-SNR series for {single.detector} in {args.calc_snr_dir}") from exc
			single._snr_series = centered_odd_cut(series, single.time, args.snr_series_half_width)

	sky_map = localize(
		event,
		waveform = args.waveform or "o2-uberbank",
		f_low = value_or_none(args.f_low),
		min_distance = value_or_none(args.min_distance),
		max_distance = value_or_none(args.max_distance),
		prior_distance_power = value_or_none(args.prior_distance_power, int),
		enable_snr_series = not args.disable_snr_series,
		f_high_truncate = value_or_none(args.f_high_truncate) if args.f_high_truncate else 0.95,
		rescale_loglikelihood = value_or_none(args.rescale_loglikelihood) if args.rescale_loglikelihood else 0.83,
	)
	sky_map.meta["objid"] = int(args.coinc_event_id)
	fits.write_sky_map(args.output_fits, sky_map, nest = True)

	if args.output_png:
		subprocess.run(["ligo-skymap-plot", args.output_fits, "--annotate", "--output", args.output_png], check = True, env = env)


if __name__ == "__main__":
	main()
