#!/usr/bin/env python3

import argparse
import csv
import glob
import math
import os
import re
import sys

import healpy
import numpy


def parse_args():
	parser = argparse.ArgumentParser(
		description="Summarize sphradiometer spherical-harmonic coefficient FITS files."
	)
	parser.add_argument("--coeff-dir", required=True, help="Directory containing *_coeffp.fits and *_coeffn.fits")
	parser.add_argument("--output", required=True, help="Output CSV path")
	parser.add_argument("--low-l-max", type=int, default=8, help="Upper l included in the low-l power fraction")
	parser.add_argument("--high-l-min", type=int, default=32, help="Lower l included in the high-l power fraction")
	return parser.parse_args()


def event_id_from_coeff_path(path, suffix):
	name = os.path.basename(path)
	if not name.endswith(suffix):
		raise ValueError(f"unexpected coefficient filename: {path}")
	return name[:-len(suffix)]


def event_sort_key(event):
	match = re.search(r"(\d+)$", event)
	return (event[:match.start()] if match else event, int(match.group(1)) if match else -1, event)


def find_pairs(coeff_dir):
	p_files = {
		event_id_from_coeff_path(path, "_coeffp.fits"): path
		for path in glob.glob(os.path.join(coeff_dir, "*_coeffp.fits"))
	}
	n_files = {
		event_id_from_coeff_path(path, "_coeffn.fits"): path
		for path in glob.glob(os.path.join(coeff_dir, "*_coeffn.fits"))
	}
	events = sorted(set(p_files) | set(n_files), key=event_sort_key)
	for event in events:
		yield event, p_files.get(event, ""), n_files.get(event, "")


def alm_lm_indices(l_max, m_max, n):
	if m_max == 0:
		return numpy.arange(n, dtype=int), numpy.zeros(n, dtype=int)
	if m_max != l_max:
		raise ValueError(f"unsupported m_max={m_max}; expected 0 or l_max={l_max}")
	return healpy.Alm.getlm(l_max, numpy.arange(n))


def alm_power_by_l(path):
	alm, m_max = healpy.fitsfunc.read_alm(path, return_mmax=True)
	l_max = healpy.Alm.getlmax(len(alm), m_max)
	ell, emm = alm_lm_indices(l_max, m_max, len(alm))
	weights = numpy.where(emm == 0, 1.0, 2.0)
	power = numpy.bincount(ell, weights=weights * numpy.abs(alm) ** 2, minlength=l_max + 1)
	return power, l_max, m_max


def entropy(prob):
	prob = prob[prob > 0]
	if len(prob) == 0:
		return 0.0
	return float(-numpy.sum(prob * numpy.log(prob)))


def summarize_one(path, low_l_max, high_l_min):
	if not path:
		return {
			"l_max": "",
			"m_max": "",
			"total_power": "",
			"l_peak": "",
			"peak_power_fraction": "",
			"low_l_fraction": "",
			"high_l_fraction": "",
			"spectral_entropy": "",
		}
	power, l_max, m_max = alm_power_by_l(path)
	total = float(numpy.sum(power))
	if total > 0:
		norm = power / total
		l_peak = int(numpy.argmax(power))
		peak_frac = float(norm[l_peak])
		low_frac = float(numpy.sum(norm[:min(low_l_max, l_max) + 1]))
		high_frac = float(numpy.sum(norm[high_l_min:])) if high_l_min <= l_max else 0.0
		ent = entropy(norm) / math.log(len(norm)) if len(norm) > 1 else 0.0
	else:
		l_peak = 0
		peak_frac = low_frac = high_frac = ent = 0.0
	return {
		"l_max": str(l_max),
		"m_max": str(m_max),
		"total_power": f"{total:.17g}",
		"l_peak": str(l_peak),
		"peak_power_fraction": f"{peak_frac:.17g}",
		"low_l_fraction": f"{low_frac:.17g}",
		"high_l_fraction": f"{high_frac:.17g}",
		"spectral_entropy": f"{ent:.17g}",
	}


def prefixed(prefix, values):
	return {f"{prefix}_{key}": value for key, value in values.items()}


def main():
	args = parse_args()
	rows = []
	for event, coeffp, coeffn in find_pairs(args.coeff_dir):
		p = summarize_one(coeffp, args.low_l_max, args.high_l_min)
		n = summarize_one(coeffn, args.low_l_max, args.high_l_min)
		p_total = float(p["total_power"] or 0.0)
		n_total = float(n["total_power"] or 0.0)
		rows.append({
			"event_id": event,
			"coeffp": coeffp,
			"coeffn": coeffn,
			**prefixed("p", p),
			**prefixed("n", n),
			"pn_total_power_ratio": f"{(p_total / n_total):.17g}" if n_total else "",
			"total_power_sum": f"{(p_total + n_total):.17g}",
		})

	if not rows:
		print(f"error: no coefficient pairs found in {args.coeff_dir}", file=sys.stderr)
		return 1

	fieldnames = [
		"event_id", "coeffp", "coeffn",
		"p_l_max", "p_m_max", "p_total_power", "p_l_peak", "p_peak_power_fraction", "p_low_l_fraction", "p_high_l_fraction", "p_spectral_entropy",
		"n_l_max", "n_m_max", "n_total_power", "n_l_peak", "n_peak_power_fraction", "n_low_l_fraction", "n_high_l_fraction", "n_spectral_entropy",
		"pn_total_power_ratio", "total_power_sum",
	]
	os.makedirs(os.path.dirname(os.path.abspath(args.output)) or ".", exist_ok=True)
	with open(args.output, "w", newline="") as f:
		writer = csv.DictWriter(f, fieldnames=fieldnames)
		writer.writeheader()
		writer.writerows(rows)
	print(f"Wrote {len(rows)} rows to {args.output}")
	return 0


if __name__ == "__main__":
	raise SystemExit(main())
