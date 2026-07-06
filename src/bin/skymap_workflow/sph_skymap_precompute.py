#!/usr/bin/env python3
"""Precompute sphradiometer localization objects for a PSD and detector set."""

import argparse
import json
import os
import shutil

import numpy as np
from sphradiometer.RapidLocalization import RapidLocalization, RapidLocalization_
from sphradiometer import sphradiometer as sph

from sph_skymap_common import load_relative_psd_weights


def parse_instruments(value):
	return sorted([ifo.strip() for ifo in value.split(",") if ifo.strip()])


def main():
	parser = argparse.ArgumentParser()
	parser.add_argument("--psd-xml", required=True)
	parser.add_argument("--output-dir", required=True)
	parser.add_argument("--instruments", required=True, help="comma-separated, e.g. H1,L1,V1")
	parser.add_argument("--combination", help="single detector combination to compute, e.g. H1L1V1")
	parser.add_argument("--mode", choices=("flat", "asd", "psd"), default="asd")
	parser.add_argument("--precalc-len", type=int, default=438)
	parser.add_argument("--delta-t", type=float, default=1.0 / 2048.0)
	parser.add_argument("--sample-rate", type=float, default=2048.0)
	parser.add_argument("--effective-sample-rate", type=int, default=512)
	parser.add_argument("--delay-only", dest="apply_projection", action="store_false", help="store delay-only harmonic plans")
	parser.add_argument("--force", action="store_true")
	parser.set_defaults(apply_projection=True)
	args = parser.parse_args()

	instruments = parse_instruments(args.instruments)
	if not instruments:
		raise ValueError("no instruments requested")

	if args.combination:
		combo = "".join(parse_instruments(",".join([args.combination[i:i + 2] for i in range(0, len(args.combination), 2)])))
		if set(parse_instruments(",".join([combo[i:i + 2] for i in range(0, len(combo), 2)]))) - set(instruments):
			raise ValueError(f"combination {args.combination} is not a subset of {instruments}")
		combo_instruments = parse_instruments(",".join([combo[i:i + 2] for i in range(0, len(combo), 2)]))
		combo_dir = os.path.join(args.output_dir, combo)
	else:
		combo = None
		combo_instruments = instruments
		combo_dir = args.output_dir

	os.makedirs(args.output_dir, exist_ok=True)
	if os.path.exists(combo_dir):
		if not args.force:
			raise FileExistsError(f"{combo_dir} exists; use --force to replace")
		shutil.rmtree(combo_dir)

	psds = load_relative_psd_weights(
		args.psd_xml,
		combo_instruments,
		args.precalc_len,
		args.sample_rate,
		args.mode,
	)

	for ifo in combo_instruments:
		arr = np.array([sph.double_array_getitem(psds[ifo].psd, i) for i in range(args.precalc_len)])
		print(
			ifo,
			"relative_weight_min",
			float(arr.min()),
			"median",
			float(np.median(arr)),
			"max",
			float(arr.max()),
		)

	if combo:
		loc = RapidLocalization_(
			psds,
			args.precalc_len,
			args.delta_t,
			effective_sample_rate=args.effective_sample_rate,
		)
		loc.write(combo_dir)
	else:
		loc = RapidLocalization(
			psds,
			args.precalc_len,
			args.delta_t,
			effective_sample_rate=args.effective_sample_rate,
		)
		loc.write(args.output_dir)

	metadata_dir = args.output_dir.rstrip(os.sep) + ".metadata"
	os.makedirs(metadata_dir, exist_ok=True)
	metadata_path = os.path.join(metadata_dir, f"{combo or 'all'}.json")
	with open(metadata_path, "w") as f:
		json.dump(
			{
				"psd_xml": os.path.abspath(args.psd_xml),
				"instruments": combo_instruments,
				"combination": combo,
				"mode": args.mode,
				"precalc_len": args.precalc_len,
				"delta_t": args.delta_t,
				"sample_rate": args.sample_rate,
				"effective_sample_rate": args.effective_sample_rate,
				"apply_projection": args.apply_projection,
			},
			f,
			indent=2,
			sort_keys=True,
		)

	print("wrote", os.path.abspath(combo_dir))


if __name__ == "__main__":
	main()
