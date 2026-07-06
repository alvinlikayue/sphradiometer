#!/usr/bin/env python3
"""Generate a harmonic-only skymap DAG from a manifest."""

import argparse
import configparser
import csv
import os
import shutil

from gstlal.htcondor_helper import baseDAG, base_htcondor_layer
from sph_skymap_common import condor_options_from_config, without_inherited_container


def read_manifest(path):
	with open(path, newline="") as f:
		rows = list(csv.DictReader(f))
	if not rows:
		raise ValueError(f"manifest is empty: {path}")
	return rows


def executable(name):
	return shutil.which(name) or os.path.join(os.path.dirname(os.path.abspath(__file__)), name)


def row_tag(row, index):
	sim_id = row.get("simulation_id", "").strip()
	coinc_id = row.get("coinc_event_id", "").strip()
	if sim_id:
		return f"sim{sim_id}"
	if coinc_id:
		return f"coinc{coinc_id}"
	return f"row{index:06d}"


def row_calc_snr_dir(row, output_root, tag):
	value = row.get("calc_snr_dir", "").strip()
	if value:
		return os.path.abspath(value)
	return os.path.join(output_root, "calc_snr", tag)


def main():
	parser = argparse.ArgumentParser()
	parser.add_argument("config")
	args = parser.parse_args()

	cp = configparser.ConfigParser()
	cp.read(args.config)
	config_dir = os.path.dirname(os.path.abspath(args.config))
	dag_dir = config_dir
	output_root = os.path.abspath(cp.get("workflow", "output_root", fallback=os.path.join(dag_dir, "outputs")))
	manifest = os.path.abspath(cp.get("manifest", "path"))
	rows = read_manifest(manifest)

	for name in ("logs", "skymap", "coeff", "skymap_branch"):
		os.makedirs(os.path.join(output_root if name != "logs" else dag_dir, name), exist_ok=True)

	condor_opts = condor_options_from_config(cp, dag_dir)
	with without_inherited_container():
		layer = base_htcondor_layer(
			executable=executable("sph_skymap_from_calc_snr.py"),
			node_name="sph_skymap_from_calc_snr",
			output_path=os.path.join(dag_dir, "sph_skymap_from_calc_snr"),
			**condor_opts,
		)

	precalc_dir = cp.get("skymap", "precalc_dir", fallback="").strip()
	psd_xml = cp.get("skymap", "psd_xml", fallback="").strip()
	write_coefficients = cp.getboolean("skymap", "write_coefficients", fallback=True)
	write_branch_maps = cp.getboolean("skymap", "write_branch_maps", fallback=False)

	opts_list = []
	io_list = []
	for index, row in enumerate(rows):
		tag = row_tag(row, index)
		calc_dir = row_calc_snr_dir(row, output_root, tag)
		fits = os.path.join(output_root, "skymap", f"{tag}.fits")
		coeff_p = os.path.join(output_root, "coeff", f"{tag}_coeffp.fits")
		coeff_n = os.path.join(output_root, "coeff", f"{tag}_coeffn.fits")
		fits_p = os.path.join(output_root, "skymap_branch", f"{tag}_p.fits")
		fits_n = os.path.join(output_root, "skymap_branch", f"{tag}_n.fits")

		opts = {
			"": calc_dir,
			"--output-fits": fits,
			"--bank-number": row["bank_number"],
			"--row-number": row["row_number"],
			"--center-time": row["center_time"],
			"--pre-trigger": row.get("pre_trigger", cp.get("skymap", "pre_trigger", fallback="0.12")),
			"--mode": cp.get("skymap", "mode", fallback="asd"),
			"--precalc-len": cp.get("skymap", "precalc_len", fallback="438"),
			"--sample-rate": cp.get("skymap", "sample_rate", fallback="2048"),
			"--effective-sample-rate": cp.get("skymap", "effective_sample_rate", fallback="512"),
		}
		if row.get("coinc_event_id", "").strip():
			opts["--coinc-event-id"] = row["coinc_event_id"]
		if row.get("instruments", "").strip():
			opts["--instruments"] = row["instruments"]
		if row.get("start_idx", "").strip():
			opts["--start-idx"] = row["start_idx"]
		if precalc_dir:
			opts["--precalc-dir"] = os.path.abspath(precalc_dir)
		else:
			row_psd = row.get("reference_psd", "").strip() or psd_xml
			if not row_psd:
				raise ValueError("set [skymap] precalc_dir, [skymap] psd_xml, or manifest reference_psd")
			opts["--psd-xml"] = os.path.abspath(row_psd)
		if write_coefficients:
			opts["--output-coeff-p"] = coeff_p
			opts["--output-coeff-n"] = coeff_n
		else:
			opts["--no-output-coefficients"] = ""
		if write_branch_maps:
			opts["--output-fits-p"] = fits_p
			opts["--output-fits-n"] = fits_n
		if cp.getboolean("skymap", "include_whitening", fallback=False):
			opts["--include-whitening"] = ""
		if cp.getboolean("skymap", "no_derotate", fallback=False):
			opts["--no-derotate"] = ""
		if cp.getboolean("skymap", "timing", fallback=False):
			opts["--timing"] = ""

		outputs = [fits]
		if write_coefficients:
			outputs.extend([coeff_p, coeff_n])
		if write_branch_maps:
			outputs.extend([fits_p, fits_n])
		opts_list.append(opts)
		io_list.append({"inputs": [calc_dir], "outputs": outputs})

	layer.batch_set_arguments(opts_list, list_of_io=io_list)
	dag = baseDAG(dag_dir=dag_dir, dag_filename=cp.get("workflow", "dag_filename", fallback="injection_skymaps.dag"))
	dag.add_layer(layer=layer, category="sph_skymap_from_calc_snr", retries=cp.getint("workflow", "retries", fallback=1))
	dag.write()

	print("Wrote DAG:", os.path.join(dag_dir, dag.dag_filename))
	print("Skymap nodes:", len(rows))
	print("Submit with:")
	print("  cd", dag_dir)
	print("  condor_submit_dag", dag.dag_filename)


if __name__ == "__main__":
	main()
