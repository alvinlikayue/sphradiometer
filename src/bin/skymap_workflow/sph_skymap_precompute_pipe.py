#!/usr/bin/env python3
"""Generate a DAG that precomputes sphradiometer components for one PSD."""

import argparse
import configparser
import itertools
import os
import shutil

from gstlal.htcondor_helper import baseDAG, base_htcondor_layer
from sph_skymap_common import condor_options_from_config, without_inherited_container


def getboolean(cp, section, option, fallback=False):
	if not cp.has_option(section, option):
		return fallback
	return cp.getboolean(section, option)


def parse_instruments(value):
	return sorted([ifo.strip() for ifo in value.replace(",", " ").split() if ifo.strip()])


def detector_combinations(instruments, spec):
	spec = str(spec or "all").strip()
	if spec == "all":
		return [
			"".join(combo)
			for n in range(1, len(instruments) + 1)
			for combo in itertools.combinations(instruments, n)
		]
	if spec == "multi":
		return [
			"".join(combo)
			for n in range(2, len(instruments) + 1)
			for combo in itertools.combinations(instruments, n)
		]
	return [item.strip() for item in spec.replace(" ", ",").split(",") if item.strip()]


def executable(name):
	return shutil.which(name) or os.path.join(os.path.dirname(os.path.abspath(__file__)), name)


def main():
	parser = argparse.ArgumentParser()
	parser.add_argument("config")
	args = parser.parse_args()

	cp = configparser.ConfigParser()
	cp.read(args.config)

	config_dir = os.path.dirname(os.path.abspath(args.config))
	dag_dir = config_dir
	instruments = parse_instruments(cp.get("precompute", "instruments"))
	combinations = detector_combinations(instruments, cp.get("precompute", "combinations", fallback="all"))
	mode = cp.get("precompute", "mode", fallback="asd")
	precalc_len = cp.get("precompute", "precalc_len", fallback="438")
	default_precalc_name = f"{''.join(instruments)}_{mode}_{precalc_len}"
	precalc_dir = os.path.abspath(cp.get(
		"precompute",
		"precalc_dir",
		fallback=os.path.join(dag_dir, "outputs", "precalc", default_precalc_name),
	))

	os.makedirs(dag_dir, exist_ok=True)
	os.makedirs(os.path.join(dag_dir, "logs"), exist_ok=True)
	os.makedirs(os.path.join(dag_dir, "outputs"), exist_ok=True)
	os.makedirs(precalc_dir, exist_ok=True)

	condor_opts = condor_options_from_config(
		cp,
		dag_dir,
		default_request_cpus="2",
		default_request_memory="6GB",
		default_request_disk="4GB",
	)

	with without_inherited_container():
		layer = base_htcondor_layer(
			executable=executable("sph_skymap_precompute.py"),
			node_name="sph_skymap_precompute",
			output_path=os.path.join(dag_dir, "sph_skymap_precompute"),
			**condor_opts,
		)

	opts_list = []
	io_list = []
	for combo in combinations:
		opts = {
			"--psd-xml": os.path.abspath(cp.get("precompute", "psd_xml")),
			"--output-dir": precalc_dir,
			"--instruments": ",".join(instruments),
			"--combination": combo,
			"--mode": mode,
			"--precalc-len": precalc_len,
			"--delta-t": cp.get("precompute", "delta_t", fallback=str(1.0 / 2048.0)),
			"--sample-rate": cp.get("precompute", "sample_rate", fallback="2048"),
			"--effective-sample-rate": cp.get("precompute", "effective_sample_rate", fallback="512"),
		}
		if not getboolean(cp, "precompute", "apply_projection", fallback=True):
			opts["--delay-only"] = ""
		if getboolean(cp, "precompute", "force", fallback=True):
			opts["--force"] = ""
		opts_list.append(opts)
		io_list.append({
			"inputs": [os.path.abspath(cp.get("precompute", "psd_xml"))],
			"outputs": [os.path.join(precalc_dir, combo)],
		})

	layer.batch_set_arguments(opts_list, list_of_io=io_list)

	dag = baseDAG(dag_dir=dag_dir, dag_filename=cp.get("workflow", "dag_filename", fallback="precompute_skymap.dag"))
	dag.add_layer(layer=layer, category="sph_skymap_precompute", retries=cp.getint("workflow", "retries", fallback=1))
	dag.write()

	print("Wrote DAG:", os.path.join(dag_dir, dag.dag_filename))
	print("Precompute nodes:", len(combinations), ", ".join(combinations))
	print("Submit with:")
	print("  cd", dag_dir)
	print("  condor_submit_dag", dag.dag_filename)


if __name__ == "__main__":
	main()
