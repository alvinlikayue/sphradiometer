#!/usr/bin/env python3
"""Generate a DAG that turns a GstLAL candidate manifest into skymaps."""

import argparse
import configparser
import csv
import glob
import importlib.util
import os
import re
import shutil
import sqlite3

from lal.utils import CacheEntry
import ligolw.utils
from ligolw import lsctables
from gstlal.htcondor_helper import baseDAG, base_htcondor_layer
from sph_skymap_common import condor_options_from_config, without_inherited_container


def log(message):
	print(f"[sph_skymap_injection_pipe] {message}", flush=True)


try:
	from gstlal import diagnostics
except ImportError:
	diagnostics = None

if diagnostics is None or any(not hasattr(diagnostics, name) for name in ("svd_bank_strings_from_cache", "query_noise_candidates_from_sqlite", "make_noise_candidate_manifest_rows")):
	source_diagnostics = "/src/gstlal/gstlal-ugly/python/diagnostics.py"
	if os.path.exists(source_diagnostics):
		spec = importlib.util.spec_from_file_location("gstlal_ugly_diagnostics", source_diagnostics)
		diagnostics = importlib.util.module_from_spec(spec)
		spec.loader.exec_module(diagnostics)


def abs_if_set(value):
	value = str(value or "").strip()
	return os.path.abspath(value) if value else ""


def split_channels(value):
	return [item.strip() for item in value.replace("\n", ",").split(",") if item.strip()]


def read_manifest(path):
	with open(path, newline="") as f:
		rows = list(csv.DictReader(f))
	if not rows:
		raise ValueError(f"manifest is empty: {path}")
	return rows


def executable(name):
	return shutil.which(name) or os.path.join(os.path.dirname(os.path.abspath(__file__)), name)


def split_combo(combo):
	return [combo[i:i + 2] for i in range(0, len(combo), 2)]


def combo_from_ifos(ifos):
	return "".join(sorted(ifo.strip() for ifo in ifos if ifo.strip()))


def canonical_combo(combo):
	if "," in combo:
		return combo_from_ifos(combo.split(","))
	return combo_from_ifos(split_combo(combo))


def discover_channels(run_dir):
	configs = [
		os.path.join(run_dir, "finalized_O3a_chunk8.ini"),
		os.path.join(run_dir, "O3a_chunk8.ini"),
	]
	for path in configs:
		if not os.path.exists(path):
			continue
		channels = {}
		ifo = None
		for line in open(path):
			match = re.match(r"\s*\[\[\[(H1|L1|V1|K1)\]\]\]", line)
			if match:
				ifo = match.group(1)
				continue
			match = re.match(r"\s*channel_name\s*=\s*(.+?)\s*$", line)
			if match and ifo:
				channels[ifo] = match.group(1).strip().strip('"')
		if channels:
			return ",".join(f"{ifo}={channels[ifo]}" for ifo in sorted(channels))
	raise FileNotFoundError(f"could not discover detector channel names in {run_dir}")


def parse_channel_map(channel_string):
	channels = {}
	for item in split_channels(channel_string):
		ifo, channel = item.split("=", 1)
		channels[ifo] = channel
	return channels


def channels_for_combo(channel_map, combo):
	return [f"{ifo}={channel_map[ifo]}" for ifo in split_combo(canonical_combo(combo))]


def skymap_backend(cp):
	backend = cp.get("skymap", "backend", fallback="coherent_likelihood").strip().lower()
	aliases = {
		"harmonic": "harmonic",
		"spherical_harmonic": "harmonic",
		"sph": "harmonic",
		"direct": "direct_time_prior",
		"direct_time": "direct_time_prior",
		"direct_time_prior": "direct_time_prior",
		"coherent_likelihood": "coherent_likelihood",
		"coherent": "coherent_likelihood",
		"coherent_cbc": "coherent_likelihood",
	}
	if backend not in aliases:
		raise ValueError("[skymap] backend must be 'harmonic', 'direct_time_prior', or 'coherent_likelihood'")
	return aliases[backend]


def row_skymap_backend(cp, backend, instruments):
	if backend == "harmonic" and len(split_combo(instruments)) < 3:
		fallback = cp.get("skymap", "low_ifo_backend", fallback="coherent_likelihood").strip().lower()
		if fallback in ("", "harmonic", "none", "off", "false"):
			return backend
		fallback_cp = configparser.ConfigParser()
		fallback_cp.add_section("skymap")
		fallback_cp.set("skymap", "backend", fallback)
		return skymap_backend(fallback_cp)
	return backend


def resolve_time_prior_center(cp, row):
	setting = cp.get("skymap", "time_prior_center", fallback="auto").strip()
	if setting.lower() in ("", "auto"):
		return row["center_time"]
	if setting.lower() in ("simulation_time", "truth"):
		return row.get("simulation_time") or row["center_time"]
	if setting.lower() == "center_time":
		return row["center_time"]
	return setting


def resolve_time_prior_half_width(cp, row):
	setting = cp.get("skymap", "time_prior_half_width", fallback="0.008").strip()
	if setting.lower() in ("snr_window", "full_snr_window"):
		if row.get("snr_start") and row.get("snr_end"):
			center = float(resolve_time_prior_center(cp, row))
			return str(max(abs(center - float(row["snr_start"])), abs(float(row["snr_end"]) - center)))
		return "0.008"
	if setting.lower() in ("", "auto"):
		return "0.008"
	return setting


def discover_first_existing(run_dir, names):
	for name in names:
		path = os.path.join(run_dir, name)
		if os.path.exists(path):
			return path
	return ""


def configured_gstlal_database(cp):
	if not cp.has_section("gstlal"):
		return ""
	for option in ("injection_database", "noise_database", "candidate_database", "database"):
		if cp.has_option("gstlal", option) and cp.get("gstlal", option).strip():
			return abs_if_set(cp.get("gstlal", option))
	return ""


def prepare_bayestar_sqlite(input_sqlite, output_sqlite, f_final):
	if not input_sqlite:
		raise ValueError("[bayestar] enabled requires [bayestar] database or a [gstlal] database")
	os.makedirs(os.path.dirname(os.path.abspath(output_sqlite)), exist_ok=True)
	if os.path.abspath(input_sqlite) != os.path.abspath(output_sqlite):
		shutil.copy2(input_sqlite, output_sqlite)
	with sqlite3.connect(output_sqlite) as conn:
		columns = {row[1] for row in conn.execute("PRAGMA table_info(sngl_inspiral)")}
		if "f_final" not in columns:
			conn.execute("ALTER TABLE sngl_inspiral ADD COLUMN f_final REAL")
		conn.execute("UPDATE sngl_inspiral SET f_final = ? WHERE f_final IS NULL OR f_final <= 0", (float(f_final),))
		conn.commit()
	return output_sqlite


def discover_template_match_files(run_dir):
	files = sorted(glob.glob(os.path.join(run_dir, "gstlal_inspiral_injection_template_match", "*INJECTION_TEMPLATE_MATCH.xml.gz")))
	if not files:
		raise FileNotFoundError("no injection-template-match XMLs found under gstlal_inspiral_injection_template_match")
	return files


def discover_svd_bank_strings_by_combo(run_dir):
	cache_patterns = [os.path.join(run_dir, "gstlal_inspiral_inj", "cache", "*_SVD_*.cache")]
	if not glob.glob(cache_patterns[0]):
		cache_patterns.append(os.path.join(run_dir, "gstlal_inspiral", "cache", "*_SVD-*.cache"))
	svd_banks = {}
	for pattern in cache_patterns:
		for cache in sorted(glob.glob(pattern)):
			combo = canonical_combo(os.path.basename(cache).split("-")[0])
			for svd_bank in diagnostics.svd_bank_strings_from_cache(cache):
				svd_banks.setdefault(combo, [])
				if svd_bank not in svd_banks[combo]:
					svd_banks[combo].append(svd_bank)
	if not svd_banks:
		raise FileNotFoundError("no SVD cache files found in the GstLAL run")
	return svd_banks


def discover_reference_psds_by_combo(run_dir):
	psds = {}
	for path in glob.glob(os.path.join(run_dir, "gstlal_reference_psd", "*", "*-REFERENCE_PSD-*.xml.gz")):
		combo = canonical_combo(os.path.basename(path).split("-")[0])
		try:
			entry = CacheEntry.from_T050017(path)
			psds.setdefault(combo, []).append((float(entry.segment[0]), float(entry.segment[1]), path))
		except Exception:
			match = re.search(r"REFERENCE_PSD-(\d+)-(\d+)\.xml", os.path.basename(path))
			if match:
				start = float(match.group(1))
				end = start + float(match.group(2))
				psds.setdefault(combo, []).append((start, end, path))
	if not psds:
		raise FileNotFoundError("no reference PSD XMLs found under gstlal_reference_psd")
	return {combo: sorted(values) for combo, values in psds.items()}


def load_segments_by_ifo(segments_xml):
	doc = ligolw.utils.load_filename(segments_xml)
	try:
		defs = {row.segment_def_id: row.ifos for row in lsctables.SegmentDefTable.get_table(doc)}
		segs = {}
		for row in lsctables.SegmentTable.get_table(doc):
			ifo = defs[row.segment_def_id]
			segs.setdefault(ifo, []).append((float(row.start_time), float(row.end_time)))
		return segs
	finally:
		doc.unlink()


def ifos_in_segments(segments_by_ifo, time):
	time = float(time)
	return [
		ifo for ifo in sorted(segments_by_ifo)
		if any(start <= time < end for start, end in segments_by_ifo[ifo])
	]


def ifos_covering_interval(segments_by_ifo, start_time, end_time):
	start_time = float(start_time)
	end_time = float(end_time)
	return [
		ifo for ifo in sorted(segments_by_ifo)
		if any(start <= start_time and end_time <= end for start, end in segments_by_ifo[ifo])
	]


def choose_reference_psd(psds, center_time):
	center_time = float(center_time)
	for start, end, path in psds:
		if start <= center_time < end:
			return path
	return min(psds, key=lambda item: abs((item[0] + item[1]) / 2.0 - center_time))[2]


def gstlal_database_mode(cp):
	mode = cp.get("gstlal", "database_type", fallback="").strip().lower()
	if not mode:
		if cp.has_option("gstlal", "noise_database") or cp.has_option("gstlal", "candidate_database"):
			mode = "noise"
		else:
			mode = "injection"
	if mode in ("background", "zerolag", "candidate", "candidates"):
		mode = "noise"
	if mode not in ("injection", "noise"):
		raise ValueError("[gstlal] database_type must be 'injection' or 'noise'")
	return mode


def gstlal_database_path(cp, mode):
	if mode == "noise":
		for option in ("noise_database", "candidate_database", "database", "injection_database"):
			if cp.has_option("gstlal", option):
				return abs_if_set(cp.get("gstlal", option))
	else:
		for option in ("injection_database", "database"):
			if cp.has_option("gstlal", option):
				return abs_if_set(cp.get("gstlal", option))
	raise ValueError(f"[gstlal] is missing a SQLite database option for {mode} mode")


def build_auto_manifest(cp, dag_dir):
	if diagnostics is None:
		raise ImportError("gstlal.diagnostics is required for [gstlal] auto-discovery mode")

	run_dir = abs_if_set(cp.get("gstlal", "run_dir"))
	mode = gstlal_database_mode(cp)
	database = gstlal_database_path(cp, mode)
	output_root = os.path.join(dag_dir, "outputs")
	os.makedirs(output_root, exist_ok=True)
	log(f"auto mode enabled ({mode})")
	log(f"gstlal run_dir = {run_dir}")
	log(f"{mode} database = {database}")
	log(f"workflow output root = {output_root}")

	manifest_cp = configparser.ConfigParser()
	manifest_cp.add_section("manifest")
	manifest_cp.set("manifest", "sqlite", database)
	manifest_cp.set("manifest", "association", cp.get("manifest", "association", fallback="exact"))
	manifest_cp.set("manifest", "order_by", cp.get("manifest", "order_by", fallback="far"))
	manifest_cp.set("manifest", "order", cp.get("manifest", "order", fallback="asc"))
	manifest_cp.set("manifest", "limit", cp.get("manifest", "limit", fallback=""))
	manifest_cp.set("manifest", "allow_missing_bank_rows", cp.get("manifest", "allow_missing_bank_rows", fallback="false"))
	for key in ("far_lt", "far_lte", "far_gt", "far_gte", "snr_lt", "snr_lte", "snr_gt", "snr_gte", "likelihood_lt", "likelihood_lte", "likelihood_gt", "likelihood_gte", "where", "instruments", "exclude_time_windows", "exclude_injection_associations", "zero_lag_only", "max_time_offset"):
		if cp.has_option("manifest", key):
			manifest_cp.set("manifest", key, cp.get("manifest", key))
	for key, fallback in (("gps_pad", "70"), ("snr_pad", "1"), ("pre_trigger", "0.12"), ("row_counts", "1")):
		manifest_cp.set("manifest", key, cp.get("manifest", key, fallback=fallback))
	if cp.has_section("calc_snr"):
		manifest_cp.set("manifest", "ht_gate_threshold", cp.get("calc_snr", "ht_gate_threshold", fallback=""))
	segments_xml = discover_first_existing(run_dir, ["segments.xml.gz", "segments.xml"])
	channel_string = discover_channels(run_dir)
	channel_map = parse_channel_map(channel_string)
	manifest_cp.set("manifest", "run_dir", run_dir)
	manifest_cp.set("manifest", "channel_name", channel_string)

	if mode == "noise":
		log(
			"querying non-injection SQLite "
			f"(order_by={manifest_cp.get('manifest', 'order_by')}, "
			f"order={manifest_cp.get('manifest', 'order')}, "
			f"limit={manifest_cp.get('manifest', 'limit') or 'none'})"
		)
		rows = diagnostics.query_noise_candidates_from_sqlite(manifest_cp)
		log(f"selected {len(rows)} noise/candidate rows from SQLite")
		log("resolving noise candidates to SVD bank rows and reference PSDs")
		manifest_rows = diagnostics.make_noise_candidate_manifest_rows(manifest_cp, rows)
		log(f"resolved {len(manifest_rows)} noise/candidate manifest rows")
		bank_map_path = ""
	else:
		manifest_cp.set("manifest", "injection_xml", discover_first_existing(run_dir, ["bbh_injections.xml", "injections.xml"]))

		log(
			"querying injection SQLite "
			f"(association={manifest_cp.get('manifest', 'association')}, "
			f"order_by={manifest_cp.get('manifest', 'order_by')}, "
			f"order={manifest_cp.get('manifest', 'order')}, "
			f"limit={manifest_cp.get('manifest', 'limit') or 'none'})"
		)
		rows = diagnostics.query_recovered_injections_from_sqlite(manifest_cp)
		log(f"selected {len(rows)} recovered injection rows from SQLite")
		min_ifo = int(cp.get("manifest", "min_ifo", fallback="1"))
		rows = [
			row for row in rows
			if len(row["instruments"].split(",")) >= min_ifo
		]
		combo_counts = {}
		for row in rows:
			combo = combo_from_ifos(row["instruments"].split(","))
			combo_counts[combo] = combo_counts.get(combo, 0) + 1
		log("selected rows by detector combo: " + ", ".join(f"{combo}={count}" for combo, count in sorted(combo_counts.items())))
		bank_map_path = ""
		log("resolving recovered injection triggers to SVD bank rows and reference PSDs")
		manifest_rows = diagnostics.make_injection_manifest_rows(manifest_cp, rows)
		log(f"resolved {len(manifest_rows)} recovered injection manifest rows")
		log("discovering reference PSD files")
		psds_by_combo = discover_reference_psds_by_combo(run_dir)
		log("found reference PSD files by combo: " + ", ".join(f"{combo}={len(values)}" for combo, values in sorted(psds_by_combo.items())))
		log("assigning reference PSD to each manifest row")
		for row in manifest_rows:
			combo = canonical_combo(row["instruments"])
			row["instruments"] = combo
			if combo not in psds_by_combo:
				raise FileNotFoundError(f"no reference PSD files found for detector combo {combo}")
			row["reference_psd"] = choose_reference_psd(psds_by_combo[combo], row["center_time"])

	manifest_path = os.path.join(output_root, "autogenerated_manifest.csv")
	with open(manifest_path, "w", newline="") as f:
		writer = csv.DictWriter(f, fieldnames=diagnostics.INJECTION_MANIFEST_COLUMNS)
		writer.writeheader()
		writer.writerows(manifest_rows)
	log(f"wrote autogenerated manifest: {manifest_path}")

	discovered = {
		"manifest": manifest_path,
		"frame_cache": abs_if_set(cp.get("gstlal", "frame_cache", fallback="")) or discover_first_existing(run_dir, ["frame.cache", "original_frame.cache"]),
		"segments_xml": segments_xml,
		"channel_name": channel_string,
		"reference_psd": manifest_rows[0]["reference_psd"] if manifest_rows else "",
		"injection_xml": discover_first_existing(run_dir, ["bbh_injections.xml", "injections.xml"]) if mode == "injection" else "",
		"injection_time_slide_file": discover_first_existing(run_dir, ["inj_tisi.xml", "injection_timeslides.xml", "injection_time_slides.xml"]) if mode == "injection" else "",
	}
	print("Auto-discovered GstLAL run products:")
	for key, value in sorted(discovered.items()):
		print(f"  {key}: {value}")
	if bank_map_path:
		print(f"  bank_row_map: {bank_map_path}")
	print(f"  manifest_rows: {len(manifest_rows)}")
	return read_manifest(manifest_path), discovered


def main():
	parser = argparse.ArgumentParser()
	parser.add_argument("config")
	args = parser.parse_args()

	cp = configparser.ConfigParser()
	read_files = cp.read(args.config)
	if not read_files:
		raise FileNotFoundError(f"could not read config file: {args.config}")

	dag_dir = os.path.dirname(os.path.abspath(args.config))
	output_root = os.path.join(dag_dir, "outputs")
	calc_snr_root = os.path.join(output_root, "calc_snr")
	skymap_root = os.path.join(output_root, "skymap")
	bayestar_root = os.path.join(output_root, "bayestar")
	skymap_branch_root = os.path.join(output_root, "skymap_branch")
	coeff_root = os.path.join(output_root, "coeff")
	log(f"reading config: {os.path.abspath(args.config)}")
	log(f"DAG directory: {dag_dir}")

	os.makedirs(dag_dir, exist_ok=True)
	os.makedirs(os.path.join(dag_dir, "logs"), exist_ok=True)
	os.makedirs(output_root, exist_ok=True)
	os.makedirs(calc_snr_root, exist_ok=True)
	os.makedirs(skymap_root, exist_ok=True)
	os.makedirs(bayestar_root, exist_ok=True)
	os.makedirs(skymap_branch_root, exist_ok=True)
	os.makedirs(coeff_root, exist_ok=True)

	if cp.has_section("gstlal") and cp.has_option("gstlal", "run_dir") and any(cp.has_option("gstlal", option) for option in ("injection_database", "noise_database", "candidate_database", "database")):
		manifest, discovered = build_auto_manifest(cp, dag_dir)
	elif cp.has_section("injections") and cp.has_option("injections", "manifest"):
		log(f"reading explicit manifest: {cp.get('injections', 'manifest')}")
		manifest = read_manifest(cp.get("injections", "manifest"))
		discovered = {}
	else:
		sections = ", ".join(cp.sections()) or "none"
		raise ValueError(
			"config must either define [gstlal] run_dir and a database "
			"for auto mode, or [injections] manifest for explicit mode. "
			f"Parsed sections: {sections}. Config path: {os.path.abspath(args.config)}"
		)
	log(f"manifest rows to schedule: {len(manifest)}")
	condor_opts = condor_options_from_config(
		cp,
		dag_dir,
		default_request_cpus="1",
		default_request_memory="4GB",
		default_request_disk="4GB",
	)
	log("building HTCondor layers")

	with without_inherited_container():
		calc_layer = base_htcondor_layer(
			executable="gstlal_inspiral_calc_snr",
			node_name="gstlal_inspiral_calc_snr",
			output_path=os.path.join(dag_dir, "gstlal_inspiral_calc_snr"),
			**condor_opts,
		)
		map_layer = base_htcondor_layer(
			executable=executable("sph_skymap_from_calc_snr.py"),
			node_name="sph_skymap_from_calc_snr",
			output_path=os.path.join(dag_dir, "sph_skymap_from_calc_snr"),
			**condor_opts,
		)
		bayestar_layer = base_htcondor_layer(
			executable=executable("sph_bayestar_from_sqlite.py"),
			node_name="sph_bayestar_from_sqlite",
			output_path=os.path.join(dag_dir, "sph_bayestar_from_sqlite"),
			**condor_opts,
		)

	channel_names = split_channels(cp.get("data", "channel_name", fallback=discovered.get("channel_name", "")))
	if not channel_names:
		raise ValueError("channel names were not configured or discovered")
	log(f"using {len(channel_names)} channels: {', '.join(channel_names)}")
	svd_banks = cp.get("data", "svd_bank", fallback="")
	backend = skymap_backend(cp)
	log(f"using skymap backend: {backend}")
	write_coefficients = cp.getboolean("skymap", "write_coefficients", fallback=True)
	log(f"writing coefficient FITS files: {write_coefficients}")
	write_branch_maps = cp.getboolean("skymap", "write_branch_maps", fallback=True)
	log(f"writing p/n branch FITS files: {write_branch_maps}")
	write_png = cp.getboolean("skymap", "write_png", fallback=True)
	log(f"writing PNG plots: {write_png}")
	skymap_timing = cp.getboolean("skymap", "timing", fallback=False)
	log(f"printing skymap timing: {skymap_timing}")
	run_bayestar = cp.getboolean("bayestar", "enabled", fallback=False)
	bayestar_write_png = cp.getboolean("bayestar", "write_png", fallback=True)
	bayestar_database = ""
	if run_bayestar:
		bayestar_database = abs_if_set(cp.get("bayestar", "database", fallback="")) or configured_gstlal_database(cp)
		bayestar_database = prepare_bayestar_sqlite(
			bayestar_database,
			os.path.join(bayestar_root, "input_with_f_final.sqlite"),
			cp.get("bayestar", "f_final", fallback="1024"),
		)
		log(f"prepared BAYESTAR input SQLite: {bayestar_database}")
		log(f"writing BAYESTAR PNG plots: {bayestar_write_png}")
	precalc_dir = abs_if_set(cp.get("skymap", "precalc_dir", fallback=""))
	psd_xml = abs_if_set(cp.get("data", "reference_psd", fallback=discovered.get("reference_psd", "")))
	global_injection_xml = abs_if_set(cp.get("injections", "injection_xml", fallback=discovered.get("injection_xml", "")))
	frame_cache = abs_if_set(cp.get("data", "frame_cache", fallback=discovered.get("frame_cache", "")))
	segments_xml = abs_if_set(cp.get("data", "segments_xml", fallback=discovered.get("segments_xml", "")))
	time_slide_file = abs_if_set(cp.get("data", "time_slide_file", fallback=discovered.get("injection_time_slide_file", "")))
	veto_segments_file = abs_if_set(cp.get("data", "veto_segments_file", fallback=""))

	calc_opts_list = []
	calc_io_list = []
	map_opts_list = []
	map_io_list = []
	bayestar_opts_list = []
	bayestar_io_list = []
	map_backend_counts = {}

	for row in manifest:
		sim_id = row.get("simulation_id") or row.get("id")
		if not sim_id:
			raise ValueError("manifest row is missing simulation_id")
		tag = f"sim{sim_id}"
		calc_dir = os.path.join(calc_snr_root, tag)
		os.makedirs(calc_dir, exist_ok=True)
		fits = os.path.join(skymap_root, f"{tag}.fits")
		fits_p = os.path.join(skymap_branch_root, f"{tag}_p.fits")
		fits_n = os.path.join(skymap_branch_root, f"{tag}_n.fits")
		coeff_p = os.path.join(coeff_root, f"{tag}_coeffp.fits")
		coeff_n = os.path.join(coeff_root, f"{tag}_coeffn.fits")
		png = os.path.join(skymap_root, f"{tag}.png")
		bayestar_fits = os.path.join(bayestar_root, f"{tag}_bayestar.fits")
		bayestar_png = os.path.join(bayestar_root, f"{tag}_bayestar.png")

		row_psd = abs_if_set(row.get("reference_psd", "")) or psd_xml
		row_svd_bank = row.get("svd_bank", "") or svd_banks
		row_channel_names = split_channels(row.get("channel_name", "")) or channel_names
		if not row_svd_bank:
			raise ValueError(f"manifest row for simulation {sim_id} has no svd_bank")
		if not row_channel_names:
			raise ValueError(f"manifest row for simulation {sim_id} has no channel_name")
		calc_opts = {
			"--verbose": "",
			"--data-source": cp.get("data", "data_source", fallback="frames"),
			"--gps-start-time": row["gps_start_time"],
			"--gps-end-time": row["gps_end_time"],
			"--start": row.get("snr_start", row["gps_start_time"]),
			"--end": row.get("snr_end", row["gps_end_time"]),
			"--frame-cache": frame_cache,
			"--frame-segments-file": segments_xml,
			"--frame-segments-name": cp.get("data", "segments_name", fallback="datasegments"),
			"--channel-name": row_channel_names,
			"--reference-psd": row_psd,
			"--psd-fft-length": cp.get("data", "psd_fft_length", fallback="32"),
			"--svd-bank": row_svd_bank,
			"--bank-number": row["bank_number"],
			"--row-number": row["row_number"],
			"--row-counts": row.get("row_counts", "1"),
			"--outdir": calc_dir,
			"--tmp-space": os.path.join(calc_dir, "tmp"),
			"--complex": "",
			"--fir-stride": cp.get("calc_snr", "fir_stride", fallback="1"),
			"--ht-gate-threshold": row.get("ht_gate_threshold", cp.get("calc_snr", "ht_gate_threshold", fallback="")),
		}
		row_injection_xml = abs_if_set(row.get("injection_xml", "")) or global_injection_xml
		if row_injection_xml:
			calc_opts["--injections"] = row_injection_xml
		if time_slide_file:
			calc_opts["--time-slide-file"] = time_slide_file
		if veto_segments_file:
			calc_opts["--veto-segments-file"] = veto_segments_file
			calc_opts["--veto-segments-name"] = cp.get("data", "veto_segments_name", fallback="vetoes")
		if not calc_opts.get("--ht-gate-threshold"):
			calc_opts.pop("--ht-gate-threshold", None)
		if not calc_opts.get("--frame-segments-file"):
			calc_opts.pop("--frame-segments-file", None)
			calc_opts.pop("--frame-segments-name", None)

		calc_inputs = [
			frame_cache,
			segments_xml,
			row_psd,
		]
		if row_injection_xml:
			calc_inputs.append(row_injection_xml)
		if time_slide_file:
			calc_inputs.append(time_slide_file)
		if veto_segments_file:
			calc_inputs.append(veto_segments_file)

		calc_opts_list.append(calc_opts)
		calc_io_list.append({"inputs": calc_inputs, "outputs": [calc_dir]})

		this_backend = row_skymap_backend(cp, backend, row.get("instruments", ""))
		map_backend_counts[this_backend] = map_backend_counts.get(this_backend, 0) + 1
		adaptive_row = (
			this_backend == "coherent_likelihood"
			and cp.getboolean("skymap", "adaptive", fallback=False)
			and len(split_combo(row.get("instruments", ""))) > 1
		)

		map_opts = {
			"--calc-snr-dir": calc_dir,
			"--output-fits": fits,
			"--instruments": ",".join(split_combo(row.get("instruments", ""))),
			"--bank-number": row["bank_number"],
			"--row-number": row["row_number"],
			"--coinc-event-id": row.get("coinc_event_id", sim_id),
			"--center-time": row["center_time"],
			"--pre-trigger": row.get("pre_trigger", cp.get("skymap", "pre_trigger", fallback="0.12")),
			"--precalc-len": cp.get("skymap", "precalc_len", fallback="438"),
			"--sample-rate": cp.get("skymap", "sample_rate", fallback="2048"),
			"--effective-sample-rate": cp.get("skymap", "effective_sample_rate", fallback="512"),
			"--mode": cp.get("skymap", "mode", fallback="asd"),
			"--ra-deg": row.get("ra_deg", ""),
			"--dec-deg": row.get("dec_deg", ""),
			"--contour": cp.get("skymap", "contour", fallback="50 90"),
		}
		map_inputs = [calc_dir]
		map_outputs = [fits]
		if write_png:
			map_opts["--output-png"] = png
			map_outputs.append(png)
		if write_branch_maps and not adaptive_row:
			map_opts["--output-fits-p"] = fits_p
			map_opts["--output-fits-n"] = fits_n
			map_outputs.extend([fits_p, fits_n])
		if write_coefficients and not adaptive_row:
			map_opts["--output-coeff-p"] = coeff_p
			map_opts["--output-coeff-n"] = coeff_n
			map_outputs.extend([coeff_p, coeff_n])
		else:
			map_opts["--no-output-coefficients"] = ""

		if this_backend in ("direct_time_prior", "coherent_likelihood"):
			if this_backend == "direct_time_prior":
				map_opts["--direct-time-prior"] = ""
				map_opts["--direct-smoothing-deg"] = cp.get("skymap", "direct_smoothing_deg", fallback="0.0")
			else:
				map_opts["--coherent-likelihood"] = ""
				map_opts["--paper-fmin"] = cp.get("skymap", "paper_fmin", fallback="0.0")
				map_opts["--paper-fmax"] = cp.get("skymap", "paper_fmax", fallback="512.0")
				map_opts["--paper-autocorr-floor"] = cp.get("skymap", "paper_autocorr_floor", fallback="1e-6")
				map_opts["--paper-snr-window"] = cp.get("skymap", "paper_snr_window", fallback="full")
				map_opts["--amplitude-model"] = cp.get("skymap", "amplitude_model", fallback="paper_cross")
				if cp.get("skymap", "peak_time_sigma", fallback="").strip():
					map_opts["--peak-time-sigma"] = cp.get("skymap", "peak_time_sigma")
					if cp.get("skymap", "peak_time_min_snr", fallback="").strip():
						map_opts["--peak-time-min-snr"] = cp.get("skymap", "peak_time_min_snr")
					if cp.get("skymap", "peak_time_min_peak_abs", fallback="").strip():
						map_opts["--peak-time-min-peak-abs"] = cp.get("skymap", "peak_time_min_peak_abs")
					if cp.get("skymap", "peak_time_min_prominence", fallback="").strip():
						map_opts["--peak-time-min-prominence"] = cp.get("skymap", "peak_time_min_prominence")
					if cp.get("skymap", "peak_time_uncertainty_model", fallback="").strip():
						map_opts["--peak-time-uncertainty-model"] = cp.get("skymap", "peak_time_uncertainty_model")
					if cp.get("skymap", "peak_time_reference_peak_abs", fallback="").strip():
						map_opts["--peak-time-reference-peak-abs"] = cp.get("skymap", "peak_time_reference_peak_abs")
					if cp.get("skymap", "peak_time_uncertainty_power", fallback="").strip():
						map_opts["--peak-time-uncertainty-power"] = cp.get("skymap", "peak_time_uncertainty_power")
						if cp.get("skymap", "peak_time_max_sigma", fallback="").strip():
							map_opts["--peak-time-max-sigma"] = cp.get("skymap", "peak_time_max_sigma")
					map_opts["--inclination-samples"] = cp.get("skymap", "inclination_samples", fallback="5")
					map_opts["--polarization-samples"] = cp.get("skymap", "polarization_samples", fallback="4")
				if cp.getboolean("skymap", "paper_flat_noise", fallback=False):
					map_opts["--paper-flat-noise"] = ""
				if cp.getboolean("skymap", "paper_delta_template", fallback=False):
					map_opts["--paper-delta-template"] = ""
				if cp.getboolean("skymap", "paper_no_shift_autocorr", fallback=False):
					map_opts["--paper-no-shift-autocorr"] = ""
				if adaptive_row:
					map_opts["--adaptive"] = ""
					map_opts["--adaptive-nside-stop"] = cp.get("skymap", "adaptive_nside_stop", fallback="256")
					map_opts["--adaptive-refine-top-pixels"] = cp.get("skymap", "adaptive_refine_top_pixels", fallback="128")
					map_opts["--adaptive-refine-probability"] = cp.get("skymap", "adaptive_refine_probability", fallback="0.0")
					map_opts["--adaptive-max-evals"] = cp.get("skymap", "adaptive_max_evals", fallback="20000")
					if cp.getboolean("skymap", "adaptive_refine_neighbors", fallback=False):
						map_opts["--adaptive-refine-neighbors"] = ""
			map_opts["--direct-score-scale"] = cp.get("skymap", "direct_score_scale", fallback="-1")
			for option in (
				"snr_ramp_low_snr",
				"snr_ramp_high_snr",
				"snr_ramp_two_ifo_cap",
				"snr_ramp_multi_ifo_cap",
			):
				if cp.get("skymap", option, fallback="").strip():
					map_opts["--" + option.replace("_", "-")] = cp.get("skymap", option)
			map_opts["--isotropic-mixture"] = cp.get("skymap", "isotropic_mixture", fallback="0.0")
			map_opts["--antenna-mixture"] = cp.get("skymap", "antenna_mixture", fallback="0.0")
			map_opts["--direct-smoothing-deg"] = cp.get("skymap", "direct_smoothing_deg", fallback="0.0")
			if row.get("snr"):
				map_opts["--network-snr"] = row["snr"]
			if cp.getboolean("skymap", "containment_calibration", fallback=False):
				map_opts["--containment-calibration"] = ""
				for option in (
					"low_snr_threshold",
					"low_snr_score_scale",
					"low_snr_smoothing_deg",
					"low_snr_isotropic_mixture",
					"low_snr_antenna_mixture",
					"high_snr_threshold",
					"high_snr_score_scale",
					"high_snr_smoothing_deg",
					"high_snr_isotropic_mixture",
					"high_snr_antenna_mixture",
				):
					if cp.get("skymap", option, fallback="").strip():
						map_opts["--" + option.replace("_", "-")] = cp.get("skymap", option)
			map_opts["--direct-nside"] = cp.get("skymap", "direct_nside", fallback="32")
			map_opts["--time-prior-center"] = resolve_time_prior_center(cp, row)
			map_opts["--time-prior-half-width"] = resolve_time_prior_half_width(cp, row)
			if cp.get("skymap", "time_prior_step", fallback="").strip():
				map_opts["--time-prior-step"] = cp.get("skymap", "time_prior_step")
			if cp.get("skymap", "direct_coeff_lmax", fallback="").strip():
				map_opts["--direct-coeff-lmax"] = cp.get("skymap", "direct_coeff_lmax")
			if cp.get("skymap", "direct_coeff_log_floor", fallback="").strip():
				map_opts["--direct-coeff-log-floor"] = cp.get("skymap", "direct_coeff_log_floor")
			if row_psd:
				map_opts["--psd-xml"] = row_psd
				map_inputs.append(row_psd)
		else:
			if precalc_dir:
				map_opts["--precalc-dir"] = precalc_dir
				map_inputs.append(precalc_dir)
			else:
				map_opts["--psd-xml"] = row_psd
				map_inputs.append(row_psd)
			if cp.getboolean("skymap", "runtime_gmst_correction", fallback=False):
				map_opts["--runtime-projection-gmst"] = ""
				runtime_projection_mode = cp.get("skymap", "runtime_projection_mode", fallback="frame_only").strip().lower()
				if runtime_projection_mode in ("frame_only", "rotation_only"):
					map_opts["--runtime-frame-gmst-only"] = ""
				elif runtime_projection_mode != "full":
					raise ValueError("[skymap] runtime_projection_mode must be 'frame_only' or 'full'")
				if cp.get("skymap", "runtime_projection_frame_gmst_scale", fallback="").strip():
					map_opts["--runtime-projection-frame-gmst-scale"] = cp.get("skymap", "runtime_projection_frame_gmst_scale")
				if cp.get("skymap", "time_prior_half_width", fallback="").strip():
					map_opts["--time-prior-center"] = resolve_time_prior_center(cp, row)
					map_opts["--time-prior-half-width"] = resolve_time_prior_half_width(cp, row)
		if row.get("start_idx"):
			map_opts["--start-idx"] = row["start_idx"]
		if skymap_timing:
			map_opts["--timing"] = ""
		boolean_map_flags = {
			"--direct-time-prior",
			"--coherent-likelihood",
			"--paper-flat-noise",
			"--paper-delta-template",
				"--paper-no-shift-autocorr",
			"--runtime-projection-gmst",
			"--runtime-frame-gmst-only",
			"--no-output-coefficients",
			"--adaptive",
			"--adaptive-refine-neighbors",
			"--containment-calibration",
			"--timing",
		}
		map_opts = {k: v for k, v in map_opts.items() if v != "" or k in boolean_map_flags}

		map_opts_list.append(map_opts)
		map_io_list.append({"inputs": map_inputs, "outputs": map_outputs})

		if run_bayestar:
			coinc_event_id = row.get("coinc_event_id", "")
			if not coinc_event_id:
				raise ValueError(f"manifest row for simulation {sim_id} has no coinc_event_id needed by BAYESTAR")
			bayestar_opts = {
				"--input-sqlite": bayestar_database,
				"--calc-snr-dir": calc_dir,
				"--coinc-event-id": coinc_event_id,
				"--output-fits": bayestar_fits,
				"--omp-num-threads": cp.get("bayestar", "omp_num_threads", fallback="").strip() or cp.get("condor", "request_cpus", fallback="1"),
			}
			if row_psd:
				bayestar_opts["--psd-xml"] = row_psd
			if bayestar_write_png:
				bayestar_opts["--output-png"] = bayestar_png
			for config_name, option in (
				("f_low", "--f-low"),
				("f_high_truncate", "--f-high-truncate"),
				("waveform", "--waveform"),
				("min_distance", "--min-distance"),
				("max_distance", "--max-distance"),
				("prior_distance_power", "--prior-distance-power"),
				("rescale_loglikelihood", "--rescale-loglikelihood"),
				("loglevel", "--loglevel"),
				("snr_series_half_width", "--snr-series-half-width"),
			):
				if cp.get("bayestar", config_name, fallback="").strip():
					bayestar_opts[option] = cp.get("bayestar", config_name)
			if cp.getboolean("bayestar", "disable_snr_series", fallback=False):
				bayestar_opts["--disable-snr-series"] = ""
			if cp.getboolean("bayestar", "keep_going", fallback=True):
				bayestar_opts["--keep-going"] = ""
			bayestar_opts_list.append(bayestar_opts)
			bayestar_outputs = [bayestar_fits]
			if bayestar_write_png:
				bayestar_outputs.append(bayestar_png)
			bayestar_inputs = [bayestar_database, calc_dir]
			if row_psd:
				bayestar_inputs.append(row_psd)
			bayestar_io_list.append({"inputs": bayestar_inputs, "outputs": bayestar_outputs})
	log(f"prepared {len(calc_opts_list)} gstlal_inspiral_calc_snr jobs")
	log(f"prepared {len(map_opts_list)} sph_skymap_from_calc_snr jobs")
	if run_bayestar:
		log(f"prepared {len(bayestar_opts_list)} sph_bayestar_from_sqlite jobs")
	log("skymap jobs by backend: " + ", ".join(f"{name}={count}" for name, count in sorted(map_backend_counts.items())))

	log("writing DAG and submit files")
	calc_layer.batch_set_arguments(calc_opts_list, list_of_io=calc_io_list)
	map_layer.batch_set_arguments(map_opts_list, list_of_io=map_io_list)
	if run_bayestar:
		bayestar_layer.batch_set_arguments(bayestar_opts_list, list_of_io=bayestar_io_list)

	dag = baseDAG(dag_dir=dag_dir, dag_filename=cp.get("workflow", "dag_filename", fallback="injection_skymaps.dag"))
	calc_dag_layer = dag.add_layer(
		layer=calc_layer,
		category="gstlal_inspiral_calc_snr",
		retries=cp.getint("workflow", "retries", fallback=1),
	)
	dag.add_layer(
		layer=map_layer,
		parents=[calc_dag_layer],
		category="sph_skymap_from_calc_snr",
		retries=cp.getint("workflow", "retries", fallback=1),
	)
	if run_bayestar:
		dag.add_layer(
			layer=bayestar_layer,
			parents=[calc_dag_layer],
			category="sph_bayestar_from_sqlite",
			retries=cp.getint("bayestar", "retries", fallback=cp.getint("workflow", "retries", fallback=1)),
		)
	dag.write()

	print("Wrote DAG:", os.path.join(dag_dir, dag.dag_filename))
	print("Submit with:")
	print("  cd", dag_dir)
	print("  condor_submit_dag", dag.dag_filename)


if __name__ == "__main__":
	main()
