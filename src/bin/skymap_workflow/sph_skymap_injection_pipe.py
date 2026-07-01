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
	return "".join(sorted(ifos))


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
	return [f"{ifo}={channel_map[ifo]}" for ifo in split_combo(combo)]


def discover_first_existing(run_dir, names):
	for name in names:
		path = os.path.join(run_dir, name)
		if os.path.exists(path):
			return path
	return ""


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
			combo = os.path.basename(cache).split("-")[0]
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
		combo = os.path.basename(path).split("-")[0]
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
	manifest_cp.set("manifest", "allow_missing_bank_rows", "false")
	for key in ("far_lt", "far_lte", "far_gt", "far_gte", "snr_lt", "snr_lte", "snr_gt", "snr_gte", "likelihood_lt", "likelihood_lte", "likelihood_gt", "likelihood_gte", "where", "instruments", "exclude_time_windows", "exclude_injection_associations"):
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
		manifest_rows = diagnostics.make_noise_candidate_manifest_rows(manifest_cp, rows)
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
		log("loading detector data segments")
		segments_by_ifo = load_segments_by_ifo(segments_xml)
		row_combo_by_time = {}
		gps_pad = float(manifest_cp.get("manifest", "gps_pad"))
		for row in rows:
			center = float(row["center_time"])
			ifos = ifos_covering_interval(segments_by_ifo, int(center - gps_pad), int(center + gps_pad + 0.999999))
			if len(ifos) < int(cp.get("manifest", "min_ifo", fallback="1")):
				continue
			row_combo_by_time[diagnostics._format_float(row["simulation_time"])] = combo_from_ifos(ifos)
		combo_counts = {}
		for combo in row_combo_by_time.values():
			combo_counts[combo] = combo_counts.get(combo, 0) + 1
		log("selected rows by detector combo: " + ", ".join(f"{combo}={count}" for combo, count in sorted(combo_counts.items())))
		simulation_id_by_time = {
			diagnostics._format_float(row["simulation_time"]): str(row["simulation_id"])
			for row in rows
			if diagnostics._format_float(row["simulation_time"]) in row_combo_by_time
		}
		log("discovering injection-template-match XML files")
		template_match_files = discover_template_match_files(run_dir)
		log(f"found {len(template_match_files)} injection-template-match XML files")
		log("discovering SVD bank groups")
		svd_banks_by_combo = discover_svd_bank_strings_by_combo(run_dir)
		log("found SVD bank group strings by combo: " + ", ".join(f"{combo}={len(values)}" for combo, values in sorted(svd_banks_by_combo.items())))
		log("matching selected injections to SVD bank rows by geocentric time")
		bank_map_rows = []
		for combo in sorted(set(row_combo_by_time.values())):
			combo_times = [time for time, row_combo in row_combo_by_time.items() if row_combo == combo]
			if combo not in svd_banks_by_combo:
				raise FileNotFoundError(f"no SVD bank caches found for detector combo {combo}")
			combo_rows = diagnostics.scan_template_match_svd_bank_rows(
				template_match_files,
				svd_banks_by_combo[combo],
				simulation_times=combo_times,
			)
			for row in combo_rows:
				row["instruments"] = combo
				row["channel_name"] = ",".join(channels_for_combo(channel_map, combo))
			bank_map_rows.extend(combo_rows)
		log(f"matched {len(bank_map_rows)} injections to SVD bank rows")
		for row in bank_map_rows:
			row["simulation_id"] = simulation_id_by_time[row["simulation_time"]]
		bank_map_path = os.path.join(output_root, "bank_row_map.csv")
		diagnostics.write_bank_row_map(bank_map_rows, bank_map_path)
		log(f"wrote bank-row map: {bank_map_path}")
		manifest_cp.set("manifest", "bank_row_map", bank_map_path)
		rows = [
			row for row in rows
			if diagnostics._format_float(row["simulation_time"]) in simulation_id_by_time
		]
		manifest_rows = diagnostics.make_injection_manifest_rows(manifest_cp, rows)
		log("discovering reference PSD files")
		psds_by_combo = discover_reference_psds_by_combo(run_dir)
		log("found reference PSD files by combo: " + ", ".join(f"{combo}={len(values)}" for combo, values in sorted(psds_by_combo.items())))
		log("assigning reference PSD to each manifest row")
		for row in manifest_rows:
			combo = row["instruments"]
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
	skymap_branch_root = os.path.join(output_root, "skymap_branch")
	coeff_root = os.path.join(output_root, "coeff")
	log(f"reading config: {os.path.abspath(args.config)}")
	log(f"DAG directory: {dag_dir}")

	os.makedirs(dag_dir, exist_ok=True)
	os.makedirs(os.path.join(dag_dir, "logs"), exist_ok=True)
	os.makedirs(output_root, exist_ok=True)
	os.makedirs(calc_snr_root, exist_ok=True)
	os.makedirs(skymap_root, exist_ok=True)
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

	channel_names = split_channels(cp.get("data", "channel_name", fallback=discovered.get("channel_name", "")))
	if not channel_names:
		raise ValueError("channel names were not configured or discovered")
	log(f"using {len(channel_names)} channels: {', '.join(channel_names)}")
	svd_banks = cp.get("data", "svd_bank", fallback="")
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

		map_opts = {
			"--calc-snr-dir": calc_dir,
			"--output-fits": fits,
			"--output-fits-p": fits_p,
			"--output-fits-n": fits_n,
			"--output-coeff-p": coeff_p,
			"--output-coeff-n": coeff_n,
			"--output-png": png,
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
		if precalc_dir:
			map_opts["--precalc-dir"] = precalc_dir
		else:
			map_opts["--psd-xml"] = row_psd
		if row.get("start_idx"):
			map_opts["--start-idx"] = row["start_idx"]
		map_opts = {k: v for k, v in map_opts.items() if v != ""}

		map_inputs = [calc_dir]
		if precalc_dir:
			map_inputs.append(precalc_dir)
		else:
			map_inputs.append(row_psd)
		map_opts_list.append(map_opts)
		map_io_list.append({"inputs": map_inputs, "outputs": [fits, fits_p, fits_n, coeff_p, coeff_n, png]})
	log(f"prepared {len(calc_opts_list)} gstlal_inspiral_calc_snr jobs")
	log(f"prepared {len(map_opts_list)} sph_skymap_from_calc_snr jobs")

	log("writing DAG and submit files")
	calc_layer.batch_set_arguments(calc_opts_list, list_of_io=calc_io_list)
	map_layer.batch_set_arguments(map_opts_list, list_of_io=map_io_list)

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
	dag.write()

	print("Wrote DAG:", os.path.join(dag_dir, dag.dag_filename))
	print("Submit with:")
	print("  cd", dag_dir)
	print("  condor_submit_dag", dag.dag_filename)


if __name__ == "__main__":
	main()
