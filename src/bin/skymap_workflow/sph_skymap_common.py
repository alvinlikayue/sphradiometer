#!/usr/bin/env python3
"""Shared helpers for exploratory sphradiometer skymap workflows."""

import contextlib
import os

import numpy as np
import ligolw.utils
from lalmetaio import series as lalseries
from sphradiometer import sphradiometer as sph


class ArrayPSD:
	"""Wrapper with the .psd attribute expected by RapidLocalization."""

	def __init__(self, values):
		values = np.asarray(values, dtype=float)
		self.psd = sph.new_double_array(len(values))
		for i, value in enumerate(values):
			sph.double_array_setitem(self.psd, i, float(value))

	def __del__(self):
		if getattr(self, "psd", None) is not None:
			sph.delete_double_array(self.psd)
			self.psd = None


def load_relative_psd_weights(psd_xml, instruments, length, sample_rate, mode):
	"""Return detector weights normalized by the network median per bin."""

	doc = ligolw.utils.load_filename(psd_xml)
	raw_psds = lalseries.read_psd_xmldoc(doc)
	freqs = np.abs(np.fft.fftfreq(length, d=1.0 / sample_rate))
	values = {}

	for ifo in instruments:
		psd = raw_psds[ifo]
		psd_freq = psd.f0 + np.arange(len(psd.data.data)) * psd.deltaF
		psd_data = np.asarray(psd.data.data, dtype=float)
		good = np.isfinite(psd_data) & (psd_data > 0)
		if not np.any(good):
			raise ValueError(f"PSD for {ifo} has no positive finite samples")
		interp = np.interp(
			freqs,
			psd_freq[good],
			psd_data[good],
			left=psd_data[good][0],
			right=psd_data[good][-1],
		)
		if mode == "flat":
			interp = np.ones_like(interp)
		elif mode == "asd":
			interp = np.sqrt(interp)
		elif mode == "psd":
			pass
		else:
			raise ValueError(f"unknown PSD weighting mode: {mode}")
		values[ifo] = interp

	stack = np.vstack([values[ifo] for ifo in instruments])
	median = np.median(stack, axis=0)
	median[~np.isfinite(median) | (median <= 0)] = 1.0
	return {ifo: ArrayPSD(values[ifo] / median) for ifo in instruments}


def _quote_condor_env(value):
	return str(value).replace('"', '\\"')


def _and_requirements(*requirements):
	requirements = [item.strip() for item in requirements if item and item.strip()]
	return " && ".join(f"({item})" for item in requirements)


def _container_settings(condor_opts):
	mode = condor_opts.pop("container_mode", "auto").strip().lower()
	image = condor_opts.pop("container_image", "").strip()
	if mode in ("", "false", "off", "no"):
		mode = "none"
	if mode == "auto":
		mode = "native" if image else "none"
	if mode not in ("none", "native"):
		raise ValueError("[condor] container_mode must be one of: auto, none, native")
	if mode != "none" and not image:
		raise ValueError("[condor] container_image is required when container_mode is not none")
	return mode, image


def condor_options_from_config(cp, dag_dir, default_request_cpus="1", default_request_memory="4GB", default_request_disk="4GB"):
	"""Return htcondor_helper options using the ligo.skymap container config style."""

	condor_opts = dict(cp.items("condor")) if cp.has_section("condor") else {}
	container_mode, container_image = _container_settings(condor_opts)
	container_bind_paths = condor_opts.pop("container_bind_paths", "").strip()
	extra_environment = condor_opts.pop("extra_environment", "").strip()

	env = [
		"GST_REGISTRY_UPDATE=no",
		"HDF5_USE_FILE_LOCKING=FALSE",
		"MPLBACKEND=Agg",
		"MPLCONFIGDIR=/tmp/sphradiometer-matplotlib-$USER",
	]
	if container_mode != "none" and container_bind_paths:
		env.append(f"SINGULARITY_BINDPATH={container_bind_paths}")
		env.append(f"APPTAINER_BINDPATH={container_bind_paths}")
	if extra_environment:
		env.append(extra_environment)

	condor_opts.setdefault("universe", "vanilla")
	condor_opts.setdefault("request_cpus", default_request_cpus)
	condor_opts.setdefault("request_memory", default_request_memory)
	condor_opts.setdefault("request_disk", default_request_disk)
	condor_opts.setdefault("transfer_executable", "False")
	condor_opts.setdefault("initialdir", dag_dir)
	condor_opts["environment"] = '"' + " ".join(_quote_condor_env(item) for item in env) + '"'

	if container_mode == "native":
		condor_opts["MY.SingularityImage"] = f'"{container_image}"'
		cluster_requirement = (
			'regexp("^hpc", Machine)'
			if condor_opts.get("cluster", "").strip().lower() == "resceubbc"
			else ""
		)
		condor_opts["requirements"] = _and_requirements(
			condor_opts.get("requirements", ""),
			"HasSingularity =?= true",
			"HasSIF =?= true",
			cluster_requirement,
		)

	return condor_opts


@contextlib.contextmanager
def without_inherited_container():
	"""Prevent local generator container state from leaking into Condor submit files."""

	names = (
		"SINGULARITY_CONTAINER",
		"APPTAINER_CONTAINER",
		"SINGULARITY_BINDPATH",
		"APPTAINER_BINDPATH",
	)
	saved = {name: os.environ.pop(name, None) for name in names}
	try:
		yield
	finally:
		for name, value in saved.items():
			if value is not None:
				os.environ[name] = value
