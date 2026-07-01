from glob import glob
from itertools import combinations
import numpy as np
import os
import sys
import tempfile
import warnings

from lal import series as lalseries
import ligolw
try:
	from ligo.skymap import io as skymap_io
except ImportError:
	warnings.warn("ligo.skymap.io import failed (not installed?).  writing to BAYESTAR compatible FITS files feature will be disabled.")

from . import sphradiometer as sph
from . import healpix as sph_healpix


#
# ==============================================================================
#
#                           Masked Memory Operation
#
# ==============================================================================
#


class coeff_series(object):
	def __init__(self, ndet):
		"""
		Parameter
		---------
		ndet : int
			the number of observing detectors
		"""
		self.ndet = ndet
		if ndet > 1:
			self.coeff = sph.new_sh_seriespp()
		else:
			self.coeff = sph.new_sh_seriesp()

	def get(self):
		if self.ndet > 1:
			return sph.sh_seriespp_value(self.coeff)
		else:
			return self.coeff

	def __del__(self):
		if self.ndet > 1:
			sph.sh_series_free(sph.sh_seriespp_value(self.coeff))
			sph.delete_sh_seriespp(self.coeff)
		else:
			sph.delete_sh_seriesp(self.coeff)


#
# ==============================================================================
#
#                                    PSDs
#
# ==============================================================================
#


class flat_psd(object):
	def __init__(self, length):
		"""
		Parameter
		---------
		length : int
			psd length
		"""
		self.psd = sph.new_double_array(length)
		for i in range(length):
			sph.double_array_setitem(self.psd, i, 1)

	def __del__(self):
		sph.delete_double_array(self.psd)


def transpose_psd_array(psds, length):
	"""
	Parameter
	---------
	psds : dict of e.g. flat_psd instances
		key is detector prefix
	length : int
		psd length
	"""
	instruments = sorted(psds)

	psds_array = sph.new_doublep_array(len(psds))
	for i, ifo in enumerate(instruments):
		sph.doublep_array_setitem(psds_array, i, psds[ifo].psd)
	tpsds_array = sph.transpose_matrix(psds_array, len(psds), length)
	sph.delete_doublep_array(psds_array)

	return tpsds_array


#
# ==============================================================================
#
#                                   Time Series
#
# ==============================================================================
#


class create_sph_COMPLEX16TimeSeries_wrap(object):
	def __init__(self, tseries):
		"""
		Parameter
		---------
		tseries : <Swig Object of type 'COMPLEX8TimeSeries *'>
		"""
		data = tseries.data.data
		data_ = sph.new_double_complex_array(len(data))
		for i, x in enumerate(data):
			sph.double_complex_array_setitem(data_, i, complex(x))

		self.series = sph.new_COMPLEX16TimeSeriesp()
		sph.create_sph_COMPLEX16TimeSeries(
			self.series,
			tseries.name,
			tseries.epoch.gpsSeconds,
			tseries.epoch.gpsNanoSeconds,
			tseries.f0,
			tseries.deltaT,
			None,
			len(data),
			data_
		)

		sph.delete_double_complex_array(data_)

	def __del__(self):
		sph.free_SNRTimeSeries(self.series)


class create_sph_COMPLEX16Sequence_wrap(object):
	def __init__(self, tseries):
		"""
		Parameter
		---------
		tseries : <Swig Object of type 'COMPLEX8Vector *'>
		"""
		data = tseries.data
		data_ = sph.new_double_complex_array(len(data))
		for i, x in enumerate(data):
			sph.double_complex_array_setitem(data_, i, complex(x))

		self.series = sph.new_COMPLEX16Sequencep()
		sph.create_sph_COMPLEX16Sequence(
			self.series,
			len(data),
			data_
		)

		sph.delete_double_complex_array(data_)

	def __del__(self):
		sph.free_SNRSequence(self.series)


class convert_TimeSeries2Sequence_wrap(object):
	def __init__(self, series):
		"""
		Parameter
		---------
		series : create_sph_COMPLEX16TimeSeries_wrap instances
		"""
		self.series = sph.convert_TimeSeries2Sequence(series.series)

	def __del__(self):
		sph.free_SNRSequence(self.series)


#
# ==============================================================================
#
#                                 Localization
#
# ==============================================================================
#


class RapidLocalization_(object):
	def __init__(self, psds, precalc_length, deltaT, effective_sample_rate=512):
		"""
		Parameter
		---------
		psds : dict of e.g. flat_psd instances
			psds for all detectors
		precalc_length : int
			legth of SNR time series used for localization.
		deltaT : float
			bin width of SNR time series used for localization.
		effective_sample_rate : int
			effective frquency to calculate spherical index of the
			precalculated objects

		NOTE
		----
		If the precalc_length in reading-process of the precalculated
		objects is NOT equal to that in writing-process, then the
		precalculated objects have to be generated again.
		"""
		# preamble
		self.precalc_length = precalc_length
		self.instruments = sorted(psds)
		self.inst_array = sph.instrument_array_new(0)
		for ifo in self.instruments:
			sph.instrument_array_append(self.inst_array, sph.instrument_new_from_name(ifo))
		psd_array = transpose_psd_array(psds, self.precalc_length)

		# prepare precalculated objects
		if len(self.instruments) > 1:
			# multi detector case
			self.baselines = sph.correlator_network_baselines_new(self.inst_array)
			self.logprior = sph.sh_series_log_uniformsky_prior(sph.correlator_network_l_max(self.baselines, deltaT))

			self.fdplansp = sph.correlator_network_plan_fd_new(self.baselines, self.precalc_length, deltaT)
			self.fdplansn = sph.correlator_network_plan_fd_copy(self.fdplansp)
			sph.correlator_network_plan_mult_by_projection(self.fdplansp, +1, 0, psd_array)
			sph.correlator_network_plan_mult_by_projection(self.fdplansn, -1, 0, psd_array)

			self.fdautoplanp = sph.autocorrelator_network_plan_fd_new(self.inst_array, +1, 0, psd_array, int(sph.pick_ith_correlator_plan_fd(self.fdplansp.plans, 0).delay_product.n), self.logprior.l_max)
			self.fdautoplann = sph.autocorrelator_network_plan_fd_new(self.inst_array, -1, 0, psd_array, int(sph.pick_ith_correlator_plan_fd(self.fdplansp.plans, 0).delay_product.n), self.logprior.l_max)

		else:
			# single detector case
			self.logprior = sph.sh_series_log_uniformsky_prior(sph.correlator_baseline_power_l_max_naive(self.precalc_length * deltaT, deltaT))

		# free
		for i in range(self.precalc_length):
			sph.delete_double_array(sph.doublep_array_getitem(psd_array, i))
		sph.delete_doublep_array(psd_array)

		# reduce the upper cutoff of spheical index l
		# Up to 512 Hz, SNR of the CBC waveforms are > 99% accumulated.
		self.reduce_l_max(1. / effective_sample_rate)


	def sphcoeff(self, snr, aut, power=0.6, overall=2.7):
		"""
		Parameter
		---------
		snr : dict of <Swig Object of type 'COMPLEX16TimeSeries *'>
			stuff of observed SNR time series for all detectors
		aut : dict of <Swig Object of type 'COMPLEX16TimeSeries *'>
			stuff of template auto-correlations on time domain for
			all detectors
		Returns
		-------
		output : tuple of coeff_series instances
			object to store a localization result for \\beta = +/-1
		"""
		# construct **COMPLEX16_array
		sph_snr = {ifo: create_sph_COMPLEX16TimeSeries_wrap(snr[ifo]) for ifo in snr}
		sph_aut = {ifo: create_sph_COMPLEX16Sequence_wrap(aut[ifo]) for ifo in aut}
		seriesp = sph.new_COMPLEX16TimeSeries_array(len(snr))
		nseriesp = sph.new_COMPLEX16Sequence_array(len(aut))
		for i, ifo in enumerate(self.instruments):
			sph.COMPLEX16TimeSeries_array_setitem(seriesp, i, sph_snr[ifo].series)
			sph.COMPLEX16Sequence_array_setitem(nseriesp, i, sph_aut[ifo].series)
		sph.preprocess_SNRTimeSeries(seriesp, nseriesp, len(self.instruments))

		# consistency check
		assert self.instruments == sorted(snr.keys()), "instruments are inconsistent"
		if len(snr) > 1:
			assert self.precalc_length == sph.pick_length_from_COMPLEX16TimeSeries(sph.COMPLEX16TimeSeries_array_getitem(seriesp, 0)), "SNR time series length is inconsistent"

		# store sph coefficient series result
		skyp = coeff_series(len(self.instruments))
		skyn = coeff_series(len(self.instruments))
		if sph.instrument_array_len(self.inst_array) > 1:
			# multi detector case
			sph.generate_alm_skys(skyp.coeff, skyn.coeff, self.fdplansp, self.fdplansn, self.fdautoplanp, self.fdautoplann, seriesp, nseriesp, self.logprior)

			# weight sph coefficient series with overall * (l /
			# l_max)^power
			# For (power, overall) = (0.6, 2.7), the p-p plot is
			# consistent for 512 Hz of the upper frequency cutoff
			sph.sh_seriespp_assign(skyp.coeff, sph.sh_series_scale_power_l(skyp.get(), power, overall))
			sph.sh_seriespp_assign(skyn.coeff, sph.sh_series_scale_power_l(skyn.get(), power, overall))
		else:
			# single detector case
			sph.sh_seriesp_assign(skyp.coeff, self.logprior)
			sph.sh_seriesp_assign(skyn.coeff, self.logprior)

		# free
		del sph_snr, sph_aut
		sph.delete_COMPLEX16TimeSeries_array(seriesp)
		sph.delete_COMPLEX16Sequence_array(nseriesp)

		return skyp, skyn

	def write(self, precalc_path):
		"""
		Parameter
		---------
		precalc_path : str
			path (name) of used precalculated objects.
		"""
		sph.make_precalc_directories(precalc_path, sph.instrument_array_len(self.inst_array))
		np.save(os.path.join(precalc_path, "instruments.npy"), self.instruments)
		sph.write_precalc_time_series_length(self.precalc_length, precalc_path)
		if sph.instrument_array_len(self.inst_array) > 1:
			# (n > 1) detector case
			sph.write_precalc_logprior(self.logprior, precalc_path)
			sph.write_precalc_correlator_network_plan_fd(self.fdplansp, self.fdplansn, precalc_path)
			sph.write_precalc_autocorrelator_network_plan_fd(self.fdautoplanp, self.fdautoplann, precalc_path)
		else:
			# 1 detector case
			sph.write_precalc_logprior(self.logprior, precalc_path)

	@classmethod
	def read(cls, precalc_path):
		"""
		Parameter
		---------
		precalc_path : str
			path (name) of used precalculated objects.
		"""
		self = cls.__new__(cls)

		# preamble
		self.instruments = list(np.load(os.path.join(precalc_path, "instruments.npy")))
		self.inst_array = sph.instrument_array_new(0)
		for ifo in self.instruments:
			sph.instrument_array_append(self.inst_array, sph.instrument_new_from_name(ifo))

		self.logprior = sph.read_precalc_logprior(precalc_path)
		if sph.instrument_array_len(self.inst_array) > 1:
			# (n > 1) detector case
			length = sph.new_uintp()
			sph.read_precalc_time_series_length(length, precalc_path)
			self.precalc_length = sph.uintp_value(length)
			sph.delete_uintp(length)

			self.baselines = sph.read_precalc_correlator_network_baselines(self.inst_array, precalc_path)
			self.fdplansp = sph.new_correlator_network_plan_fdp()
			self.fdplansn = sph.new_correlator_network_plan_fdp()
			sph.read_precalc_correlator_network_plan_fd(self.fdplansp, self.fdplansn, self.baselines, self.precalc_length, precalc_path)

			self.fdautoplanp = sph.new_autocorrelator_network_plan_fdp()
			self.fdautoplann = sph.new_autocorrelator_network_plan_fdp()
			sph.read_precalc_autocorrelator_network_plan_fd(self.fdautoplanp, self.fdautoplann, self.inst_array, self.precalc_length, precalc_path)

		return self

	def reduce_l_max(self, deltaT):
		"""
		Parameter
		---------
		deltaT : double
			effective down-sampled time bin width.  The maximum of
			l is determined from the effective sample frequency.
		"""
		if len(self.instruments) > 1:
			l_max = sph.correlator_network_l_max(self.baselines, deltaT)
			self.logprior = sph.sh_series_resize(self.logprior, l_max)
			self.fdplansp = sph.correlator_network_plan_fd_set_l(self.fdplansp, l_max)
			self.fdplansn = sph.correlator_network_plan_fd_set_l(self.fdplansn, l_max)
			self.fdautoplanp = sph.autocorrelator_network_plan_fd_set_l(self.fdautoplanp, l_max)
			self.fdautoplann = sph.autocorrelator_network_plan_fd_set_l(self.fdautoplann, l_max)
		else:
			self.logprior = sph.sh_series_resize(self.logprior, 4)

	def __del__(self):
		sph.sh_series_free(self.logprior)
		if sph.instrument_array_len(self.inst_array) > 1:
			sph.correlator_network_plan_fd_free(self.fdplansp)
			sph.correlator_network_plan_fd_free(self.fdplansn)
			sph.autocorrelator_network_plan_fd_free(self.fdautoplanp)
			sph.autocorrelator_network_plan_fd_free(self.fdautoplann)
			sph.correlator_network_baselines_free(self.baselines)
		sph.instrument_array_free(self.inst_array)


class RapidLocalization(object):
	def __init__(self, psds, precalc_length, deltaT, effective_sample_rate=512):
		"""
		Parameter
		---------
		psds : dict of e.g. flat_psd instances
			psds for all detectors
		precalc_length : int
			legth of SNR time series used for localization.
		deltaT : float
			bin width of SNR time series used for localization.
		effective_sample_rate : int
			effective frquency to calculate spherical index of the
			precalculated objects

		NOTE
		----
		If the precalc_length in reading-process of the precalculated
		objects is NOT equal to that in writing-process, then the
		precalculated objects have to be generated again.
		"""
		self.instruments = sorted(psds)
		self.precalcs = {}
		for n in range(1, len(self.instruments) + 1):
			for c in combinations(self.instruments, n):
				print("initialize for", "".join(c), file=sys.stderr)
				psds_ = {ifo : psds[ifo] for ifo in c}
				self.precalcs["".join(c)] = RapidLocalization_(
					psds_,
					precalc_length,
					deltaT,
					effective_sample_rate
				)

	def sphcoeff(self, snr, aut, power=0.6, overall=2.7):
		"""
		Parameter
		---------
		snr : dict of <Swig Object of type 'COMPLEX16TimeSeries *'>
			stuff of observed SNR time series for all detectors
		aut : dict of <Swig Object of type 'COMPLEX16TimeSeries *'>
			stuff of template auto-correlations on time domain for
			all detectors
		Returns
		-------
		output : tuple of coeff_series instances
			object to store a localization result for \\beta = +/-1
		"""
		return self.precalcs["".join(sorted(snr))].sphcoeff(snr, aut, power=power, overall=overall)

	def write(self, precalc_path):
		"""
		Parameter
		---------
		precalc_path : str
			path (name) of used precalculated objects.
		"""
		os.mkdir(precalc_path)
		for n in range(1, len(self.instruments) + 1):
			for c in combinations(self.instruments, n):
				name = "".join(c)
				print("write for", name, file=sys.stderr)
				self.precalcs[name].write(os.path.join(precalc_path, name))

	@classmethod
	def read(cls, precalc_path):
		"""
		Parameter
		---------
		precalc_path : str
			path (name) of used precalculated objects.
		"""
		self = cls.__new__(cls)

		self.precalcs = {}
		for prei in [os.path.basename(d) for d in glob(os.path.join(precalc_path, "*"))]:
			print("read for", prei, file=sys.stderr)
			self.precalcs[prei] = RapidLocalization_.read(os.path.join(precalc_path, prei))
		return self

	def reduce_l_max(self, deltaT):
		"""
		Parameter
		---------
		deltaT : double
			effective down-sampled time bin width.  The maximum of
			l is determined from the effective sample frequency.
		"""
		for name in self.precalcs:
			print("reduce l_max of", name, file=sys.stderr)
			self.precalcs[name].reduce_l_max(deltaT)


#
# ==============================================================================
#
#                              Skymap I/O Helper
#
# ==============================================================================
#


class rapidskyloc_io(object):
	def __init__(self, series_p, series_n, coinc_event_id = None):
		"""
		Parameter
		---------
		series_p : coeff_series instances
		series_n : coeff_series instances
		"""
		# an integer
		self.coinc_event_id = coinc_event_id
		# an sh_series object
		series_p = series_p.get()
		series_n = series_n.get()
		# marginalize in the pixel domain.  compute healpix pixel
		# basis representation of the posterior probability
		# density.  the LVK convention is to store pixel
		# probability mass.  the rapidskyloc code computes the
		# spherical harmonic basis representation of the log of the
		# probability density up to an unknown constant offset (the
		# result is not normalized).  the conversion sequence is:
		# (i) convert to pixel domain to obtain un-normalized log
		# probability density in pixel basis; (ii) make the
		# approximation that the probability density within each
		# pixel is constant and equal to the value at the pixel
		# centre and proceed with normalizing; (iii) healpix pixel
		# maps have equal-area pixels, the area of each pixel is 4
		# pi / (# pixels) rad^2, but because this is a constant we
		# can absorb it into the yet-to-be-computed overall
		# normalization required to make the total probability = 1;
		# (iv) we want to normalize the logarithms directly because
		# until the map is normalized exponentiating it is a
		# numerically risky procedure but we cannot use logsumexp()
		# thanks to the comically ancient scipy version on
		# CalTech's LIGO Laboratory cluster, therefore instead we
		# adjust the logarithms so that the mode is 1, which should
		# be sufficient to prevent numerical problems in the next
		# step;  (v) exponentiate to obtain the not-yet-normalized
		# probability mass for each pixel;  (vi) apply a final
		# normalization to make the sum of the probabilities 1.
		_, _, log_prob_p = sph_healpix.healpy_alm_to_map(*sph_healpix.sh_series_to_healpy_alm(series_p))
		_, _, log_prob_n = sph_healpix.healpy_alm_to_map(*sph_healpix.sh_series_to_healpy_alm(series_n))

		# The detected-CBC posterior marginalizes over beta = +/-1 by
		# summing the two branch likelihoods, then normalizing the final
		# map.  Normalizing the two branches independently before summing
		# erases their relative evidence and can over-weight an antipodal
		# branch.
		log_prob_max = max(log_prob_p.max(), log_prob_n.max())
		self.prob = 0.5 * np.exp(log_prob_p - log_prob_max) + 0.5 * np.exp(log_prob_n - log_prob_max)
		self.prob /= sum(self.prob)

	def to_xml(self, name):
		"""
		Serialize the rapidskyloc_io object to LIGO Light-Weight
		XML format.  The return value is the root element of the
		XML tree containing the serialization.  The name is
		recorded with the object, and can be used at a later time
		to search for and retrieve the object from a document.

		NOTE:  unlike the healpix FITS encodings, complex-valued
		functions on the sphere are fully supported by this
		encoding.
		"""
		elem = ligolw.LIGO_LW()
		elem.Name = "%s:rapidskyloc_map" % name
		elem.appendChild(ligolw.Param.build("coinc_event:coinc_event_id", "int_8s", self.coinc_event_id, comment = "corresponding coinc_event Table entry"))
		elem.appendChild(ligolw.Array.build("prob", self.prob))
		return elem

	@classmethod
	def get_xml_root(cls, xml, name):
		return ligolw.LIGO_LW.get_ligo_lw(xml, "%s:rapidskyloc_map" % name)

	@classmethod
	def from_xml(cls, xml, name):
		"""
		Search for the serialization of a rapidskyloc_io object
		whose name is name, within the LIGO Light-Weight XML tree
		rooted at xml.  Raises ValueError if not exactly 1 such
		serialized object exists.  Returns the newly reconstructed
		rapidskyloc_io object.
		"""
		self = cls.__new__(cls)

		xml = self.get_xml_root(xml, name)
		coinc_event_id = ligolw.Param.get_param(xml, "coinc_event:coinc_event_id").value
		prob = ligolw.Array.get_array(xml, "prob").array
		self.prob = prob
		self.coinc_event_id = coinc_event_id
		return self

	def to_fits_buffer(self, fmt = "map"):
		"""
		Return a buffer of the bytes of a FITS file containing an
		encoding of the internal probability data.

		fmt is one of:
		"map" (the default):
			Write probability density in healpix map (pixel domain)
			format to FITS file.  This requires a frequency domain
			to pixel domain conversion, so this can be
			substantially slower.
		"bayestar":
			Write BAYESTAR compatible normalized pixel
			probability mass in healpix map (pixel domain)
			format to FITS file.  This feature requires
			ligo-skymap to be installed.
		"""
		# this is stupid.  there doesn't seem to be a way to create
		# a fits file in ram.  the only way to get a buffer
		# containing the bytes of a fits file is to write it to a
		# named file on disk, then read it back in and delete the
		# file.  sigh.  oh, and fits barfs if the file exists.  it
		# won't overwrite a file, so you can't create a standing
		# temporary file that you write to repeatedly if you need
		# to do this more than once.  you have to delete it and
		# make a fresh one each time.  omg.
		with tempfile.TemporaryDirectory() as tmpdir:
			filename = os.path.join(tmpdir, "deleteme.fits")
			if fmt == "bayestar":
				skymap_io.write_sky_map(
					filename,
					self.prob
				)
			elif fmt == "map":
				sph_healpix.write_fits_healpy_map(filename, self.prob)
			else:
				raise ValueError("fmt must be one of \"map\", \"bayestar\"")
			return open(filename, mode = "rb").read()


#
# ==============================================================================
#
#                                    Main
#
# ==============================================================================
#


if __name__ == "__main__":
	#
	# prepare data
	#

	print("preamble")
	coinc_file = "coinc.xml"
	precalc_path = "precalcs"
	print("assigned precalculated object is", precalc_path)

	snr = {}
	aut = {}
	ligolw.set_verbose(True)
	xmldoc = ligolw.utils.load_filename(coinc_file)
	sngl_inspiral_index = dict((row.event_id, row) for row in ligolw.lsctables.SnglInspiralTable.get_table(xmldoc))
	for elem in ligolw.LIGO_LW.get_ligo_lw(xmldoc, "COMPLEX8TimeSeries", unique = False):
		sngl_inspiral = sngl_inspiral_index[int(ligolw.Param.get_param(elem, "event_id").value)]
		snr[sngl_inspiral.ifo] = lalseries.parse_COMPLEX8TimeSeries(elem)
		# Currently nseriesp is dummy information, so that seriesp is set
		aut[sngl_inspiral.ifo] = snr[sngl_inspiral.ifo].data
	instruments = sorted(snr.keys())

	# Currently psd is dummy information, so that it's set one
	precalc_length = sph.precalculated_TimeSeries_length(
		snr[instruments[0]].data.length,
		snr[instruments[0]].deltaT
	)
	psds = {}
	for ifo in instruments:
		psds[ifo] = flat_psd(precalc_length)


	#
	# Localize
	#


	print("prepare precalculated objects")
	if os.path.exists(precalc_path):
		print("read objects")
		rapidloc = RapidLocalization.read(precalc_path)
	else:
		print("make objects")
		rapidloc = RapidLocalization(
			psds,
			precalc_length,
			snr[instruments[0]].deltaT
		)
		print("write objects")
		rapidloc.write(precalc_path)

	# sph coeff series for \beta = +1 & -1
	print("calc")
	skyp, skyn = rapidloc.sphcoeff(snr, aut)

	# save coeff series
	print("save")
	open("map.fits", "wb").write(rapidskyloc_io(skyp, skyn).to_fits_buffer(fmt = "map"))


	#
	# free
	#


	print("free")
	del psds
	del rapidloc
	del skyp
	del skyn
