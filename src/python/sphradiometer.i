/*
 * Copyright (C) 2006  Kipp C. Cannon
 * Copyright (C) 2020  Takuya Tsutsui
 *
 * This program is free software; you can redistribute it and/or modify it
 * under the terms of the GNU General Public License as published by the
 * Free Software Foundation; either version 2 of the License, or (at your
 * option) any later version.
 *
 * This program is distributed in the hope that it will be useful, but
 * WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General
 * Public License for more details.
 *
 * You should have received a copy of the GNU General Public License along
 * with this program; if not, write to the Free Software Foundation, Inc.,
 * 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301, USA.
 */

%module sphradiometer
%include carrays.i
%include ccomplex.i
%include cdata.i
%include cpointer.i
%include typemaps.i

/*%include <lal/SWIGCommon.i>
#ifndef SWIGIMPORTED
%import <lal/swiglal.i>
#endif*/

%{
#define SWIG_FILE_WITH_INIT
#include <float.h>
#include <lal/Date.h>
#include <sphradiometer/instrument.h>
#include <sphradiometer/sh_series.h>
#include <sphradiometer/inject.h>
#include <sphradiometer/projection.h>
#include <sphradiometer/correlator.h>
#include <sphradiometer/sky.h>
#include <sphradiometer/deconvolution.h>
#include <sphradiometer/inspiral_sky_map.h>
%}

%native(direct_paper_logprob_for_radec_c) _wrap_direct_paper_logprob_for_radec_c;
%{
static int sph_get_ro_buffer(PyObject *obj, Py_buffer *view, const char *name)
{
	if(PyObject_GetBuffer(obj, view, PyBUF_CONTIG_RO) < 0) {
		PyErr_Format(PyExc_TypeError, "%s must provide a contiguous buffer", name);
		return -1;
	}
	return 0;
}

static int sph_get_rw_buffer(PyObject *obj, Py_buffer *view, const char *name)
{
	if(PyObject_GetBuffer(obj, view, PyBUF_CONTIG) < 0) {
		PyErr_Format(PyExc_TypeError, "%s must provide a writable contiguous buffer", name);
		return -1;
	}
	if(view->readonly) {
		PyBuffer_Release(view);
		PyErr_Format(PyExc_TypeError, "%s must be writable", name);
		return -1;
	}
	return 0;
}

static double sph_logaddexp(double a, double b)
{
	if(a == -INFINITY)
		return b;
	if(b == -INFINITY)
		return a;
	return a > b ? a + log1p(exp(b - a)) : b + log1p(exp(a - b));
}

static PyObject *_wrap_direct_paper_logprob_for_radec_c(PyObject *self, PyObject *args)
{
	PyObject *locations_obj, *responses_obj, *freqs_obj, *times_obj, *epochs_obj;
	PyObject *data_obj, *den_weight_obj, *ra_obj, *dec_obj, *out_p_obj, *out_n_obj;
	Py_buffer locations = {0}, responses = {0}, freqs = {0}, times = {0}, epochs = {0};
	Py_buffer data = {0}, den_weight = {0}, ra = {0}, dec = {0}, out_p = {0}, out_n = {0};
	int nifo, nfreq, ntime, npix, paper_conj_response, amplitude_model, inclination_samples, polarization_samples;
	double phase_sign, score_scale;
	double *sin_gmsts = NULL, *cos_gmsts = NULL;
	int *run_starts = NULL, *run_stops = NULL;
	int nruns = 0;
	const double c_si = 299792458.0;
	const double two_pi = 2.0 * M_PI;

	(void) self;
	if(!PyArg_ParseTuple(args, "OOOOOOOOOddiiiiOO",
		&locations_obj,
		&responses_obj,
		&freqs_obj,
		&times_obj,
		&epochs_obj,
		&data_obj,
		&den_weight_obj,
		&ra_obj,
		&dec_obj,
			&phase_sign,
			&score_scale,
			&paper_conj_response,
			&amplitude_model,
			&inclination_samples,
			&polarization_samples,
			&out_p_obj,
			&out_n_obj))
		return NULL;

	if(sph_get_ro_buffer(locations_obj, &locations, "locations") ||
	   sph_get_ro_buffer(responses_obj, &responses, "responses") ||
	   sph_get_ro_buffer(freqs_obj, &freqs, "freqs") ||
	   sph_get_ro_buffer(times_obj, &times, "times_geo") ||
	   sph_get_ro_buffer(epochs_obj, &epochs, "epochs") ||
	   sph_get_ro_buffer(data_obj, &data, "data") ||
	   sph_get_ro_buffer(den_weight_obj, &den_weight, "den_weight") ||
	   sph_get_ro_buffer(ra_obj, &ra, "ra") ||
	   sph_get_ro_buffer(dec_obj, &dec, "dec") ||
	   sph_get_rw_buffer(out_p_obj, &out_p, "out_p") ||
	   sph_get_rw_buffer(out_n_obj, &out_n, "out_n"))
		goto error;

	if(locations.len % (3 * (Py_ssize_t) sizeof(double))) {
		PyErr_SetString(PyExc_ValueError, "locations length is not a multiple of 3 float64 values");
		goto error;
	}
	nifo = (int) (locations.len / (3 * (Py_ssize_t) sizeof(double)));
	if(!nifo) {
		PyErr_SetString(PyExc_ValueError, "empty detector location array");
		goto error;
	}
	if(responses.len != (Py_ssize_t) nifo * 9 * (Py_ssize_t) sizeof(double) ||
	   epochs.len != (Py_ssize_t) nifo * (Py_ssize_t) sizeof(double) ||
	   den_weight.len != (Py_ssize_t) nifo * (Py_ssize_t) sizeof(double)) {
		PyErr_SetString(PyExc_ValueError, "detector array lengths are inconsistent");
		goto error;
	}
	if(freqs.len % (Py_ssize_t) sizeof(double) || times.len % (Py_ssize_t) sizeof(double)) {
		PyErr_SetString(PyExc_ValueError, "frequency/time arrays must contain float64 values");
		goto error;
	}
	nfreq = (int) (freqs.len / (Py_ssize_t) sizeof(double));
	ntime = (int) (times.len / (Py_ssize_t) sizeof(double));
	if(!nfreq || !ntime) {
		PyErr_SetString(PyExc_ValueError, "empty frequency or time grid");
		goto error;
	}
	if(data.len != (Py_ssize_t) nifo * (Py_ssize_t) nfreq * (Py_ssize_t) sizeof(double complex)) {
		PyErr_SetString(PyExc_ValueError, "data must have nifo * nfreq complex128 values");
		goto error;
	}
	if(ra.len != dec.len || ra.len != out_p.len || ra.len != out_n.len || ra.len % (Py_ssize_t) sizeof(double)) {
		PyErr_SetString(PyExc_ValueError, "ra, dec, out_p, and out_n must be same-length float64 arrays");
		goto error;
	}
	if(amplitude_model < 0 || amplitude_model > 3) {
		PyErr_SetString(PyExc_ValueError, "amplitude_model must be 0 (circular), 1 (two_pol), 2 (cbc), or 3 (paper_cross)");
		goto error;
	}
	if(inclination_samples < 1 || inclination_samples > 33) {
		PyErr_SetString(PyExc_ValueError, "inclination_samples must be in [1, 33]");
		goto error;
	}
	if(polarization_samples < 1 || polarization_samples > 16) {
		PyErr_SetString(PyExc_ValueError, "polarization_samples must be in [1, 16]");
		goto error;
	}
	npix = (int) (ra.len / (Py_ssize_t) sizeof(double));
	sin_gmsts = malloc((size_t) ntime * sizeof(*sin_gmsts));
	cos_gmsts = malloc((size_t) ntime * sizeof(*cos_gmsts));
	if(!sin_gmsts || !cos_gmsts) {
		PyErr_NoMemory();
		goto error;
	}
	{
		const double *time_values = (const double *) times.buf;
		for(int it = 0; it < ntime; it++) {
			LIGOTimeGPS gps;
			XLALGPSSetREAL8(&gps, time_values[it]);
			const double gmst = XLALGreenwichMeanSiderealTime(&gps);
			sin_gmsts[it] = sin(gmst);
			cos_gmsts[it] = cos(gmst);
		}
	}
	run_starts = malloc((size_t) nfreq * sizeof(*run_starts));
	run_stops = malloc((size_t) nfreq * sizeof(*run_stops));
	if(!run_starts || !run_stops) {
		PyErr_NoMemory();
		goto error;
	}
	{
		const double *freq_values = (const double *) freqs.buf;
		int start = 0;
		while(start < nfreq) {
			int stop = start + 1;
			if(stop < nfreq) {
				const double df = freq_values[stop] - freq_values[start];
				const double scale = fmax(1.0, fabs(df));
				while(stop + 1 < nfreq && fabs((freq_values[stop + 1] - freq_values[stop]) - df) <= 1e-10 * scale)
					stop++;
			}
			run_starts[nruns] = start;
			run_stops[nruns] = stop + 1;
			nruns++;
			start = stop + 1;
		}
	}

	{
	const double *loc = (const double *) locations.buf;
	const double *resp = (const double *) responses.buf;
	const double *freq = (const double *) freqs.buf;
	const double *time = (const double *) times.buf;
	const double *epoch = (const double *) epochs.buf;
	const double complex *prepared = (const double complex *) data.buf;
	const double *denw = (const double *) den_weight.buf;
	const double *ras = (const double *) ra.buf;
	const double *decs = (const double *) dec.buf;
	double *logp = (double *) out_p.buf;
	double *logn = (double *) out_n.buf;

		Py_BEGIN_ALLOW_THREADS
		#pragma omp parallel for schedule(static)
		for(int pix = 0; pix < npix; pix++) {
		const double ra_pix = ras[pix];
		const double dec_pix = decs[pix];
		const double cos_ra = cos(ra_pix);
		const double sin_ra = sin(ra_pix);
		const double cos_dec = cos(dec_pix);
		const double sin_dec = sin(dec_pix);
			double lp = -INFINITY;
			double ln = -INFINITY;
			for(int it = 0; it < ntime; it++) {
			const double tg = time[it];
			const double cos_alpha = cos_ra * cos_gmsts[it] + sin_ra * sin_gmsts[it];
			const double sin_alpha = sin_ra * cos_gmsts[it] - cos_ra * sin_gmsts[it];
			const double source[3] = {
				cos_dec * cos_alpha,
				cos_dec * sin_alpha,
				sin_dec
			};
			const double p[3] = {sin_alpha, -cos_alpha, 0.0};
			const double q[3] = {
				-sin_dec * cos_alpha,
				-sin_dec * sin_alpha,
				cos_dec
			};
				double complex nump = 0.0;
				double complex numn = 0.0;
				double denp = 0.0;
				double denn = 0.0;
				double complex bplus = 0.0;
				double complex bcross = 0.0;
				double gpp = 0.0;
				double gpc = 0.0;
				double gcc = 0.0;
				double complex num_cbc[33 * 16];
				double den_cbc[33 * 16];
				double fplus_arr[16];
				double fcross_arr[16];
				double complex shifted_data[16][nfreq];
				if(nifo > 16) {
					continue;
				}
				if(amplitude_model == 2) {
					for(int inc = 0; inc < inclination_samples * polarization_samples; inc++) {
						num_cbc[inc] = 0.0;
						den_cbc[inc] = 0.0;
					}
				}
				for(int ifo = 0; ifo < nifo; ifo++) {
				const double *r = resp + 9 * ifo;
				const double *x = loc + 3 * ifo;
				const double delay = -(x[0] * source[0] + x[1] * source[1] + x[2] * source[2]) / c_si;
				double fplus = 0.0, fcross = 0.0;
				double complex shifted_sum = 0.0;
				for(int a = 0; a < 3; a++) {
					for(int b = 0; b < 3; b++) {
						const double rab = r[3 * a + b];
						fplus += rab * (p[a] * p[b] - q[a] * q[b]);
						fcross += rab * (p[a] * q[b] + q[a] * p[b]);
					}
				}
					const double phase_arg = tg + delay - epoch[ifo];
					const double complex *ifo_data = prepared + (Py_ssize_t) ifo * nfreq;
					for(int irun = 0; irun < nruns; irun++) {
						const int start = run_starts[irun];
						const int stop = run_stops[irun];
						double s, c;
						const double phase = two_pi * phase_sign * freq[start] * phase_arg;
						s = sin(phase);
						c = cos(phase);
						double complex phasor = c + I * s;
						if(stop - start > 1) {
							double ss, cs;
							const double phase_step = two_pi * phase_sign * (freq[start + 1] - freq[start]) * phase_arg;
							ss = sin(phase_step);
							cs = cos(phase_step);
							const double complex step = cs + I * ss;
							for(int k = start; k < stop; k++) {
								const double complex shifted = ifo_data[k] * phasor;
								shifted_sum += shifted;
								if(amplitude_model == 3)
									shifted_data[ifo][k] = shifted;
								phasor *= step;
							}
						} else {
							const double complex shifted = ifo_data[start] * phasor;
							shifted_sum += shifted;
							if(amplitude_model == 3)
								shifted_data[ifo][start] = shifted;
						}
					}
					if(amplitude_model == 3) {
						fplus_arr[ifo] = fplus;
						fcross_arr[ifo] = fcross;
					}
					{
					const double complex response_p = fplus + I * fcross;
					const double complex response_n = fplus - I * fcross;
					if(amplitude_model == 2) {
						for(int inc = 0; inc < inclination_samples; inc++) {
							const double cosi = inclination_samples == 1 ? 1.0 : -1.0 + 2.0 * (double) inc / (double) (inclination_samples - 1);
							const double aplus = 0.5 * (1.0 + cosi * cosi);
							const double across = cosi;
							for(int pol = 0; pol < polarization_samples; pol++) {
								const int idx = inc * polarization_samples + pol;
								const double psi = M_PI * (double) pol / (double) polarization_samples;
								const double c2p = cos(2.0 * psi);
								const double s2p = sin(2.0 * psi);
								const double fplus_psi = fplus * c2p + fcross * s2p;
								const double fcross_psi = -fplus * s2p + fcross * c2p;
								const double complex response = aplus * fplus_psi + I * across * fcross_psi;
								const double complex response_factor = paper_conj_response ? conj(response) : response;
								num_cbc[idx] += response_factor * shifted_sum;
								den_cbc[idx] += (aplus * aplus * fplus_psi * fplus_psi + across * across * fcross_psi * fcross_psi) * denw[ifo];
							}
						}
					} else if(amplitude_model == 1) {
						bplus += fplus * shifted_sum;
						bcross += fcross * shifted_sum;
						gpp += fplus * fplus * denw[ifo];
						gpc += fplus * fcross * denw[ifo];
						gcc += fcross * fcross * denw[ifo];
					} else {
						nump += (paper_conj_response ? conj(response_p) : response_p) * shifted_sum;
						numn += (paper_conj_response ? conj(response_n) : response_n) * shifted_sum;
						denp += (fplus * fplus + fcross * fcross) * denw[ifo];
						denn += (fplus * fplus + fcross * fcross) * denw[ifo];
					}
					}
				}
				if(amplitude_model == 3) {
					for(int beta = -1; beta <= 1; beta += 2) {
						double norm = 0.0;
						double score = 0.0;
						for(int ifo = 0; ifo < nifo; ifo++)
							norm += fplus_arr[ifo] * fplus_arr[ifo] + fcross_arr[ifo] * fcross_arr[ifo];
						if(norm > 0.0) {
							for(int i = 0; i < nifo; i++) {
								const double complex response_i = fplus_arr[i] + I * (double) beta * fcross_arr[i];
								for(int j = 0; j < i; j++) {
									const double complex response_j = fplus_arr[j] + I * (double) beta * fcross_arr[j];
									const double complex proj = response_i * conj(response_j) / norm;
									const double pair_den = sqrt(fmax(denw[i] * denw[j], 1.0));
									double complex cross = 0.0;
									for(int k = 0; k < nfreq; k++)
										cross += conj(shifted_data[i][k]) * shifted_data[j][k];
									score += creal(proj * cross) / ((double) nfreq * pair_den);
								}
							}
							if(beta > 0)
								lp = sph_logaddexp(lp, score_scale * score);
							else
								ln = sph_logaddexp(ln, score_scale * score);
						}
					}
				} else if(amplitude_model == 2) {
					for(int inc = 0; inc < inclination_samples * polarization_samples; inc++) {
						if(den_cbc[inc] > 0.0) {
							const double score = (creal(num_cbc[inc]) * creal(num_cbc[inc]) + cimag(num_cbc[inc]) * cimag(num_cbc[inc])) / den_cbc[inc];
							lp = sph_logaddexp(lp, score_scale * score);
						}
					}
				} else if(amplitude_model == 1) {
					const double det = gpp * gcc - gpc * gpc;
					if(det > 1e-300) {
						const double score = (
							gcc * (creal(bplus) * creal(bplus) + cimag(bplus) * cimag(bplus)) +
							gpp * (creal(bcross) * creal(bcross) + cimag(bcross) * cimag(bcross)) -
							2.0 * gpc * creal(conj(bplus) * bcross)
						) / det;
						lp = sph_logaddexp(lp, score_scale * score);
						ln = sph_logaddexp(ln, score_scale * score);
					}
				} else {
					if(denp > 0.0)
						lp = sph_logaddexp(lp, score_scale * (creal(nump) * creal(nump) + cimag(nump) * cimag(nump)) / denp);
					if(denn > 0.0)
						ln = sph_logaddexp(ln, score_scale * (creal(numn) * creal(numn) + cimag(numn) * cimag(numn)) / denn);
				}
		}
		if(amplitude_model == 2) {
			logp[pix] = lp - log((double) ntime * (double) inclination_samples * (double) polarization_samples);
			logn[pix] = logp[pix];
		} else {
			logp[pix] = lp - log((double) ntime);
			logn[pix] = ln - log((double) ntime);
		}
	}
	Py_END_ALLOW_THREADS
	}

	PyBuffer_Release(&locations);
	PyBuffer_Release(&responses);
	PyBuffer_Release(&freqs);
	PyBuffer_Release(&times);
	PyBuffer_Release(&epochs);
	PyBuffer_Release(&data);
	PyBuffer_Release(&den_weight);
	PyBuffer_Release(&ra);
	PyBuffer_Release(&dec);
	PyBuffer_Release(&out_p);
	PyBuffer_Release(&out_n);
	free(sin_gmsts);
	free(cos_gmsts);
	free(run_starts);
	free(run_stops);
	Py_RETURN_NONE;

error:
	free(sin_gmsts);
	free(cos_gmsts);
	free(run_starts);
	free(run_stops);
	if(locations.buf) PyBuffer_Release(&locations);
	if(responses.buf) PyBuffer_Release(&responses);
	if(freqs.buf) PyBuffer_Release(&freqs);
	if(times.buf) PyBuffer_Release(&times);
	if(epochs.buf) PyBuffer_Release(&epochs);
	if(data.buf) PyBuffer_Release(&data);
	if(den_weight.buf) PyBuffer_Release(&den_weight);
	if(ra.buf) PyBuffer_Release(&ra);
	if(dec.buf) PyBuffer_Release(&dec);
	if(out_p.buf) PyBuffer_Release(&out_p);
	if(out_n.buf) PyBuffer_Release(&out_n);
	return NULL;
}
%}

%inline %{
        struct correlator_plan_fd *pick_ith_correlator_plan_fd(struct correlator_plan_fd **array, int i) {
                return array[i];
        }


        double pick_deltaT_from_COMPLEX16TimeSeries(COMPLEX16TimeSeries *series) {
                return series->deltaT;
        }


        long pick_length_from_COMPLEX16TimeSeries(COMPLEX16TimeSeries *series) {
                return series->data->length;
        }


        void free_SNRTimeSeries(COMPLEX16TimeSeries *series) {
                XLALDestroyCOMPLEX16TimeSeries(series);
        }


        void free_SNRSequence(COMPLEX16Sequence *series) {
                XLALDestroyCOMPLEX16Sequence(series);
        }


        int create_sph_COMPLEX16TimeSeries(COMPLEX16TimeSeries *result, char *name, double epoch_Seconds, double epoch_NanoSeconds, double f0, double deltaT, LALUnit *sampleUnits, size_t length, double complex *data) {
                int i;
                if(!result || !data) {
                        fprintf(stderr, "memory error\n");
                        return -1;
                }
                result->data = XLALCreateCOMPLEX16Sequence(length);

                result->epoch.gpsSeconds = epoch_Seconds;
                result->epoch.gpsNanoSeconds = epoch_NanoSeconds;

                strncpy(result->name, name, LALNameLength - 1);
                result->name[LALNameLength - 1] = '\0';
                result->f0 = f0;
                result->deltaT = deltaT;
                for(i = 0; i < (int) result->data->length; i++) {
                        result->data->data[i] = data[i];
                }
                if(sampleUnits)
                        result->sampleUnits = *sampleUnits;

                return 0;
        }


        int create_sph_COMPLEX16Sequence(COMPLEX16Sequence *result, size_t length, double complex *data) {
                int i;
                if(!result || !data) {
                        fprintf(stderr, "memory error\n");
                        return -1;
                }

                result->data = malloc(length * sizeof(*result->data));
                if(!result->data) {
                        fprintf(stderr, "memory error\n");
                        return -1;
                }

                result->length = length;
                for(i = 0; i < (int) length; i++) {
                        result->data[i] = data[i];
                }

                return 0;
        }
%}

%include <sphradiometer/instrument.h>
%include <sphradiometer/sh_series.h>
%include <sphradiometer/inject.h>
%include <sphradiometer/projection.h>
%include <sphradiometer/correlator.h>
%include <sphradiometer/sky.h>
%include <sphradiometer/deconvolution.h>
%include <sphradiometer/inspiral_sky_map.h>

%pointer_functions(unsigned int, uintp);
%pointer_functions(struct correlator_network_plan_fd, correlator_network_plan_fdp);
%pointer_functions(struct autocorrelator_network_plan_fd, autocorrelator_network_plan_fdp);
%pointer_functions(struct sh_series, sh_seriesp);
%pointer_functions(struct sh_series *, sh_seriespp);
%pointer_functions(COMPLEX16TimeSeries, COMPLEX16TimeSeriesp);
%pointer_functions(COMPLEX16Sequence, COMPLEX16Sequencep);

%array_functions(double, double_array);
%array_functions(double *, doublep_array);
%array_functions(double complex, double_complex_array);
%array_functions(COMPLEX16TimeSeries *, COMPLEX16TimeSeries_array);
%array_functions(COMPLEX16Sequence *, COMPLEX16Sequence_array);
