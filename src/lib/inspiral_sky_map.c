/*
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


/*
 * ============================================================================
 *
 *                                  Preamble
 *
 * ============================================================================
 */


#include <complex.h>
#include <fftw3.h>
#include <math.h>
#include <sphradiometer/inspiral_sky_map.h>
#include <sphradiometer/instrument.h>
#include <sphradiometer/sh_series.h>
#include <sphradiometer/correlator.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>

#include <lal/Date.h>
#include <lal/DetResponse.h>
#include <lal/LALFrStream.h>
#include <lal/LALDatatypes.h>
#include <lal/LALMalloc.h>
#include <lal/LALSimulation.h>
#include <lal/Sequence.h>
#include <lal/TimeSeries.h>
#include <lal/XLALError.h>
#include <lal/Units.h>
#include <lal/Window.h>


#define FILE_LEN 110


/*
 * ============================================================================
 *
 *                                 Data Input
 *
 * ============================================================================
 */


COMPLEX16TimeSeries *get_complex16series_from_cache(const char *cache_name, const char *channel_name)
{
	char instrument[] = {channel_name[0], channel_name[1], '\0'};
	LALCache *cache;
	LALFrStream *stream;
	COMPLEX8TimeSeries *data;
	COMPLEX16TimeSeries *result;
	unsigned i;

	/* construct stream */
	cache = XLALCacheImport(cache_name);
	if(!cache)
		XLAL_ERROR_NULL(XLAL_EFUNC);
	XLALCacheSieve(cache, 0, 0, instrument, NULL, NULL);
	if(cache->length != 1) {
		XLALPrintError("error: cache must contain exactly 1 file for instrument %s", instrument);
		XLALDestroyCache(cache);
		XLAL_ERROR_NULL(XLAL_EDATA);
	}
	stream = XLALFrStreamCacheOpen(cache);
	XLALDestroyCache(cache);
	if(!stream)
		XLAL_ERROR_NULL(XLAL_EFUNC);

	/* turn on checking for missing data */
	stream->mode = LAL_FR_STREAM_VERBOSE_MODE;

	/* get data */
	data = XLALFrFileReadCOMPLEX8TimeSeries(stream->file, channel_name, 0);

	/* check for gaps and close */
	XLALFrStreamClose(stream);

	/* error checking */
	if(!data)
		XLAL_ERROR_NULL(XLAL_EFUNC);

	/* convert to double precision */
	result = XLALCreateCOMPLEX16TimeSeries(data->name, &data->epoch, data->f0, data->deltaT, &data->sampleUnits, data->data->length);
	for(i = 0; i < data->data->length; i++)
		result->data->data[i] = data->data->data[i];
	XLALDestroyCOMPLEX8TimeSeries(data);
	/*for(i = 0; i < result->data->length; i++) fprintf(stderr, "%g+I*%g\n", creal(result->data->data[i]), cimag(result->data->data[i]));*/

	/* done */
	{
	char *s = XLALGPSToStr(NULL, &result->epoch);
	fprintf(stderr, "%s: loaded \"%s\" at %s s, duration=%g s (%d samples)\n", instrument, result->name, s, result->data->length * result->deltaT, result->data->length);
	XLALFree(s);
	}
	return result;
}


COMPLEX16Sequence *convert_TimeSeries2Sequence(COMPLEX16TimeSeries *series)
{
	return XLALCopyCOMPLEX16Sequence(series->data);
}


long precalculated_TimeSeries_length(long length, double deltaT)
{
	return (long) (round(2 * LAL_REARTH_SI / LAL_C_SI / deltaT) + length);
}


/*
 * make the time series span the same intervals
 * FIXME:  this leaves the final GPS times slightly different.  why?
 */


static int time_series_pad(COMPLEX16TimeSeries **series, COMPLEX16Sequence **nseries, int n_series)
{
	int i;
	LIGOTimeGPS start;
	LIGOTimeGPS end;

	start = end = series[0]->epoch;
	XLALGPSAdd(&end, series[0]->deltaT * series[0]->data->length);
	for(i = 1; i < n_series; i++) {
		LIGOTimeGPS this_end = series[i]->epoch;
		XLALGPSAdd(&this_end, series[i]->deltaT * series[i]->data->length);
		if(XLALGPSCmp(&start, &series[i]->epoch) > 0)
			start = series[i]->epoch;
		if(XLALGPSCmp(&end, &this_end) < 0)
			end = this_end;
	}

	long data_length = precalculated_TimeSeries_length(series[0]->data->length, series[0]->deltaT);
	for(i = 0; i < n_series; i++) {
		XLALResizeCOMPLEX16TimeSeries(series[i], round(XLALGPSDiff(&start, &series[i]->epoch) / series[i]->deltaT), round(XLALGPSDiff(&end, &start) / series[i]->deltaT));
		XLALResizeCOMPLEX16TimeSeries(series[i], round((series[i]->data->length - data_length) / 2.), data_length);
		XLALResizeCOMPLEX16Sequence(nseries[i], -((int) series[i]->data->length - (int) nseries[i]->length) / 2, series[i]->data->length);
	}

#if 0
	for(i = 0; i < n_series; i++) {
		char *s = XLALGPSToStr(NULL, &series[i]->epoch);
		fprintf(stderr, "zero-padded interval for \"%s\": %s s, duration=%g s (%d samples)\n", series[i]->name, s, series[i]->data->length * series[i]->deltaT, series[i]->data->length);
		XLALFree(s);
		{
		unsigned j;
		for(j = 0; j < series[i]->data->length; j++) {
			LIGOTimeGPS t = series[i]->epoch;
			XLALGPSAdd(&t, j * series[i]->deltaT);
			s = XLALGPSToStr(NULL, &t);
			fprintf(stderr, "%s: %g+I*%g\n", s, creal(series[i]->data->data[j]), cimag(series[i]->data->data[j]));
			XLALFree(s);
		}
		fprintf(stderr, "\n");
		}
	}
#endif

	return 0;
}


static void scale_COMPLEX16Sequence(COMPLEX16Sequence *series, double factor)
{
	unsigned int i;
	for(i = 0; i < series->length; i++)
		series->data[i] *= factor;
}


void preprocess_SNRTimeSeries(COMPLEX16TimeSeries **series, COMPLEX16Sequence **nseries, int n_series)
{
	int k;

	/* Tukey Window */
	for(k = 0; k < n_series; k++){
		int j;
		REAL8Window *window = XLALCreateTukeyREAL8Window(series[k]->data->length, 0.1);
		for(j = 0; j < (int) window->data->length; j++){
			series[k]->data->data[j] *= window->data->data[j];
			nseries[k]->data[j] *= window->data->data[j];
		}
		XLALDestroyREAL8Window(window);
	}

	/* bring to a common interval with zero-padding */
	/* To set epochs of SNR time series, original time series are padded,
	 * that is, the lengths become longer.  Then the d.o.f. of data are
	 * diluted.  The dilution happens in the normalization 1 / (data
	 * length) in correlator_network_integrate_power_fd().  To correct the
	 * normalization factor, the padded SNR time series have to be scaled
	 * by \sqrt{(padded data length) / (original data length)}. */
	for(k = 0; k < n_series; k++)
		scale_COMPLEX16Sequence(series[k]->data, sqrt(1. / series[k]->data->length));
	time_series_pad(series, nseries, n_series);
	for(k = 0; k < n_series; k++)
		scale_COMPLEX16Sequence(series[k]->data, sqrt(series[k]->data->length));
}


double **transpose_matrix(double **mat, int n_det, int length)
{
	/* mat is a (n_det * length)-matrix */
	int i, j;
	double **tmat = malloc(length * sizeof(*tmat));

	if(!tmat)
		return NULL;

	for(i = 0; i < length; i++) {
		tmat[i] = malloc(n_det * sizeof(*tmat[i]));
		if(!tmat[i]) {
			free(tmat);
			return NULL;
		}
	}

	/* tmat is a (length * n_det)-matrix */
	for(i = 0; i < length; i++)
		for(j = 0; j < n_det; j++)
			tmat[i][j] = mat[j][i];

	return tmat;
}


/*
 * ============================================================================
 *
 *                              Time Conversion
 *
 * ============================================================================
 */


static double gmst_from_epoch_and_offset(LIGOTimeGPS epoch, double offset)
{
	return XLALGreenwichSiderealTime(XLALGPSAdd(&epoch, offset), 0.0);
}


/*
 * ============================================================================
 *
 *                             Projection for antenna
 *
 * ============================================================================
 */

#if 0

/*
 * compute normalized F+ and Fx vectors for the direction (theta, phi) for
 * all n detectors in the dominant polarization frame.
 */


static void FDP(double *fplus, double *fcross, const LALDetector **det, int n, double theta, double phi)
{
	double twopsi;
	double normplus2;
	double normcross2;
	double product;
	int i;

	/* compute F+ and Fx.  GSMT is set to 0, the Earth will be rotated
	 * in generate_alm_sky() */
	for(i = 0; i < n; i++)
		XLALComputeDetAMResponse(&fplus[i], &fcross[i], det[i]->response, phi, M_PI_2 - theta, 0.0, 0.0);

	/* dominant polarization angle */
	normplus2 = normcross2 = product = 0.0;
	for(i = 0; i < n; i++) {
		normplus2 += fplus[i] * fplus[i];
		normcross2 += fcross[i] * fcross[i];
		product += fplus[i] * fcross[i];
	}
	twopsi = atan2(2.0 * product, normplus2 - normcross2) / 2.;

	/* rotate (F+, Fx) for each detector by 2psi */
	for(i = 0; i < n; i++) {
		double temp = cos(twopsi) * fplus[i] + sin(twopsi) * fcross[i];
		fcross[i] = -sin(twopsi) * fplus[i] + cos(twopsi) * fcross[i];
		fplus[i] = temp;
	}

	/* normalize the {F+} and {Fx} vectors to be unit vectors */
	normplus2 = normcross2 = 0.0;
	for(i = 0; i < n; i++) {
		normplus2 += fplus[i] * fplus[i];
		normcross2 += fcross[i] * fcross[i];
	}
	for(i = 0; i < n; i++) {
		fplus[i] /= sqrt(normplus2);
		fcross[i] /= sqrt(normcross2);
	}
}


static complex double ExcessProjectionMatrix(double theta, double phi, int i, int j, const LALDetector **det, int n, double *psds)
{
	double fplus[n], fcross[n];

	FDP(fplus, fcross, det, n, theta, phi);

	return fplus[i] * fplus[j] + fcross[i] * fcross[j] - (int)(i == j);
}

#endif


/*
 * psds have a different normalization with the paper.  However, fplus & fcross
 * and those normalization terms weighted with the psds have the same excess
 * factors.  Projection operator is defined like $P \sim fs / |fs|$, and then
 * the operator is convention-free.  Thus you don't have to consider the
 * convention difference.
 */


static complex double CBCProjectionMatrix(double theta, double phi, int i, int j, const LALDetector **det, int n, double beta, double psi, double gmst, double *psds, int subtract_identity)
{
	/* this is CBC parameterized one for arbitrary beta & psi */
	double fplus[n], fcross[n];
	double normplus2;
	double normcross2;

	int k;
	normplus2 = normcross2 = 0.0;
	for(k = 0; k < n; k++){
		/* store fp, fc */
		XLALComputeDetAMResponse(&fplus[k], &fcross[k], det[k]->response, phi, M_PI_2 - theta, psi, gmst);
		fplus[k] /= psds[k];
		fcross[k] /= psds[k];
		/* calculate norms of vector fp & fc */
		normplus2 += fplus[k] * fplus[k];
		normcross2 += fcross[k] * fcross[k];
	}

	return (fplus[i] + I * beta * fcross[i]) * (fplus[j] - I * beta * fcross[j]) / (normplus2 + beta*beta * normcross2) - (subtract_identity && i == j);
}


struct ProjectionMatrixWrapperData {
	int i, j;
	const LALDetector **det;
	int n;
	double beta;
	double psi;
	double gmst;
	double *psds;
	int subtract_identity;
};


static complex double ProjectionMatrixWrapper(double theta, double phi, void *_data)
{
	struct ProjectionMatrixWrapperData *data = _data;

#if 0
	return ExcessProjectionMatrix(theta, phi, data->i, data->j, data->det, data->n);
#else
	return CBCProjectionMatrix(theta, phi, data->i, data->j, data->det, data->n, data->beta, data->psi, data->gmst, data->psds, data->subtract_identity);
#endif
}


static struct correlator_plan_fd *correlator_plan_mult_by_projection(struct correlator_plan_fd *plan, struct sh_series *const *projection)
{
	struct sh_series_array *result;
	struct sh_series_product_plan *product_plan;
	int i;

	result = sh_series_array_new(plan->delay_product->n, plan->delay_product->l_max, plan->delay_product->series[0].polar && projection[0]->polar);
	if(!result)
		return NULL;
	product_plan = sh_series_product_plan_new(&result->series[0], &plan->delay_product->series[0], projection[0]);
	if(!product_plan) {
		XLALPrintError("sh_series_product_plan_new() failed\n");
		sh_series_array_free(result);
		return NULL;
	}

	for(i = 0; i < plan->delay_product->n; i++) {
		fprintf(stderr, "%.3g%%   \r", 100. * (i + 1.) / plan->delay_product->n);
		if(!sh_series_product(&result->series[i], &plan->delay_product->series[i], projection[i], product_plan)) {
			XLALPrintError("sh_series_product() failed\n");
			sh_series_array_free(result);
			return NULL;
		}
	}
	fprintf(stderr, "\n");

	sh_series_product_plan_free(product_plan);
	sh_series_array_free(plan->delay_product);
	plan->delay_product = result;

	return plan;
}


int correlator_network_plan_mult_by_projection_gmst_frame(struct correlator_network_plan_fd *plan, double beta, double psi, double response_gmst, double frame_gmst, double **psd)
{
	int i, j;
	struct sh_series **projection = malloc(plan->plans[0]->delay_product->n * sizeof(*projection));
	const struct instrument_array *instruments = plan->baselines->baselines[0]->instruments;
	const LALDetector **det = malloc(instrument_array_len(instruments) * sizeof(*det));

	if(!projection || !instruments || !det) {
		free(projection);
		free(det);
		return -1;
	}

	for(i = 0; i < instrument_array_len(instruments); i++) {
		det[i] = instrument_array_get(instruments, i)->data;
		if(!det[i]) {
			free(projection);
			free(det);
			return -1;
		}
	}


	for(i = 0; i < plan->baselines->n_baselines; i++) {
		/* initialize */
		for(j = 0; j < plan->plans[i]->delay_product->n; j++) {
			projection[j] = sh_series_new(plan->plans[i]->delay_product->l_max, 0);
			if(!projection[j]) {
				goto error;
			}
		}

		fprintf(stderr, "generate Projection distribution from %d and %d\n", plan->baselines->baselines[i]->index_a, plan->baselines->baselines[i]->index_b);
		for(j = 0; j < plan->plans[i]->delay_product->n; j++) {
			struct ProjectionMatrixWrapperData data = {
				.i = plan->baselines->baselines[i]->index_a,
				.j = plan->baselines->baselines[i]->index_b,
				.det = det,
				.n = instrument_array_len(instruments),
				.beta = beta,
				.psi = psi,
				.gmst = response_gmst,
				.psds = psd[j],
				.subtract_identity = 1
			};
			if(!sh_series_from_func(projection[j], ProjectionMatrixWrapper, &data)) {
				goto error;
			}

#if 0
			/* plot projection */
			char filename[32] ={"\0"};
			char istr[12];
			snprintf(istr, sizeof(istr), "%d", i);
			strcat(filename, "projection");
			strcat(filename, istr);
			strcat(filename, ".fits");
			if(sh_series_write_healpix_alm(projection[j], filename)) {
				fprintf(stderr, "write \"%s\" failed\n", filename);
				free(det);
				exit(1);
			}
#endif
		}

		/* the delay operator is computed with the baseline rotated
		 * to lie along the z axis to take advantage of the
		 * aziumthal symmetry of that operator, so we need to
		 * rotate the projection function we've just computed the
		 * same way */
			{
				double *R = sh_series_invrot_matrix(plan->baselines->baselines[i]->theta, plan->baselines->baselines[i]->phi + frame_gmst);
			struct sh_series_rotation_plan *rot = sh_series_rotation_plan_new(projection[0], R);
			free(R);
			for(j = 0; j < plan->plans[i]->delay_product->n; j++) {
				struct sh_series *cpy = sh_series_copy(projection[j]);
				sh_series_rotate(projection[j], cpy, rot);
				sh_series_free(cpy);
			}
			sh_series_rotation_plan_free(rot);
			}
		/* multiply each frequency bin of the delay operator by the
		 * projection operator */
		if(!correlator_plan_mult_by_projection(plan->plans[i], projection)) {
			goto error;
		}
		/* replace the correlator's intermediate "1D" storage with
		 * one that does not have azimuthal symmetry, increase it
		 * to the correct harmonic order, and compute a new
		 * rotation plan for it.  NOTE:  the two-stage design that
		 * takes advantage of rotational symmetry about the
		 * baseline axis cannot be used here because after
		 * multiplying by the projection operator above we no
		 * longer have that rotational symmetry;  really we should
		 * compute the delay matrix with the correct orientation
		 * with respect to the sky, and not pay the price of the
		 * rotation */
		if(!sh_series_set_polar(plan->plans[i]->power_1d, 0) || !sh_series_resize(plan->plans[i]->power_1d, plan->plans[i]->delay_product->series[0].l_max)) {
			goto error;
			}
			{
			double *R = sh_series_rot_matrix(plan->baselines->baselines[i]->theta, plan->baselines->baselines[i]->phi + frame_gmst);
		if(!R) {
			goto error;
		}
		sh_series_rotation_plan_free(plan->plans[i]->rotation_plan);
		plan->plans[i]->rotation_plan = sh_series_rotation_plan_new(plan->plans[i]->power_1d, R);
		free(R);
		if(!plan->plans[i]->rotation_plan) {
			goto error;
		}
		}
		/* free */
		for(j = 0; j < plan->plans[i]->delay_product->n; j++)
			sh_series_free(projection[j]);
	}

	free(projection);
	free(det);

	return 0;

error:
	for(i = 0; i < plan->plans[0]->delay_product->n; i++)
		sh_series_free(projection[i]);
	free(projection);
	free(det);
	return -1;
}


int correlator_network_plan_mult_by_projection_gmst(struct correlator_network_plan_fd *plan, double beta, double psi, double gmst, double **psd)
{
	return correlator_network_plan_mult_by_projection_gmst_frame(plan, beta, psi, gmst, gmst, psd);
}


int correlator_network_plan_mult_by_projection(struct correlator_network_plan_fd *plan, double beta, double psi, double **psd)
{
	return correlator_network_plan_mult_by_projection_gmst_frame(plan, beta, psi, 0.0, 0.0, psd);
}


int correlator_network_plan_conj_delay_product(struct correlator_network_plan_fd *plan)
{
	int i, j;

	for(i = 0; i < plan->baselines->n_baselines; i++)
		for(j = 0; j < plan->plans[i]->delay_product->n; j++)
			sh_series_conj(&plan->plans[i]->delay_product->series[j]);

	return 0;
}


/*
 * ============================================================================
 *
 *                                Diagonal Part
 *
 * ============================================================================
 */


static struct sh_series ***diagonal_projections(const struct instrument_array *instruments, double beta, double psi, double **psd, int length, int l_max)
{
	int i, j, k;
	struct sh_series ***projection = malloc(instrument_array_len(instruments) * sizeof(*projection));
	const LALDetector **det = malloc(instrument_array_len(instruments) * sizeof(*det));

	if(!projection || !det) {
		free(projection);
		free(det);
		return NULL;
	}

	/* set instruments information */
	for(i = 0; i < instrument_array_len(instruments); i++) {
		det[i] = instrument_array_get(instruments, i)->data;
		projection[i] = malloc(length * sizeof(*projection));
		if(!det[i] || !projection[i]) {
			free(projection);
			free(det);
			return NULL;
		}
	}

	for(k = 0; k < instrument_array_len(instruments); k++) {
		for(i = 0; i < length; i++) {
			projection[k][i] = sh_series_new(l_max, 0);
			if(!projection[k][i]) {
				for(j = i; j > -1; j--) {
					sh_series_free(projection[k][j]);
				}
				free(projection);
				free(det);
				fprintf(stderr, "fail %d-th frequency projection operator\n", i);
				return NULL;
			}
		}
	}

	for(i = 0; i < instrument_array_len(instruments); i++) {
		fprintf(stderr, "generate Projection distribution from %d and %d\n", i, i);
		for(j = 0; j < length; j++) {
			struct ProjectionMatrixWrapperData data = {
				.i = i,
				.j = i,
				.det = det,
					.n = instrument_array_len(instruments),
					.beta = beta,
					.psi = psi,
					.psds = psd[j],
					.subtract_identity = 0
				};
			if(!sh_series_from_func(projection[i][j], ProjectionMatrixWrapper, &data)) {
				goto error;
			}

#if 0
			/* plot projection */
			char filename[32] ={"\0"};
			char istr[12];
			snprintf(istr, sizeof(istr), "%d", i);
			strcat(filename, "projection");
			strcat(filename, istr);
			strcat(filename, ".fits");
			if(sh_series_write_healpix_alm(projection[j], filename)) {
				fprintf(stderr, "write \"%s\" failed\n", filename);
				free(det);
				exit(1);
			}
#endif
		}
	}

	free(det);
	return projection;

error:
	for(i = 0; i < instrument_array_len(instruments); i++) {
		for(j = 0; j < length; j++)
			sh_series_free(projection[i][j]);
	}
	free(projection);
	free(det);
	return NULL;
}


struct autocorrelator_network_plan_fd *autocorrelator_network_plan_fd_new(const struct instrument_array *instruments, double beta, double psi, double **psd, int length, int l_max)
{
	struct autocorrelator_network_plan_fd *new = malloc(sizeof(*new));

	if(!new) {
		free(new);
		return NULL;
	}

	new->instruments = instruments;
	new->length = length;
	new->projections = diagonal_projections(instruments, beta, psi, psd, length, l_max);

	return new;
}


struct autocorrelator_network_plan_fd *autocorrelator_network_plan_fd_set_l(struct autocorrelator_network_plan_fd *plan, unsigned l_max)
{
	int i, j;

	for(i = 0; i < instrument_array_len(plan->instruments); i++) {
		for(j = 0; j < plan->length; j++) {
			if(!sh_series_resize(plan->projections[i][j], l_max))
				return NULL;
		}
	}

	return plan;
}


void autocorrelator_network_plan_fd_free(struct autocorrelator_network_plan_fd *plan)
{
	int i, j;

	for(i = 0; i < instrument_array_len(plan->instruments); i++) {
		for(j = 0; j < plan->length; j++) {
			sh_series_free(plan->projections[i][j]);
		}
		free(plan->projections[i]);
	}

	free(plan->projections);
	free(plan);
}


static int autocorrelator_network_from_projection(struct sh_series *sky, complex double **fseries, struct autocorrelator_network_plan_fd *plan, double auto_term_scale)
{
	int i, j;
	double normalization_factor;

	normalization_factor = 1. / (plan->length * plan->length);
	for(j = 2; j <= instrument_array_len(plan->instruments); j++) {
		normalization_factor /= j;
	}

	for(i = 0; i < instrument_array_len(plan->instruments); i++) {
			/* execute calc. */
			for(j = 0; j < plan->length; j++)
				sh_series_add(sky, auto_term_scale * fseries[i][j] * conj(fseries[i][j]) * normalization_factor, plan->projections[i][j]);
	}

	return 0;
}


/*
 * ============================================================================
 *
 *                                Whitening
 *
 * ============================================================================
 */


static complex double inner_product(complex double *a, complex double *b, int length)
{
	int i;
	complex double prod = 0;

	for(i = 0; i < length; i++)
		prod += *a++ * conj(*b++);

	return prod;
}


static complex double *matrix_dot_vector(complex double **mat, complex double *vec, int dim)
{
	int i;
	complex double *result = malloc(dim * sizeof(*result));

	for(i = 0; i < dim; i++)
		result[i] = inner_product(mat[i], vec, dim);

	return result;
}


static void complex_conjugate_vector(complex double *vec, int dim)
{
	int i;

	for(i = 0; i < dim; i++)
		vec[i] = conj(vec[i]);
}


static complex double **convert_basis_from_frequency(complex double **mat, complex double **evec_columns, int dim)
{
	/* mat is on frequency domain.  this function convert mat on evec. */
	int i, j;
	complex double **result = malloc(dim * sizeof(*result));

	/* initialize */
	for(i = 0; i < dim; i++)
		result[i] = malloc(dim * sizeof(*result[i]));

	/* M_{i, j} = \tilde{evec[i]}^{f*} M_{f, f'} \tilde{evec[j]}^{f'} */
	for(i = 0; i < dim; i++) {
		/* When constructing fftplan, save_i is broken.  Not to loss
		 * evec_columns, save_i is used. */
		complex double *save_i = malloc(dim * sizeof(*save_i));
		memcpy(save_i, evec_columns[i], dim * sizeof(*save_i));
		complex double *fevec_i = malloc(dim * sizeof(*fevec_i));
		fftw_plan fftplan_i = correlator_ctseries_to_fseries_plan(save_i, fevec_i, dim);
		correlator_ctseries_to_fseries(fftplan_i);

		/* make fevec_i complex conjugate */
		complex_conjugate_vector(fevec_i, dim);

		for(j = 0; j < dim; j++) {
			/* When constructing fftplan, save_j is broken.  Not to loss
			 * evec_columns, save_j is used. */
			complex double *save_j = malloc(dim * sizeof(*save_j));
			memcpy(save_j, evec_columns[j], dim * sizeof(*save_j));
			complex double *fevec_j = malloc(dim * sizeof(*fevec_j));
			fftw_plan fftplan_j = correlator_ctseries_to_fseries_plan(save_j, fevec_j, dim);
			correlator_ctseries_to_fseries(fftplan_j);

			/* store */
			complex double *M_dot_e = matrix_dot_vector(mat, fevec_j, dim);
			result[i][j] = inner_product(fevec_i, M_dot_e, dim);

			/* free */
			free(M_dot_e);
			free(save_j);
			free(fevec_j);
			fftw_destroy_plan(fftplan_j);
		}

		/* free */
		free(save_i);
		free(fevec_i);
		fftw_destroy_plan(fftplan_i);
	}

	return result;
}


static double *KL_transform(COMPLEX16TimeSeries *series, complex double **evec_columns)
{
	int i;
	double *result = malloc(series->data->length * sizeof(*result));

	for(i = 0; i < (int) series->data->length; i++)
		result[i] = inner_product(series->data->data, evec_columns[i], (int) series->data->length);

	return result;
}


static int whiten(complex double *series, complex double *noise, int length)
{
	int i;

	for(i = 0; i < length; i++)
		series[i] /= sqrt(cabs(noise[i]));

	return 0;
}


static int shift_autocorrelation_zero_lag(const complex double *series, complex double *shifted, int length)
{
	int i;
	int zero_lag = 0;
	double max = -1.0;

	for(i = 0; i < length; i++) {
		const double value = cabs(series[i]);
		if(value > max) {
			max = value;
			zero_lag = i;
		}
	}
	for(i = 0; i < length; i++)
		shifted[i] = series[(i + zero_lag) % length];

	return 0;
}


static int make_autocorrelation_template_series(complex double **template_fseries, COMPLEX16Sequence **nseries, int n)
{
	int k;

	for(k = 0; k < n; k++) {
		complex double *shifted = malloc(nseries[k]->length * sizeof(*shifted));
		fftw_plan plan;

		if(!shifted)
			return -1;
		if(shift_autocorrelation_zero_lag(nseries[k]->data, shifted, nseries[k]->length)) {
			free(shifted);
			return -1;
		}
		plan = correlator_ctseries_to_fseries_plan(shifted, template_fseries[k], nseries[k]->length);
		if(!plan) {
			free(shifted);
			return -1;
		}
		correlator_ctseries_to_fseries(plan);
		fftw_destroy_plan(plan);
		free(shifted);
	}

	return 0;
}


static int make_template_weighted_series(complex double **weighted, complex double **fseries, complex double **template_fseries, int n, int length)
{
	int k;

	for(k = 0; k < n; k++) {
		int i;
		for(i = 0; i < length; i++) {
			const double norm = cabs(template_fseries[k][i]);
			weighted[k][i] = norm > 0.0 ? fseries[k][i] * conj(template_fseries[k][i]) / norm : 0.0;
		}
	}

	return 0;
}


/*
 * ============================================================================
 *
 *                                   prior
 *
 * ============================================================================
 */


static double log_uniformsky_prior(double theta, double phi, void *data)
{
	return log(sin(theta) / (4 * M_PI));
}


struct sh_series *sh_series_log_uniformsky_prior(int l_max)
{
	struct sh_series *new = sh_series_new(l_max, 0);
	if(!new)
		return NULL;
	if(!sh_series_from_realfunc(new, log_uniformsky_prior, NULL)) {
		sh_series_free(new);
		return NULL;
	}
	return new;
}


/*
 * ============================================================================
 *
 *                                  alm map
 *
 * ============================================================================
 */



static complex double *geocentric_time_window_new(const COMPLEX16TimeSeries *series, double center, double half_width, int *is_noop)
{
	complex double *window;
	double epoch;
	double duration;
	unsigned i;
	unsigned n_in_window = 0;

	*is_noop = 0;
	if(half_width < 0.0)
		return NULL;

	epoch = XLALGPSGetREAL8(&series->epoch);
	duration = series->data->length * series->deltaT;
	if(center - half_width <= epoch && epoch + duration <= center + half_width) {
		*is_noop = 1;
		fprintf(stderr, "geocentric time window covers full span; using unwindowed FD correlator\n");
		return NULL;
	}

	window = calloc(series->data->length, sizeof(*window));
	if(!window)
		return NULL;

	for(i = 0; i < series->data->length; i++) {
		const double t = epoch + i * series->deltaT;
		if(fabs(t - center) <= half_width) {
			window[i] = 1.0;
			n_in_window++;
		}
	}

	if(!n_in_window) {
		fprintf(stderr, "empty geocentric time window: center=%.16g half_width=%.16g epoch=%.16g duration=%.16g samples=%u\n", center, half_width, epoch, duration, series->data->length);
		free(window);
		return NULL;
	}

	fprintf(stderr, "using geocentric time window: center=%.16g half_width=%.16g samples=%u\n", center, half_width, n_in_window);
	return window;
}


static int generate_alm_sky(struct sh_series *sky, struct correlator_network_plan_fd *fdplans, struct autocorrelator_network_plan_fd *fdautoplan, complex double **fseries, complex double **fnseries, const complex double *time_window, double gmst, struct sh_series *logprior, double auto_term_scale, double regulator_scale)
{
	if(regulator_scale < 0.0)
		regulator_scale = fdplans->baselines->n_baselines * 4.0;

	/*
	 * Compute angular distribution of integrated cross power.
	 */


	fprintf(stderr, "starting integration\n");
	if(!correlator_network_integrate_power_fd_window(sky, fseries, time_window, fdplans)) {
		fprintf(stderr, "cross-correlator failed\n");
		return -1;
	}

	/* We calculated contrubutions from lower triangular part of Projection
	 * matrix, cross-correlation. However upper one still remain.
	 * Fortunately this calculations are easy because Projection and Time
	 * shift operator is symmetric w.r.t. baseline index, e.g. H1 <--> L1.
	 * Therefore it's OK to twice simply. NOTE: We have to pick up only
	 * real part because correlator is Hermite. However this manipulation
	 * is already done in sh_series_write_healpix_alm(). */
	sh_series_scale(sky, 2.0);
	/* above one is for only cross-correlator, below one for cross- and
	 * auto-correlator */
	if(auto_term_scale != 0.0) {
		/* add contributions from diagonal part, auto-correlation. FIXME: check
		 * consistency between above and below sh_series_scale() */
		fprintf(stderr, "including auto-correlation terms with scale %.16g\n", auto_term_scale);
		if(autocorrelator_network_from_projection(sky, fseries, fdautoplan, auto_term_scale)) {
			fprintf(stderr, "auto-correlator failed\n");
			return -1;
		}
	}
	/* multiply baseline numbers. Correlator is normalized by it (See
	 * correlator_network_integrate_power_fd()). However our calculation
	 * doesn't need it. */
	/* Correlator is normalized by data length (See
	 * correlator_plan_fd_new() or
	 * correlator_baseline_integrate_power_fd(). Honestly we have either
	 * one enough because Kipp's paper has incorrect calculation. However
	 * codes pass all consistency checks. Threfore our codes doesn't have
	 * error but mistakes. We can neglect the mistakes because our result
	 * is correct). However our Likelihood does't need it. */
	/* Now our calculation has no whitening. To justify it, we recognize it
	 * as multiplying a regulator. The regulator should be normalized. One
	 * SNR timeseries give us one 2 as normalized factor. Hence multiply 4.
	 * */
	sh_series_scale(sky, regulator_scale);
	fprintf(stderr, "finished integration\n");


	/*
	 * prior
	 */


#if 1
	fprintf(stderr, "start multiply prior\n");
	sh_series_add(sky, 1.0, logprior);
#endif


	/*
	 * Rotate sky.
	 */


	fprintf(stderr, "gmst = %.16g rad\n", gmst);
	sh_series_rotate_z(sky, sky, gmst);

	return 0;
}


int generate_alm_skys_options(struct sh_series **skyp, struct sh_series **skyn, struct correlator_network_plan_fd *fdplansp, struct correlator_network_plan_fd *fdplansn, struct autocorrelator_network_plan_fd *fdautoplanp, struct autocorrelator_network_plan_fd *fdautoplann, COMPLEX16TimeSeries **series, COMPLEX16Sequence **nseries, struct sh_series *logprior, double auto_term_scale, int include_whitening, double regulator_scale, double time_prior_center, double time_prior_half_width, int use_autocorrelation_template)
{
	complex double **fseries;
	complex double **fnseries;
	complex double **template_fseries;
	complex double **weighted_fseries;
	complex double **sky_fseries;
	complex double *time_window;
	int time_window_is_noop = 0;
	fftw_plan *fftplans;
	fftw_plan *nfftplans;
	int n = instrument_array_len(fdplansp->baselines->baselines[0]->instruments);
	int k;


	/*
	 * Prepare some serieses
	 */


	fseries = malloc(n * sizeof(*fseries));
	fnseries = malloc(n * sizeof(*fnseries));
	template_fseries = malloc(n * sizeof(*template_fseries));
	weighted_fseries = malloc(n * sizeof(*weighted_fseries));
	fftplans = malloc(n * sizeof(*fftplans));
	nfftplans = malloc(n * sizeof(*nfftplans));
	if(!fseries || !fftplans || !fnseries || !nfftplans || !template_fseries || !weighted_fseries) {
		XLALPrintError("out of memory\n");
		return -1;
	}


	/*
	 * prepare correlator
	 */


	for(k = 0; k < n; k++) {
		complex double *save = malloc(series[k]->data->length * sizeof(*save));
		complex double *nsave = malloc(nseries[k]->length * sizeof(*nsave));
		memcpy(save, series[k]->data->data, series[k]->data->length * sizeof(*save));
		memcpy(nsave, nseries[k]->data, nseries[k]->length * sizeof(*nsave));
		fseries[k] = malloc(series[k]->data->length * sizeof(**fseries));
		fnseries[k] = malloc(nseries[k]->length * sizeof(**fnseries));
		template_fseries[k] = malloc(nseries[k]->length * sizeof(**template_fseries));
		weighted_fseries[k] = malloc(series[k]->data->length * sizeof(**weighted_fseries));
		if(!fseries[k] || !fnseries[k] || !template_fseries[k] || !weighted_fseries[k]) {
			XLALPrintError("out of memory\n");
			return -1;
		}
		fftplans[k] = correlator_ctseries_to_fseries_plan(series[k]->data->data, fseries[k], series[k]->data->length);
		nfftplans[k] = correlator_ctseries_to_fseries_plan(nseries[k]->data, fnseries[k], nseries[k]->length);
		memcpy(series[k]->data->data, save, series[k]->data->length * sizeof(*save));
		memcpy(nseries[k]->data, nsave, nseries[k]->length * sizeof(*nsave));
		free(save);
		free(nsave);
	}


	/*
	 * Fourier transform data
	 */


	for(k = 0; k < n; k++) {
		correlator_ctseries_to_fseries(fftplans[k]);
		correlator_ctseries_to_fseries(nfftplans[k]);
#if 0
		unsigned i; for(i = 0; i < series[k]->data->length; i++) fprintf(stderr, "%3d  %g  %g\n", i, cabs(fseries[k][i]), cabs(fnseries[k][i])); fprintf(stderr, "\n");
		int j = 0;
		double temp = cabs(fseries[k][0]);
		for(i = 1; i < series[k]->data->length; i++) {
			if(temp < cabs(fseries[k][i])){
				j = i;
				temp = cabs(fseries[k][i]);
			}
		}
		fprintf(stderr, "signal %d %g\n", j, temp);

		j = 0;
		temp = cabs(fnseries[k][0]);
		for(i = 1; i < nseries[k]->length; i++) {
			if(temp < cabs(fnseries[k][i])){
				j = i;
				temp = cabs(fnseries[k][i]);
			}
		}
		fprintf(stderr, "noise  %d %g\n\n", j, temp);
#endif
	}

	if(use_autocorrelation_template) {
		fprintf(stderr, "using autocorrelation-shaped SNR templates\n");
		if(make_autocorrelation_template_series(template_fseries, nseries, n)) {
			fprintf(stderr, "failed to construct autocorrelation template spectra\n");
			return -1;
		}
		if(make_template_weighted_series(weighted_fseries, fseries, template_fseries, n, series[0]->data->length)) {
			fprintf(stderr, "failed to construct template-weighted SNR spectra\n");
			return -1;
		}
		sky_fseries = weighted_fseries;
	} else {
		sky_fseries = fseries;
	}


	/*
	 * Whitening
	 */


	if(include_whitening) {
	//	double min = cabs(fnseries[0][0]);
	//	for(k = 0; k < n; k++)
	//		for(j = 0; j < (int) series[k]->data->length; j++)
	//			if(min > cabs(fnseries[k][j]))
	//				min = cabs(fnseries[k][j]);
	//	min *= 1e3;
	//	fprintf(stderr, "threshold = %g\n", min);

		for(k = 0; k < n; k++){
			double temp = 0;
			unsigned i;
			for(i = 0; i < series[k]->data->length; i++)
				temp += cabs(fnseries[k][i]);
			for(i = 0; i < series[k]->data->length; i++)
				fnseries[k][i] /= temp * temp;

			//threshold(fseries[k], fnseries[k], series[k]->data->length, min);
			whiten(fseries[k], fnseries[k], series[k]->data->length);
			//partial_whiten(fseries[k], fnseries[k], (int) (1024 * series[k]->data->length * series[k]->deltaT), (int) (2048 * series[k]->data->length * series[k]->deltaT));
			//filter(fseries[k], (int) (15 * series[k]->data->length * series[k]->deltaT), (int) (1024 * series[k]->data->length * series[k]->deltaT), series[k]->data->length);
		}
	}
#if 0
	for(k = 0; k < n; k++){
		unsigned i; for(i = 0; i < series[k]->data->length; i++) fprintf(stderr, "%3d %g  %g\n", i, cabs(fseries[k][i]), cabs(fnseries[k][i])); fprintf(stderr, "\n");
		j = 0;
		double temp = cabs(fseries[k][0]);
		for(i = 1; i < series[k]->data->length; i++) {
			if(temp < cabs(fseries[k][i])){
				j = i;
				temp = cabs(fseries[k][i]);
			}
		}
		fprintf(stderr, "%d %g\n\n", j, temp);
	}
#endif

	time_window = geocentric_time_window_new(series[0], time_prior_center, time_prior_half_width, &time_window_is_noop);
	if(time_prior_half_width >= 0.0 && !time_window && !time_window_is_noop) {
		fprintf(stderr, "failed to construct geocentric time window\n");
		return -1;
	}


	/*
	 * Generate sky alm for each parameters of CBC
	 */


	*skyp = sh_series_new_zero(logprior->l_max, 0);
	if(generate_alm_sky(*skyp, fdplansp, fdautoplanp, sky_fseries, fnseries, time_window, gmst_from_epoch_and_offset(series[0]->epoch, series[0]->data->length * series[0]->deltaT / 2.0), logprior, auto_term_scale, regulator_scale)) {
		fprintf(stderr, "positive generate_alm_sky error\n");
		return -1;
	}


	*skyn = sh_series_new_zero(logprior->l_max, 0);
	if(generate_alm_sky(*skyn, fdplansn, fdautoplann, sky_fseries, fnseries, time_window, gmst_from_epoch_and_offset(series[0]->epoch, series[0]->data->length * series[0]->deltaT / 2.0), logprior, auto_term_scale, regulator_scale)) {
		fprintf(stderr, "negative generate_alm_sky error\n");
		return -1;
	}


	/*
	 * Clean up
	 */


	for(k = 0; k < n; k++) {
		free(fseries[k]);
		free(fnseries[k]);
		free(template_fseries[k]);
		free(weighted_fseries[k]);
		fftw_destroy_plan(fftplans[k]);
		fftw_destroy_plan(nfftplans[k]);
	}
	free(fseries);
	free(fnseries);
	free(template_fseries);
	free(weighted_fseries);
	free(time_window);
	free(fftplans);
	free(nfftplans);

	return 0;
}


int generate_alm_skys(struct sh_series **skyp, struct sh_series **skyn, struct correlator_network_plan_fd *fdplansp, struct correlator_network_plan_fd *fdplansn, struct autocorrelator_network_plan_fd *fdautoplanp, struct autocorrelator_network_plan_fd *fdautoplann, COMPLEX16TimeSeries **series, COMPLEX16Sequence **nseries, struct sh_series *logprior)
{
	return generate_alm_skys_options(skyp, skyn, fdplansp, fdplansn, fdautoplanp, fdautoplann, series, nseries, logprior, 0.0, 0, -1.0, 0.0, -1.0, 1);
}


/*
 * ============================================================================
 *
 *                                Read precalcs
 *
 * ============================================================================
 */


static struct correlator_baseline *read_precalc_correlator_baseline(const struct instrument_array *instruments, int i, char *parent_dir)
{
	char filename[strlen(parent_dir) + FILE_LEN];
	FILE *fp;
	struct correlator_baseline *new = NULL;
	int index_a, index_b;
	double theta, phi;
	int success = 1;

	/* read no pointer objects */
	sprintf(filename, "%s/correlator_network_plan_fd/correlator_network_baselines/baselines/%d/correlator_baseline.dat", parent_dir, i);
	fp = fopen(filename, "rb");
	if(!fp) {
		fprintf(stderr, "no correlator_baseline.dat in %d\n", i);
		goto done;
	}
	/* following order must be consistent with the one in the writer */
	success &= fread(&index_a, sizeof(index_a), 1, fp) == 1;
	success &= fread(&index_b, sizeof(index_b), 1, fp) == 1;
	success &= fread(&theta, sizeof(theta), 1, fp) == 1;
	success &= fread(&phi, sizeof(phi), 1, fp) == 1;
	fclose(fp);
	if(!success) {
		fprintf(stderr, "failure reading correlator_baseline.dat in %d\n", i);
		goto done;
	}

	new = correlator_baseline_new(instruments, index_a, index_b);
	if(!new) {
		goto done;
	}

	if(new->theta != theta || new->phi != phi) {
		fprintf(stderr, "inconsistent data in correlator_baseline.dat\n");
		correlator_baseline_free(new);
		new = NULL;
		goto done;
	}

#if 0
	/* read d */
	sprintf(filename, "%s/correlator_network_plan_fd/correlator_network_baselines/baselines/%d/d.dat", parent_dir, i);
	fp = fopen(filename, "rb");
	if(!fp) {
		fprintf(stderr, "no d.dat in %d\n", i);
		return NULL;
	}
	new->d = gsl_vector_alloc(3);
	gsl_vector_fread(fp, new->d);
	fclose(fp);

	/* substitute instruments */
	new->instruments = instruments;
#endif

done:
	return new;
}


struct correlator_network_baselines *read_precalc_correlator_network_baselines(const struct instrument_array *instruments, char *parent_dir)
{
	char filename[strlen(parent_dir) + FILE_LEN];
	struct correlator_network_baselines *new = malloc(sizeof(*new));
	int i;

	/* read n_baselines */
	sprintf(filename, "%s/correlator_network_plan_fd/correlator_network_baselines/n_baselines.dat", parent_dir);
	FILE *fp = fopen(filename, "rb");
	if(!fp) {
		fprintf(stderr, "no n_baselines.dat\n");
		return NULL;
	}
	fread(&new->n_baselines, sizeof(new->n_baselines), 1, fp);
	fclose(fp);

	/* read baselines */
	new->baselines = malloc(new->n_baselines * sizeof(*new->baselines));
	for(i = 0; i < new->n_baselines; i++)
		new->baselines[i] = read_precalc_correlator_baseline(instruments, i, parent_dir);

	return new;
}


static struct sh_series_rotation_plan *read_precalc_sh_series_rotation_plan(int i, char *parent_dir)
{
	char filename[strlen(parent_dir) + FILE_LEN];
	FILE *fp;
	struct sh_series_rotation_plan *new = malloc(sizeof(*new));

	/* read l_max */
	sprintf(filename, "%s/correlator_network_plan_fd/correlator_plan_fd/%d/rotation_plan/l_max.dat", parent_dir, i);
	fp = fopen(filename, "rb");
	if(!fp) {
		fprintf(stderr, "no %dth l_max.dat\n", i);
		return NULL;
	}
	fread(&new->l_max, sizeof(new->l_max), 1, fp);
	fclose(fp);

	/* read D */
	sprintf(filename, "%s/correlator_network_plan_fd/correlator_plan_fd/%d/rotation_plan/D.dat", parent_dir, i);
	fp = fopen(filename, "rb");
	if(!fp) {
		fprintf(stderr, "no %dth D.dat\n", i);
		return NULL;
	}
	new->D = malloc(new->l_max * sizeof(*new->D));
	new->D--;
	unsigned int l;
	for(l = 1; l <= new->l_max; l++) {
		new->D[l] = malloc((2 * l + 1) * (2 * l + 1) * sizeof(*new->D[l]));
		fread(new->D[l], sizeof(*new->D[l]), (2 * l + 1) * (2 * l + 1), fp);
		new->D[l] += (2 * l + 1) * l + l;
	}
	fclose(fp);

	return new;
}


static int read_precalc_correlator_plan_fd(struct correlator_plan_fd *fdplanp, struct correlator_plan_fd *fdplann, const struct correlator_network_baselines *baselines, int tseries_length, int i, char *parent_dir)
{
	char filename[strlen(parent_dir) + FILE_LEN];
	FILE *fp;
	int j;

	if(!fdplanp || !fdplann) {
		fprintf(stderr, "memory of correlator_plan_fd must be allocated\n");
		return -1;
	}

	/* read no pointer objects */
	sprintf(filename, "%s/correlator_network_plan_fd/correlator_plan_fd/%d/correlator_plan_fd.dat", parent_dir, i);
	fp = fopen(filename, "rb");
	if(!fp) {
		fprintf(stderr, "no %dth correlator_plan_fd.dat\n", i);
		return -1;
	}
	/* following order must be consistent with the one in the writer */
	fread(&fdplanp->delta_t, sizeof(fdplanp->delta_t), 1, fp);
	fread(&fdplanp->transient, sizeof(fdplanp->transient), 1, fp);
	fclose(fp);
	fdplann->delta_t = fdplanp->delta_t;
	fdplann->transient = fdplanp->transient;

	/* read baseline */
	fdplanp->baseline = baselines->baselines[i];
	fdplann->baseline = baselines->baselines[i];

	/* read rotation_plan */
	fdplanp->rotation_plan = read_precalc_sh_series_rotation_plan(i, parent_dir);
	fdplann->rotation_plan = sh_series_rotation_plan_copy(fdplanp->rotation_plan);

	/* read delay_product */
	/* positive case */
	fdplanp->delay_product = malloc(sizeof(*fdplanp->delay_product));
	sprintf(filename, "%s/correlator_network_plan_fd/correlator_plan_fd/%d/delay_product_p/sh_series_array.dat", parent_dir, i);
	fp = fopen(filename, "rb");
	if(!fp) {
		fprintf(stderr, "can't open %dth positive sh_series_array.dat\n", i);
		return -1;
	}
	/* following order must be consistent with the one in the writer */
	fread(&fdplanp->delay_product->l_max, sizeof(fdplanp->delay_product->l_max), 1, fp);
	fread(&fdplanp->delay_product->polar, sizeof(fdplanp->delay_product->polar), 1, fp);
	fread(&fdplanp->delay_product->n, sizeof(fdplanp->delay_product->n), 1, fp);
	fread(&fdplanp->delay_product->stride, sizeof(fdplanp->delay_product->stride), 1, fp);
	fdplanp->delay_product->coeff = malloc(fdplanp->delay_product->n * fdplanp->delay_product->stride * sizeof(*fdplanp->delay_product->coeff));
	fread(fdplanp->delay_product->coeff, sizeof(*fdplanp->delay_product->coeff), fdplanp->delay_product->n * fdplanp->delay_product->stride, fp);
	fclose(fp);

	fdplanp->delay_product->series = malloc(fdplanp->delay_product->n * sizeof(*fdplanp->delay_product->series));
	for(j = 0; j < fdplanp->delay_product->n; j++) {
		fdplanp->delay_product->series[j] = (struct sh_series) {
			.l_max = fdplanp->delay_product->l_max,
			.polar = fdplanp->delay_product->polar,
			.coeff = fdplanp->delay_product->coeff + j * fdplanp->delay_product->stride
		};
	}

	/* negative case */
	fdplann->delay_product = malloc(sizeof(*fdplann->delay_product));
	sprintf(filename, "%s/correlator_network_plan_fd/correlator_plan_fd/%d/delay_product_n/sh_series_array.dat", parent_dir, i);
	fp = fopen(filename, "rb");
	if(!fp) {
		fprintf(stderr, "can't open %dth negative sh_series_array.dat\n", i);
		return -1;
	}
	/* following order must be consistent with the one in the writer */
	fread(&fdplann->delay_product->l_max, sizeof(fdplann->delay_product->l_max), 1, fp);
	fread(&fdplann->delay_product->polar, sizeof(fdplann->delay_product->polar), 1, fp);
	fread(&fdplann->delay_product->n, sizeof(fdplann->delay_product->n), 1, fp);
	fread(&fdplann->delay_product->stride, sizeof(fdplann->delay_product->stride), 1, fp);
	fdplann->delay_product->coeff = malloc(fdplann->delay_product->n * fdplann->delay_product->stride * sizeof(*fdplann->delay_product->coeff));
	fread(fdplann->delay_product->coeff, sizeof(*fdplann->delay_product->coeff), fdplann->delay_product->n * fdplann->delay_product->stride, fp);
	fclose(fp);

	fdplann->delay_product->series = malloc(fdplann->delay_product->n * sizeof(*fdplann->delay_product->series));
	for(j = 0; j < fdplann->delay_product->n; j++) {
		fdplann->delay_product->series[j] = (struct sh_series) {
			.l_max = fdplann->delay_product->l_max,
			.polar = fdplann->delay_product->polar,
			.coeff = fdplann->delay_product->coeff + j * fdplann->delay_product->stride
		};
	}


	/* read fseries_product */
	/* fseries_product is one of the results.  don't read but initialize. */
	fdplanp->fseries_product = malloc(tseries_length * sizeof(*fdplanp->fseries_product));
	fdplann->fseries_product = malloc(tseries_length * sizeof(*fdplann->fseries_product));

	/* read power_1d */
	/* power_1d is one of the results.  don't read but initialize. */
	fdplanp->power_1d = sh_series_new(fdplanp->rotation_plan->l_max, fdplanp->delay_product->polar);
	fdplann->power_1d = sh_series_new(fdplann->rotation_plan->l_max, fdplann->delay_product->polar);

	return 0;
}


int read_precalc_correlator_network_plan_fd(struct correlator_network_plan_fd *fdplansp, struct correlator_network_plan_fd *fdplansn, const struct correlator_network_baselines *baselines, int tseries_length, char *parent_dir)
{
	if(!fdplansp || !fdplansn) {
		fprintf(stderr, "memory of correlator_network_plan_fd must be allocated\n");
		return -1;
	}

	/* read baselines */
	fdplansp->baselines = baselines;
	fdplansn->baselines = baselines;

	/* read plans */
	fdplansp->plans = malloc(fdplansp->baselines->n_baselines * sizeof(*fdplansp->plans));
	fdplansn->plans = malloc(fdplansn->baselines->n_baselines * sizeof(*fdplansn->plans));
	int i;
	for(i = 0; i < fdplansp->baselines->n_baselines; i++) {
		fdplansp->plans[i] = malloc(sizeof(*fdplansp->plans[i]));
		fdplansn->plans[i] = malloc(sizeof(*fdplansn->plans[i]));
		if(read_precalc_correlator_plan_fd(fdplansp->plans[i], fdplansn->plans[i], baselines, tseries_length, i, parent_dir)) {
			fprintf(stderr, "can't read %dth correlator_plan_fd\n", i);
			return -1;
		}
	}

	return 0;
}


int read_precalc_autocorrelator_network_plan_fd(struct autocorrelator_network_plan_fd *fdplansp, struct autocorrelator_network_plan_fd *fdplansn, const struct instrument_array *instruments, int tseries_length, char *parent_dir)
{
	char filename[strlen(parent_dir) + FILE_LEN];
	FILE *fp;
	int i, j;

	if(!fdplansp || !fdplansn) {
		fprintf(stderr, "memory of correlator_network_plan_fd must be allocated\n");
		return -1;
	}

	/* read no pointer objects */
	sprintf(filename, "%s/autocorrelator_network_plan_fd/length.dat", parent_dir);
	fp = fopen(filename, "rb");
	if(!fp) {
		fprintf(stderr, "no length.dat\n");
		return -1;
	}
	fread(&fdplansp->length, sizeof(fdplansp->length), 1, fp);
	fclose(fp);
	fdplansn->length = fdplansp->length;

	/* read projections */
	/* positive case */
	fdplansp->projections = malloc(instrument_array_len(instruments) * sizeof(*fdplansp->projections));
	for(i = 0; i < instrument_array_len(instruments); i++) {
		fdplansp->projections[i] = malloc(fdplansp->length * sizeof(*fdplansp->projections[i]));
		for(j = 0; j < fdplansp->length; j++) {
			sprintf(filename, "%s/autocorrelator_network_plan_fd/psh_series/%d/sh_series_p/projections%d.fits", parent_dir, i, j);
			fdplansp->projections[i][j] = sh_series_read_healpix_alm(filename);
			if(!fdplansp->projections[i][j]) {
				fprintf(stderr, "no %dth projections%d.fits\n", i, j);
				return -1;
			}
		}
	}

	/* negative case */
	fdplansn->projections = malloc(instrument_array_len(instruments) * sizeof(*fdplansn->projections));
	for(i = 0; i < instrument_array_len(instruments); i++) {
		fdplansn->projections[i] = malloc(fdplansn->length * sizeof(*fdplansn->projections[i]));
		for(j = 0; j < fdplansn->length; j++) {
			sprintf(filename, "%s/autocorrelator_network_plan_fd/psh_series/%d/sh_series_n/projections%d.fits", parent_dir, i, j);
			fdplansn->projections[i][j] = sh_series_read_healpix_alm(filename);
			if(!fdplansn->projections[i][j]) {
				fprintf(stderr, "no %dth projections%d.fits\n", i, j);
				return -1;
			}
		}
	}

	/* substitute instruments */
	fdplansp->instruments = instruments;
	fdplansn->instruments = instruments;

	return 0;
}


struct sh_series *read_precalc_logprior(char *parent_dir)
{
	char filename[strlen(parent_dir) + FILE_LEN];
	sprintf(filename, "%s/sh_series/logprior.fits", parent_dir);
	return sh_series_read_healpix_alm(filename);
}


int read_precalc_time_series_length(unsigned int *length, char *parent_dir)
{
	char filename[strlen(parent_dir) + FILE_LEN];
	FILE *fp;

	/* read time series length */
	sprintf(filename, "%s/time_series_length.dat", parent_dir);
	fp = fopen(filename, "rb");
	if(!fp) {
		fprintf(stderr, "no time_series_length.dat\n");
		return -1;
	}
	fread(length, sizeof(*length), 1, fp);
	fclose(fp);
	return 0;
}


/*
 * ============================================================================
 *
 *                                Write precalcs
 *
 * ============================================================================
 */


int make_precalc_directories(char *precalc_path, int n_ifo)
{
	int k;

	mode_t mode = S_IRWXU | S_IRWXG | S_IRWXO;
	char dirname[strlen(precalc_path) + FILE_LEN];

	fprintf(stderr, "make precalc directory in %s\n", precalc_path);
	if(mkdir(precalc_path, mode)) {
		fprintf(stderr, "error making %s\n", precalc_path);
		return -1;
	}
	sprintf(dirname, "%s/correlator_network_plan_fd", precalc_path);
	if(mkdir(dirname, mode)) {
		fprintf(stderr, "error making %s\n", dirname);
		return -1;
	}
	sprintf(dirname, "%s/correlator_network_plan_fd/correlator_network_baselines", precalc_path);
	if(mkdir(dirname, mode)) {
		fprintf(stderr, "error making %s\n", dirname);
		return -1;
	}
	sprintf(dirname, "%s/correlator_network_plan_fd/correlator_network_baselines/baselines", precalc_path);
	if(mkdir(dirname, mode)) {
		fprintf(stderr, "error making %s\n", dirname);
		return -1;
	}
	for(k = 0; k < n_ifo; k++){
		sprintf(dirname, "%s/correlator_network_plan_fd/correlator_network_baselines/baselines/%d", precalc_path, k);
		if(mkdir(dirname, mode)) {
			fprintf(stderr, "error making %s\n", dirname);
			return -1;
		}
	}
	sprintf(dirname, "%s/correlator_network_plan_fd/correlator_plan_fd", precalc_path);
	if(mkdir(dirname, mode)) {
		fprintf(stderr, "error making %s\n", dirname);
		return -1;
	}
	for(k = 0; k < n_ifo; k++){
		sprintf(dirname, "%s/correlator_network_plan_fd/correlator_plan_fd/%d", precalc_path, k);
		if(mkdir(dirname, mode)) {
			fprintf(stderr, "error making %s\n", dirname);
			return -1;
		}
		sprintf(dirname, "%s/correlator_network_plan_fd/correlator_plan_fd/%d/delay_product_n", precalc_path, k);
		if(mkdir(dirname, mode)) {
			fprintf(stderr, "error making %s\n", dirname);
			return -1;
		}
		sprintf(dirname, "%s/correlator_network_plan_fd/correlator_plan_fd/%d/delay_product_p", precalc_path, k);
		if(mkdir(dirname, mode)) {
			fprintf(stderr, "error making %s\n", dirname);
			return -1;
		}
		sprintf(dirname, "%s/correlator_network_plan_fd/correlator_plan_fd/%d/rotation_plan", precalc_path, k);
		if(mkdir(dirname, mode)) {
			fprintf(stderr, "error making %s\n", dirname);
			return -1;
		}
	}
	sprintf(dirname, "%s/autocorrelator_network_plan_fd", precalc_path);
	if(mkdir(dirname, mode)) {
		fprintf(stderr, "error making %s\n", dirname);
		return -1;
	}
	sprintf(dirname, "%s/autocorrelator_network_plan_fd/psh_series", precalc_path);
	if(mkdir(dirname, mode)) {
		fprintf(stderr, "error making %s\n", dirname);
		return -1;
	}
	for(k = 0; k < n_ifo; k++){
		sprintf(dirname, "%s/autocorrelator_network_plan_fd/psh_series/%d", precalc_path, k);
		if(mkdir(dirname, mode)) {
			fprintf(stderr, "error making %s\n", dirname);
			return -1;
		}
		sprintf(dirname, "%s/autocorrelator_network_plan_fd/psh_series/%d/sh_series_n", precalc_path, k);
		if(mkdir(dirname, mode)) {
			fprintf(stderr, "error making %s\n", dirname);
			return -1;
		}
		sprintf(dirname, "%s/autocorrelator_network_plan_fd/psh_series/%d/sh_series_p", precalc_path, k);
		if(mkdir(dirname, mode)) {
			fprintf(stderr, "error making %s\n", dirname);
			return -1;
		}
	}
	sprintf(dirname, "%s/sh_series", precalc_path);
	if(mkdir(dirname, mode)) {
		fprintf(stderr, "error making %s\n", dirname);
		return -1;
	}

	return 0;
}


static int write_precalc_correlator_baseline(const struct correlator_baseline *baseline, int i, char *parent_dir)
{
	char filename[strlen(parent_dir) + FILE_LEN];
	FILE *fp;

	/* write no pointer objects */
	sprintf(filename, "%s/correlator_network_plan_fd/correlator_network_baselines/baselines/%d/correlator_baseline.dat", parent_dir, i);
	fp = fopen(filename, "wb");
	if(!fp) {
		fprintf(stderr, "can't open correlator_baseline.dat\n");
		return -1;
	}
	/* following order must be consistent with the one in the reader */
	fwrite(&baseline->index_a, sizeof(baseline->index_a), 1, fp);
	fwrite(&baseline->index_b, sizeof(baseline->index_b), 1, fp);
	fwrite(&baseline->theta, sizeof(baseline->theta), 1, fp);
	fwrite(&baseline->phi, sizeof(baseline->phi), 1, fp);
	fclose(fp);

	/* write d */
	sprintf(filename, "%s/correlator_network_plan_fd/correlator_network_baselines/baselines/%d/d.dat", parent_dir, i);
	fp = fopen(filename, "wb");
	if(!fp) {
		fprintf(stderr, "can't open d.dat\n");
		return -1;
	}
	gsl_vector_fwrite(fp, baseline->d);
	fclose(fp);

	/* instruments which is a member of correlator_baseline are given by
	 * command line options.  Then, we don't have to store it. */

	return 0;
}


static int write_precalc_correlator_network_baselines(const struct correlator_network_baselines *baselines, char *parent_dir)
{
	char filename[strlen(parent_dir) + FILE_LEN];

	/* write n_baselines */
	sprintf(filename, "%s/correlator_network_plan_fd/correlator_network_baselines/n_baselines.dat", parent_dir);
	FILE *fp = fopen(filename, "wb");
	if(!fp) {
		fprintf(stderr, "can't open n_baselines.dat\n");
		return -1;
	}
	fwrite(&baselines->n_baselines, sizeof(baselines->n_baselines), 1, fp);
	fclose(fp);

	/* write baselines */
	int i;
	for(i = 0; i < baselines->n_baselines; i++) {
		if(write_precalc_correlator_baseline(baselines->baselines[i], i, parent_dir)) {
			fprintf(stderr, "can't save %dth correlator_baseline\n", i);
			return -1;
		}
	}

	return 0;
}


static int write_precalc_sh_series_rotation_plan(const struct sh_series_rotation_plan *plan, int i, char *parent_dir)
{
	char filename[strlen(parent_dir) + FILE_LEN];
	FILE *fp;

	/* write l_max */
	sprintf(filename, "%s/correlator_network_plan_fd/correlator_plan_fd/%d/rotation_plan/l_max.dat", parent_dir, i);
	fp = fopen(filename, "wb");
	if(!fp) {
		fprintf(stderr, "can't open %dth l_max.dat\n", i);
		return -1;
	}
	fwrite(&plan->l_max, sizeof(plan->l_max), 1, fp);
	fclose(fp);

	/* write D */
	sprintf(filename, "%s/correlator_network_plan_fd/correlator_plan_fd/%d/rotation_plan/D.dat", parent_dir, i);
	fp = fopen(filename, "wb");
	if(!fp) {
		fprintf(stderr, "can't open %dth D.dat\n", i);
		return -1;
	}
	unsigned int l;
	for(l = 1; l <= plan->l_max; l++) {
		plan->D[l] -= (2 * l + 1) * l + l;
		fwrite(plan->D[l], sizeof(*plan->D[l]), (2 * l + 1) * (2 * l + 1), fp);
		plan->D[l] += (2 * l + 1) * l + l;
	}
	fclose(fp);

	return 0;
}


static int write_precalc_correlator_plan_fd(const struct correlator_plan_fd *planp, const struct correlator_plan_fd *plann, int i, char *parent_dir)
{
	char filename[strlen(parent_dir) + FILE_LEN];
	FILE *fp;

	/* write no pointer objects */
	sprintf(filename, "%s/correlator_network_plan_fd/correlator_plan_fd/%d/correlator_plan_fd.dat", parent_dir, i);
	fp = fopen(filename, "wb");
	if(!fp) {
		fprintf(stderr, "can't open %dth correlator_plan_fd.dat\n", i);
		return -1;
	}
	/* following order must be consistent with the one in the reader */
	fwrite(&planp->delta_t, sizeof(planp->delta_t), 1, fp);
	fwrite(&planp->transient, sizeof(planp->transient), 1, fp);
	fclose(fp);

	/* write rotation_plan */
	if(write_precalc_sh_series_rotation_plan(planp->rotation_plan, i, parent_dir)) {
		fprintf(stderr, "can't open %dth rotation_plan.dat\n", i);
		return -1;
	}

	/* write delay_product (sh_series_array) */
	/* positive case */
	sprintf(filename, "%s/correlator_network_plan_fd/correlator_plan_fd/%d/delay_product_p/sh_series_array.dat", parent_dir, i);
	fp = fopen(filename, "wb");
	if(!fp) {
		fprintf(stderr, "can't open %dth positive sh_series_array.dat\n", i);
		return -1;
	}
	/* following order must be consistent with the one in the reader */
	fwrite(&planp->delay_product->l_max, sizeof(planp->delay_product->l_max), 1, fp);
	fwrite(&planp->delay_product->polar, sizeof(planp->delay_product->polar), 1, fp);
	fwrite(&planp->delay_product->n, sizeof(planp->delay_product->n), 1, fp);
	fwrite(&planp->delay_product->stride, sizeof(planp->delay_product->stride), 1, fp);
	fwrite(planp->delay_product->coeff, sizeof(*planp->delay_product->coeff), planp->delay_product->n * planp->delay_product->stride, fp);
	fclose(fp);

	/* negative case */
	sprintf(filename, "%s/correlator_network_plan_fd/correlator_plan_fd/%d/delay_product_n/sh_series_array.dat", parent_dir, i);
	fp = fopen(filename, "wb");
	if(!fp) {
		fprintf(stderr, "can't open %dth negative sh_series_array.dat\n", i);
		return -1;
	}
	/* following order must be consistent with the one in the reader */
	fwrite(&plann->delay_product->l_max, sizeof(plann->delay_product->l_max), 1, fp);
	fwrite(&plann->delay_product->polar, sizeof(plann->delay_product->polar), 1, fp);
	fwrite(&plann->delay_product->n, sizeof(plann->delay_product->n), 1, fp);
	fwrite(&plann->delay_product->stride, sizeof(plann->delay_product->stride), 1, fp);
	fwrite(plann->delay_product->coeff, sizeof(*plann->delay_product->coeff), plann->delay_product->n * plann->delay_product->stride, fp);
	fclose(fp);

	/* write baseline */
	/* same information as correlator_network_baselines.  Then omit */

	/* write fseries_product */
	/* fseries_product is one of the results.  don't save. */

	/* write power_1d */
	/* power_1d is one of the results.  don't save. */

	return 0;
}


int write_precalc_correlator_network_plan_fd(const struct correlator_network_plan_fd *fdplansp, const struct correlator_network_plan_fd *fdplansn, char* parent_dir)
{
	/* write baselines */
	if(write_precalc_correlator_network_baselines(fdplansp->baselines, parent_dir)) {
		fprintf(stderr, "can't save correlator_network_baselines\n");
		return -1;
	}

	/* write plans */
	int i;
	for(i = 0; i < fdplansp->baselines->n_baselines; i++) {
		if(write_precalc_correlator_plan_fd(fdplansp->plans[i], fdplansn->plans[i], i, parent_dir)) {
			fprintf(stderr, "can't save %dth correlator_plan\n", i);
			return -1;
		}
	}

	return 0;
}


int write_precalc_autocorrelator_network_plan_fd(const struct autocorrelator_network_plan_fd *fdplansp, const struct autocorrelator_network_plan_fd *fdplansn, char* parent_dir)
{
	char filename[strlen(parent_dir) + FILE_LEN];
	FILE *fp;
	int i, k;

	/* write no pointer objects */
	sprintf(filename, "%s/autocorrelator_network_plan_fd/length.dat", parent_dir);
	fp = fopen(filename, "wb");
	if(!fp) {
		fprintf(stderr, "can't open length.dat\n");
		return -1;
	}
	fwrite(&fdplansp->length, sizeof(fdplansp->length), 1, fp);
	fclose(fp);

	/* write projections */
	/* positive case */
	for(i = 0; i < instrument_array_len(fdplansp->instruments); i++) {
		for(k = 0; k < fdplansp->length; k++) {
			sprintf(filename, "%s/autocorrelator_network_plan_fd/psh_series/%d/sh_series_p/projections%d.fits", parent_dir, i, k);
			if(sh_series_write_healpix_alm(fdplansp->projections[i][k], filename)) {
				fprintf(stderr, "can't save %dth projections%d.fits\n", i, k);
				return -1;
			}
		}
	}

	/* negative case */
	for(i = 0; i < instrument_array_len(fdplansn->instruments); i++) {
		for(k = 0; k < fdplansn->length; k++) {
			sprintf(filename, "%s/autocorrelator_network_plan_fd/psh_series/%d/sh_series_n/projections%d.fits", parent_dir, i, k);
			if(sh_series_write_healpix_alm(fdplansn->projections[i][k], filename)) {
				fprintf(stderr, "can't save %dth projections%d.fits\n", i, k);
				return -1;
			}
		}
	}

	/* instruments which is a member of correlator_baseline are given by
	 * command line options.  Then, we don't have to store it. */

	return 0;
}


int write_precalc_logprior(const struct sh_series *series, char *parent_dir)
{
	char filename[strlen(parent_dir) + FILE_LEN];
	sprintf(filename, "%s/sh_series/logprior.fits", parent_dir);
	return sh_series_write_healpix_alm(series, filename);
}


int write_precalc_time_series_length(unsigned int length, char *parent_dir)
{
	char filename[strlen(parent_dir) + FILE_LEN];
	FILE *fp;

	/* write time series length */
	sprintf(filename, "%s/time_series_length.dat", parent_dir);
	fp = fopen(filename, "wb");
	if(!fp) {
		fprintf(stderr, "can't open time_series_length.dat\n");
		return -1;
	}
	fwrite(&length, sizeof(length), 1, fp);
	fclose(fp);

	return 0;
}
