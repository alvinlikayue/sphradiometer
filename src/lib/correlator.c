/*
 * Copyright (C) 2006--2009,2012,2019  Kipp C. Cannon
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
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <fftw3.h>
#include <gsl/gsl_math.h>
#include <gsl/gsl_vector.h>
#include <sphradiometer/instrument.h>
#include <sphradiometer/projection.h>
#include <sphradiometer/correlator.h>


/*
 * ============================================================================
 *
 *                                 Utilities
 *
 * ============================================================================
 */


/*
 * Desired spherical harmonic order for cross power.
 */


unsigned int correlator_baseline_power_l_max_naive(double D, double delta_t)
{
	/* the expression below is the naive estimate, and in the Phys.
	 * Rev.  paper I reported using l_max + 2 to achieve the desired
	 * numerical accuracy.  since then it seems that improvements to
	 * numerical stability in other parts of the library have allowed
	 * this parameter to be reduced to the naive estimate, while
	 * retaining the desired accurancy in the correlator output */

	return floor(M_PI / (2. * asin(delta_t / (2. * D))) + 1.);
}


unsigned int correlator_baseline_power_l_max(const struct correlator_baseline *baseline, double delta_t)
{
	double D = vector_magnitude(baseline->d);

	if(delta_t <= 0. || D <= 0.)
		return (unsigned int) -1;

	return correlator_baseline_power_l_max_naive(D, delta_t);
}


/*
 * Suggested correlator dump interval, based on 1 Earth rotation in 86164
 * seconds (1 sidereal day).  The input is the maximum order of the
 * spherical harmonic expansion of the angular distribution of power, and
 * the number of times a snapshot (correlator dump) is desired as the Earth
 * rotates through one lobe.
 */


double correlator_dump_interval(unsigned int power_l_max, unsigned int dumps_per_lobe)
{
	const double seconds_per_day = 86164.0;
	const double radians_per_second = 2 * M_PI / seconds_per_day;
	const double radians_per_lobe = 2 * M_PI / power_l_max;
	const double seconds_per_lobe = radians_per_lobe / radians_per_second;
	const double seconds_per_dump = seconds_per_lobe / dumps_per_lobe;

	return seconds_per_dump;
}


/*
 * Correlation transient.
 */


int correlator_transient(const struct sh_series_array *delay_a, const struct sh_series_array *delay_b)
{
	const int na = delay_a->n;
	const int nb = delay_b->n;

	return ((na > nb ? na : nb) - 1) / 2;
}


/*
 * ============================================================================
 *
 *                                 Windowing
 *
 * ============================================================================
 */


/*
 * Construct a square window for a data series having n samples.  The first
 * and last zero_samples samples are set to 0, the rest are set to
 * amplitude.
 *
 * The integral of the window function is
 *
 * amplitude * (n - 2 * zero_samples)
 */


double *correlator_square_window_new(const int n, const int zero_samples, const double amplitude)
{
	double *series = malloc(n * sizeof(*series));
	int i;

	if(!series || (2 * zero_samples > n) || zero_samples < 0) {
		free(series);
		return NULL;
	}

	memset(series, 0, zero_samples * sizeof(*series));
	memset(series + n - zero_samples, 0, zero_samples * sizeof(*series));
	for(i = zero_samples; i < n - zero_samples; i++)
		series[i] = amplitude;

	return series;
}


/*
 * Construct a Tukey window for a data series having n samples.  In units
 * of amplitude, the window is 1 for the central portion of the data
 * series, 0 for the first and last zero_samples samples of the data
 * series, and the transitions between the two take taper_samples samples
 * each and are 1/4 cycles of a sine squared function (the first pi/2
 * radians for the off-to-on transition, and the second pi/2 radians for
 * the on-to-off transition).  The return value is a pointer to the input
 * series on success, NULL on failure (for example if the n, zero_samples,
 * and taper_samples parameters are incompatible).
 *
 * Precisely, for 0-origin sample number n, the window is
 *
 *        { 0, for 0 <= n < zero_samples
 * w(n) = { sin^2(pi/2 / taper_samples * (n - zero_samples)), for
 *                         zero_samples <= n < zero_samples + taper_samples
 *
 * and the mirror image of that for the end of the series.
 *
 * The integral of the window function is (FIXME: is this correct, and I
 * mean *to the sample*?  I think there's an off-by-one sample error in the
 * integral through the taper because the taper at the end is being
 * generated in reverse (which has to be done to ensure symmetry, and avoid
 * the window inducing a phase shift);  the correct term might be
 * -(taper_samples+1)).
 *
 * amplitude * (n - 2 * zero_samples - taper_samples)
 */


double *correlator_tukey_window_new(const int n, const int zero_samples, const int taper_samples, const double amplitude)
{
	double *series = malloc(n * sizeof(*series));
	const double dtheta = M_PI_2 / taper_samples;
	int i;

	if(!series || (2 * (zero_samples + taper_samples) > n) || zero_samples < 0) {
		free(series);
		return NULL;
	}

	memset(series, 0, zero_samples * sizeof(*series));
	memset(series + n - zero_samples, 0, zero_samples * sizeof(*series));
	for(i = 0; i < taper_samples; i++) {
		const double w = sin(dtheta * i);
		series[zero_samples + i] = series[n - 1 - zero_samples - i] = amplitude * w * w;
	}
	for(i = zero_samples + taper_samples; i < n - zero_samples - taper_samples; i++)
		series[i] = amplitude;

	return series;
}


/*
 * ============================================================================
 *
 *                             Fourier Transforms
 *
 * ============================================================================
 */


fftw_plan correlator_tseries_to_fseries_plan(double *tseries, complex double *fseries, int n)
{
	if(!tseries || !fseries || n <= 0)
		return NULL;
	return fftw_plan_dft_r2c_1d(n, tseries, fseries, FFTW_MEASURE);
}


void correlator_tseries_to_fseries(double *tseries, complex double *fseries, int n, fftw_plan plan)
{
	int i;

	fftw_execute(plan);
	for(i = 1; i < n / 2; i++)
		fseries[n - i] = conj(fseries[i]);
}


fftw_plan correlator_ctseries_to_fseries_plan(complex double *tseries, complex double *fseries, int n)
{
	return fftw_plan_dft_1d(n, tseries, fseries, FFTW_FORWARD, FFTW_MEASURE);
}


void correlator_ctseries_to_fseries(fftw_plan plan)
{
	fftw_execute(plan);
}


/*
 * ============================================================================
 *
 *                              Baseline Object
 *
 * ============================================================================
 */


/*
 * Create and destroy an object defining a baseline in the correlator.
 */


struct correlator_baseline *correlator_baseline_new(const struct instrument_array *instruments, int index_a, int index_b)
{
	struct correlator_baseline *new = malloc(sizeof(*new));
	gsl_vector *d = instrument_baseline(instrument_array_get(instruments, index_a), instrument_array_get(instruments, index_b));

	if(!new || !d) {
		gsl_vector_free(d);
		free(new);
		return NULL;
	}

	new->instruments = instruments;
	new->index_a = index_a;
	new->index_b = index_b;
	new->d = d;
	vector_direction(d, &new->theta, &new->phi);

	return new;
}


void correlator_baseline_free(struct correlator_baseline *baseline)
{
	if(baseline) {
		gsl_vector_free(baseline->d);
	}
	free(baseline);
}


struct correlator_baseline *correlator_baseline_copy(const struct correlator_baseline *baseline)
{
	struct correlator_baseline *new = malloc(sizeof(*new));

	if(!new)
		return NULL;

	*new = *baseline;
	new->d = gsl_vector_alloc(3);
	if(!new->d) {
		free(new);
		return NULL;
	}
	/* do not make a copy of the instrument array:  we don't own it,
	 * share the original pointer */
	gsl_vector_memcpy(new->d, baseline->d);

	return new;
}


/*
 * ============================================================================
 *
 *                              Correlation Plan
 *
 * ============================================================================
 */


/*
 * Time-domain correlation plan.
 */


struct correlator_plan_td *correlator_plan_td_new(const struct correlator_baseline *baseline, double delta_t)
{
	struct correlator_plan_td *new = malloc(sizeof(*new));
	gsl_vector *d_prime = gsl_vector_alloc(3);
	unsigned int a_l_max = projection_matrix_l_max(vector_magnitude(baseline->d) / 2, delta_t);
	unsigned int b_l_max = projection_matrix_l_max(vector_magnitude(baseline->d) / 2, delta_t);
	struct sh_series_array *proj_a = NULL;
	struct sh_series_array *proj_b = NULL;
	struct sh_series *sample_a = sh_series_new(a_l_max, 1);
	struct sh_series *sample_b = sh_series_new(b_l_max, 1);
	/* power_l_max need not excede a_l_max + b_l_max or we're wasting
	 * cpu cycles */
	unsigned int power_l_max = correlator_baseline_power_l_max(baseline, delta_t);
	if(power_l_max > a_l_max + b_l_max)
		power_l_max = a_l_max + b_l_max;
	struct sh_series *product = sh_series_new(power_l_max, 1);
	struct sh_series *power_1d = sh_series_new(power_l_max, 1);
	double d_length = vector_magnitude(baseline->d);
	double *R = sh_series_rot_matrix(baseline->theta, baseline->phi);
	struct sh_series_product_plan *product_plan = NULL;
	struct sh_series_rotation_plan *rotation_plan = NULL;

	if(!new || !d_prime || !sample_a || !sample_b || !product || !power_1d || !R || delta_t <= 0.)
		goto error;

	/* set d_prime to +d_length/2 * \hat{z}, and compute projection
	 * matrix */
	gsl_vector_set(d_prime, 0, 0);
	gsl_vector_set(d_prime, 1, 0);
	gsl_vector_set(d_prime, 2, +d_length / 2.);
	proj_a = projection_matrix_delay(projection_matrix_n_elements(vector_magnitude(d_prime), delta_t), a_l_max, d_prime, delta_t);

	/* set d_prime to -d_length/2 * \hat{z}, and compute projection
	 * matrix */
	gsl_vector_set(d_prime, 0, 0);
	gsl_vector_set(d_prime, 1, 0);
	gsl_vector_set(d_prime, 2, -d_length / 2.);
	proj_b = projection_matrix_delay(projection_matrix_n_elements(vector_magnitude(d_prime), delta_t), b_l_max, d_prime, delta_t);

	product_plan = sh_series_product_plan_new(power_1d, sample_a, sample_b);

	rotation_plan = sh_series_rotation_plan_new(power_1d, R);

	if(!proj_a || !proj_b || !product_plan || !rotation_plan)
		goto error;

	new->baseline = baseline;
	new->delta_t = delta_t;
	new->transient = correlator_transient(proj_a, proj_b);
	new->proj_a = proj_a;
	new->proj_b = proj_b;
	new->sample_a = sample_a;
	new->sample_b = sample_b;
	new->product = product;
	new->power_1d = power_1d;
	new->product_plan = product_plan;
	new->rotation_plan = rotation_plan;

	free(R);
	gsl_vector_free(d_prime);
	return new;

error:
	sh_series_rotation_plan_free(rotation_plan);
	sh_series_product_plan_free(product_plan);
	free(R);
	sh_series_free(power_1d);
	sh_series_free(product);
	sh_series_free(sample_b);
	sh_series_free(sample_a);
	sh_series_array_free(proj_b);
	sh_series_array_free(proj_a);
	gsl_vector_free(d_prime);
	free(new);
	return NULL;
}


void correlator_plan_td_free(struct correlator_plan_td *plan)
{
	if(plan) {
		sh_series_rotation_plan_free(plan->rotation_plan);
		sh_series_product_plan_free(plan->product_plan);
		sh_series_free(plan->power_1d);
		sh_series_free(plan->product);
		sh_series_free(plan->sample_b);
		sh_series_free(plan->sample_a);
		sh_series_array_free(plan->proj_b);
		sh_series_array_free(plan->proj_a);
	}
	free(plan);
}


/*
 * Frequency-domain correlation plan.
 */


struct correlator_plan_fd *correlator_plan_fd_new(const struct correlator_baseline *baseline, int n, double delta_t)
{
	struct correlator_plan_fd *new = malloc(sizeof(*new));
	/* use the TD plan constructor to do the work */
	struct correlator_plan_td *tdplan = correlator_plan_td_new(baseline, delta_t);
	complex double *fseries_product = malloc(n * sizeof(*fseries_product));
	struct sh_series_array *delay_product = NULL;
	complex double phase_a, phase_b;
	int i;

	if(!new || !tdplan || !fseries_product)
		goto error;

	delay_product = sh_series_array_new(n, tdplan->product->l_max, tdplan->product->polar);
	if(!delay_product)
		goto error;

	/* Fourier transform projection matrices.  First, zero-pad the
	 * matrices to match the input vector length, then forward
	 * transform */
	phase_a = I * 2 * M_PI * ((tdplan->proj_a->n - 1) / 2) / n;
	phase_b = I * 2 * M_PI * ((tdplan->proj_b->n - 1) / 2) / n;
	if(!sh_series_array_set_len(tdplan->proj_a, n))
		goto error;
	if(!sh_series_array_set_len(tdplan->proj_b, n))
		goto error;
	sh_series_array_forward_fft(tdplan->proj_a);
	sh_series_array_forward_fft(tdplan->proj_b);

	/* rotate the phases so that it is as though the elements were
	 * centred on 0.  Note that because the one vector will be
	 * complex-conjugated and then multiplied by the other, if the
	 * phase adjustement is the same for both then it need not be done
	 * at all since the phases will cancel out in the product. */
	if(phase_a != phase_b) {
		/* FIXME: do I have to handle the negative frequencies as
		 * negative frequencies, or can I treat them as >Nyquist
		 * positive frequencies?  It would simplify this stuff to
		 * just let the loop run up to n */
		for(i = 1; i < n / 2; i++) {
			sh_series_scale(&tdplan->proj_a->series[i], cexp(phase_a * i));
			sh_series_scale(&tdplan->proj_a->series[n - i], cexp(phase_a * -i));
			sh_series_scale(&tdplan->proj_b->series[i], cexp(phase_b * i));
			sh_series_scale(&tdplan->proj_b->series[n - i], cexp(phase_b * -i));
		}
		/* i = n / 2 */
		if(i == n - i) {
			/* then there is a Nyquist component */
			sh_series_scale(&tdplan->proj_a->series[i], cexp(phase_a * i));
			sh_series_scale(&tdplan->proj_b->series[i], cexp(phase_b * i));
		}
	}

	/* Compute and store their product.  Note that the "a" matrix is
	 * complex-conjugated so the "a" frequency series has to be
	 * conjugated in the correlator.  Note that the frequencies get
	 * inverted!  DC component is left in place, others are swapped,
	 * negative<-->positive */
	sh_series_conj(&tdplan->proj_a->series[0]);
	if(!sh_series_product(&delay_product->series[0], &tdplan->proj_a->series[0], &tdplan->proj_b->series[0], tdplan->product_plan))
		goto error;
	for(i = 1; i < n; i++) {
		sh_series_conj(&tdplan->proj_a->series[i]);
		if(!sh_series_product(&delay_product->series[n - i], &tdplan->proj_a->series[i], &tdplan->proj_b->series[i], tdplan->product_plan))
			goto error;
	}

	/* Apply one factor of 1/N to the delay product */
	sh_series_array_scale(delay_product, 1.0 / delay_product->n);

	/* Done */
	new->baseline = baseline;
	new->delta_t = delta_t;
	new->transient = tdplan->transient;
	new->delay_product = delay_product;
	new->fseries_product = fseries_product;
	new->power_1d = tdplan->power_1d;
	new->rotation_plan = tdplan->rotation_plan;

	/* Clean up */
	sh_series_product_plan_free(tdplan->product_plan);
	sh_series_free(tdplan->product);
	sh_series_free(tdplan->sample_b);
	sh_series_free(tdplan->sample_a);
	sh_series_array_free(tdplan->proj_b);
	sh_series_array_free(tdplan->proj_a);

	return new;

error:
	sh_series_array_free(delay_product);
	free(fseries_product);
	correlator_plan_td_free(tdplan);
	free(new);
	return NULL;
}


void correlator_plan_fd_free(struct correlator_plan_fd *plan)
{
	if(plan) {
		free(plan->fseries_product);
		sh_series_array_free(plan->delay_product);
		sh_series_free(plan->power_1d);
		sh_series_rotation_plan_free(plan->rotation_plan);
	}
	free(plan);
}


struct correlator_plan_fd *correlator_plan_fd_copy(const struct correlator_plan_fd *plan)
{
	struct correlator_plan_fd *new = malloc(sizeof(*new));
	if(!new)
		return NULL;

	*new = *plan;

	/* share baseline pointer, we don't own it */
	new->fseries_product = malloc(plan->delay_product->n * sizeof(*new->fseries_product));
	if(!new->fseries_product) {
		free(new);
		return NULL;
	}
	memcpy(new->fseries_product, plan->fseries_product, plan->delay_product->n * sizeof(*plan->fseries_product));
	new->delay_product = sh_series_array_copy(plan->delay_product);
	new->power_1d = sh_series_copy(plan->power_1d);
	new->rotation_plan = sh_series_rotation_plan_copy(plan->rotation_plan);
	if(!new->delay_product || !new->power_1d || !new->rotation_plan) {
		/* we have enough of it we can use this to clean up */
		correlator_plan_fd_free(new);
		return NULL;
	}

	return new;
}


/*
 * Reduce the l_max of the output of a frequency-domain correlator.  The
 * reduction is performed in-place.  Combine with correlator_plan_fd_copy()
 * to perserve the original.  Returns NULL on failure, or the plan object
 * on success.  On failure, the state of the correlator plan is
 * undefined (although it can be freed).
 */


struct correlator_plan_fd *correlator_plan_fd_set_l(struct correlator_plan_fd *plan, unsigned l_max)
{
	/* sanity check current state of the plan */
	if(plan->delay_product->l_max != plan->power_1d->l_max || plan->rotation_plan->l_max != plan->power_1d->l_max)
		return NULL;
	/* check for invalid request */
	if(l_max > plan->power_1d->l_max)
		return NULL;
	/* check for no-op */
	if(l_max == plan->power_1d->l_max)
		return plan;

	/* resize the components.  these conversions are done in-place.  if
	 * they fail, some leave their inputs undefined and some leave
	 * their inputs unchanged.  overall, our contents become undefined
	 * on failure. */
	if(!sh_series_array_set_l(plan->delay_product, l_max) || !sh_series_resize(plan->power_1d, l_max) || !sh_series_rotation_plan_set_l(plan->rotation_plan, l_max))
		return NULL;

	return plan;
}


/*
 * ============================================================================
 *
 *                                Correlation
 *
 * ============================================================================
 */


/*
 * Compute the angular distribution of coherent power for a baseline.
 * Following the successful completion of this function, the power_1d
 * element of the baseline object contains the azimuthally-symmetric power
 * distribution (distribution aligned with baseline's axis).  On success a
 * newly allocated sh_series object containing the power distribution on
 * the Earth-fixed sky is returned.  NULL is returned on failure.
 *
 * During integration, the cross power time series will be windowed
 * according to the window function.  The window function must have 2 *
 * baseline->transient fewer samples than the input time series, and will
 * be applied centred in the time series (i.e., the first sample of the
 * window function will be applied to the first computed cross power
 * sample, which is at sample # baseline->transient).  Typically the window
 * function will be normalized so that its integral is 1.  For example, to
 * not window at all, the window function should have (n - 2 *
 * baseline->transient) samples all set to 1.0 / (n - 2 *
 * baseline->transient).
 *
 * There is at least one important example in which the window function is
 * intentionally not normalized to have an integral of 1, and that is when
 * using the window function to implement smooth interpolation of the
 * integrand across joins when integrating a long time series by a sequence
 * of independant calls to this function.  The technique is as follows.
 * Each segment of the long time series is correlated applying a Tukey
 * window to the cross power time series.  A Tukey window has a sin^2
 * off-to-on transition, then a sin^2 on-to-off transition, so if two
 * adjacent Tukey windows whose peak values are 1 are overlaped correctly
 * their sum everywhere in the transition region is 1, all the while the
 * first is being smoothly turned off and the second is smoothly turned on.
 * If two calls to correlator_power() are made passing time series segments
 * that overlap by a number of samples given by 2 * baseline->transient
 * plus the length of the Tukey window transition, and the window functions
 * are set to have a peak value appropriate for a square window, then
 * adding the two results is equivalent to integrating a cross power time
 * series equal to the two integrands joined via a smooth interpolation
 * across the interface.  Because the window function is normalized for a
 * square window even though its ends are tapered, the integrated cross
 * power returned by each function call will be a little too small, but the
 * two function calls together integrate the data in the interface between
 * the two time series twice, with the net result being that the sum of the
 * two results is the correct total integrated cross power.
 */


struct sh_series *correlator_baseline_integrate_power_td(const double *time_series_a, const double *time_series_b, const double *window, int n, struct correlator_plan_td *plan)
{
	struct sh_series *power_2d;
	n -= 2 * plan->transient;
	if(n < 0)
		return NULL;

	sh_series_zero(plan->power_1d);

	while(n--) {
		if(!sh_series_array_dot(plan->sample_a, plan->proj_a, time_series_a++))
			return NULL;
		if(!sh_series_array_dot(plan->sample_b, plan->proj_b, time_series_b++))
			return NULL;
		if(!sh_series_product(plan->product, plan->sample_a, plan->sample_b, plan->product_plan))
			return NULL;
		if(!sh_series_add(plan->power_1d, *window++, plan->product))
			return NULL;
	}

	power_2d = sh_series_new(plan->power_1d->l_max, 0);
	if(!power_2d || !sh_series_rotate(power_2d, plan->power_1d, plan->rotation_plan))
		return NULL;

	return power_2d;
}


/*
 * Frequency domain version.
 */


struct sh_series *correlator_baseline_integrate_power_fd(const complex double *freq_series_a, const complex double *freq_series_b, struct correlator_plan_fd *plan)
{
	struct sh_series *power_2d;
	int i;

	/* multiply the two frequency series */
	for(i = 0; i < plan->delay_product->n; i++)
		plan->fseries_product[i] = *freq_series_b++ * conj(*freq_series_a++);

	/* compute the inner product of the frequency series and the DFT'ed
	 * delay matrix */
	if(!sh_series_array_dotc(plan->power_1d, plan->delay_product, plan->fseries_product))
		return NULL;

	/* FIXME:  to remove the correlator transient as in the time-domain
	 * case, or just generally apply a window, rather than computing
	 * the inner product directly, the products should be left
	 * un-summed (call _windowc() rather than _dotc()), the resulting
	 * array transformed to the time domain, and then windowed.  This
	 * is very costly. */

	/* normalize */
	sh_series_scale(plan->power_1d, 1.0 / plan->delay_product->n);

	/* rotate to Earth-fixed equatorial co-ordinates */
	power_2d = sh_series_new(plan->power_1d->l_max, 0);
	if(!power_2d || !sh_series_rotate(power_2d, plan->power_1d, plan->rotation_plan))
		return NULL;

	return power_2d;
}


struct sh_series *correlator_baseline_integrate_power_fd_window(const complex double *freq_series_a, const complex double *freq_series_b, const complex double *window, struct correlator_plan_fd *plan)
{
	struct sh_series *power_2d;
	struct sh_series_array *power_t;
	int i;

	if(!window)
		return correlator_baseline_integrate_power_fd(freq_series_a, freq_series_b, plan);

	/* multiply the two frequency series */
	for(i = 0; i < plan->delay_product->n; i++)
		plan->fseries_product[i] = *freq_series_b++ * conj(*freq_series_a++);

	/* Leave the frequency-bin contributions un-summed, transform the
	 * sky coefficients back to the cross-power time domain, and apply
	 * the requested geocentric-time window before summing. */
	power_t = sh_series_array_copy(plan->delay_product);
	if(!power_t)
		return NULL;
	if(!sh_series_array_windowc(power_t, plan->fseries_product)) {
		sh_series_array_free(power_t);
		return NULL;
	}
	if(!sh_series_array_reverse_fft(power_t)) {
		sh_series_array_free(power_t);
		return NULL;
	}
	if(!sh_series_array_dotc(plan->power_1d, power_t, window)) {
		sh_series_array_free(power_t);
		return NULL;
	}
	sh_series_array_free(power_t);

	/* Match the normalization used by the unwindowed FD correlator. */
	sh_series_scale(plan->power_1d, 1.0 / plan->delay_product->n);

	power_2d = sh_series_new(plan->power_1d->l_max, 0);
	if(!power_2d || !sh_series_rotate(power_2d, plan->power_1d, plan->rotation_plan))
		return NULL;

	return power_2d;
}


/*
 * ============================================================================
 *
 *                                  Network
 *
 * ============================================================================
 */


/*
 * A network of baselines
 */


struct correlator_network_baselines *correlator_network_baselines_new(const struct instrument_array *instruments)
{
	struct correlator_network_baselines *new = malloc(sizeof(*new));
	struct correlator_baseline **baselines = malloc(instrument_array_len(instruments) * (instrument_array_len(instruments) - 1) / 2 * sizeof(*baselines));
	int i, j, k;

	if(!new || !baselines) {
		free(new);
		free(baselines);
		return NULL;
	}

	k = 0;
	for(i = 1; i < instrument_array_len(instruments); i++)
		for(j = 0; j < i; j++, k++) {
			baselines[k] = correlator_baseline_new(instruments, i, j);
			if(!baselines[k]) {
				while(--k >= 0)
					correlator_baseline_free(baselines[k]);
				free(new);
				free(baselines);
				return NULL;
			}
		}

	new->baselines = baselines;
	new->n_baselines = instrument_array_len(instruments) * (instrument_array_len(instruments) - 1) / 2;

	return new;
}


void correlator_network_baselines_free(struct correlator_network_baselines *network)
{
	if(network) {
		int i;
		for(i = 0; i < network->n_baselines; i++)
			correlator_baseline_free(network->baselines[i]);
		/* do not free instruments pointer, we don't own it */
		free(network->baselines);
	}
	free(network);
}


struct correlator_network_baselines *correlator_network_baselines_copy(const struct correlator_network_baselines *network)
{
	struct correlator_network_baselines *new = malloc(sizeof(*new));
	int i;

	if(!new)
		return NULL;

	*new = *network;
	new->baselines = calloc(network->n_baselines, sizeof(*new->baselines));
	if(!new->baselines) {
		free(new);
		return NULL;
	}
	for(i = 0; i < network->n_baselines; i++) {
		new->baselines[i] = correlator_baseline_copy(network->baselines[i]);
		if(!new->baselines[i]) {
			/* have enough of it we can use this to clean up */
			correlator_network_baselines_free(new);
			return NULL;
		}
	}

	return new;
}


unsigned int correlator_network_l_max(struct correlator_network_baselines *network, double delta_t)
{
	unsigned int l_max = 0;
	int i;

	for(i = 0; i < network->n_baselines; i++) {
		const unsigned int l = correlator_baseline_power_l_max(network->baselines[i], delta_t);
		if(l > l_max)
			l_max = l;
	}

	return l_max;
}


/*
 * Time-domain correlation plan for a baseline network
 */


struct correlator_network_plan_td *correlator_network_plan_td_new(struct correlator_network_baselines *baselines, double delta_t)
{
	struct correlator_network_plan_td *new = malloc(sizeof(*new));
	struct correlator_plan_td **plans = malloc(baselines->n_baselines * sizeof(*plans));
	int i;

	if(!new || !plans) {
		free(new);
		free(plans);
		return NULL;
	}

	for(i = 0; i < baselines->n_baselines; i++) {
		plans[i] = correlator_plan_td_new(baselines->baselines[i], delta_t);
		if(!plans[i]) {
			while(--i >= 0)
				correlator_plan_td_free(plans[i]);
			free(new);
			free(plans);
			return NULL;
		}
	}

	new->baselines = baselines;
	new->plans = plans;

	return new;
}


void correlator_network_plan_td_free(struct correlator_network_plan_td *plan)
{
	if(plan) {
		int i;
		for(i = 0; i < plan->baselines->n_baselines; i++)
			correlator_plan_td_free(plan->plans[i]);
	}
	free(plan);
}


/*
 * Frequency-domain correlation plan for a baseline network
 */


struct correlator_network_plan_fd *correlator_network_plan_fd_new(struct correlator_network_baselines *baselines, int tseries_length, double delta_t)
{
	struct correlator_network_plan_fd *new = malloc(sizeof(*new));
	struct correlator_plan_fd **plans = malloc(baselines->n_baselines * sizeof(*plans));
	int i;

	if(!new || !plans) {
		free(new);
		free(plans);
		return NULL;
	}

	for(i = 0; i < baselines->n_baselines; i++) {
		plans[i] = correlator_plan_fd_new(baselines->baselines[i], tseries_length, delta_t);
		if(!plans[i]) {
			while(--i >= 0)
				correlator_plan_fd_free(plans[i]);
			free(new);
			free(plans);
			return NULL;
		}
	}

	new->baselines = baselines;
	new->plans = plans;
	new->swap_frequency_product = 0;

	return new;
}


void correlator_network_plan_fd_free(struct correlator_network_plan_fd *plan)
{
	if(plan) {
		int i;
		for(i = 0; i < plan->baselines->n_baselines; i++)
			correlator_plan_fd_free(plan->plans[i]);
		free(plan->plans);
	}
	free(plan);
}


struct correlator_network_plan_fd *correlator_network_plan_fd_copy(const struct correlator_network_plan_fd *plan)
{
	struct correlator_network_plan_fd *new = malloc(sizeof(*new));
	int i;

	if(!new)
		return NULL;

	/* do not duplicate baselines, share the pointer, we don't own it */
	*new = *plan;
	/* allocate a new set of plans for the baselines */
	new->plans = calloc(plan->baselines->n_baselines, sizeof(*plan->plans));
	if(!new->plans) {
		free(new);
		return NULL;
	}

	for(i = 0; i < plan->baselines->n_baselines; i++) {
		new->plans[i] = correlator_plan_fd_copy(plan->plans[i]);
		if(!new->plans[i]) {
			/* have enough of it we can use this for clean up */
			correlator_network_plan_fd_free(new);
			return NULL;
		}
	}

	return new;
}


/*
 * Reduce the l_max of a correlator network.  Any baselines whose l_max are
 * already below the requested value are unchanged (their l_max is not
 * increased to match the requested value).  The conversion is done
 * in-place.  Combine with correlator_network_plan_fd_copy() for an
 * out-of-place conversion.  Returns the network plan object on success, or
 * NULL on failure.  On failure, the network plan's contents are undefined,
 * and it should be disposed of.  It cannot be used.
 */


struct correlator_network_plan_fd *correlator_network_plan_fd_set_l(struct correlator_network_plan_fd *plan, unsigned l_max)
{
	int i;
	for(i = 0; i < plan->baselines->n_baselines; i++) {
		if(plan->plans[i]->power_1d->l_max <= l_max)
			continue;
		if(!correlator_plan_fd_set_l(plan->plans[i], l_max))
			return NULL;
	}
	return plan;
}


struct correlator_network_plan_fd *correlator_network_plan_fd_set_swap_frequency_product(struct correlator_network_plan_fd *plan, int swap_frequency_product)
{
	plan->swap_frequency_product = swap_frequency_product ? 1 : 0;
	return plan;
}


struct correlator_network_plan_fd *correlator_network_plan_fd_set_frame_gmst(struct correlator_network_plan_fd *plan, double frame_gmst)
{
	int i;

	for(i = 0; i < plan->baselines->n_baselines; i++) {
		const struct correlator_baseline *baseline = plan->baselines->baselines[i];
		double *R = sh_series_rot_matrix(baseline->theta, baseline->phi + frame_gmst);
		struct sh_series_rotation_plan *rotation_plan;

		if(!R)
			return NULL;
		rotation_plan = sh_series_rotation_plan_new(plan->plans[i]->power_1d, R);
		free(R);
		if(!rotation_plan)
			return NULL;
		sh_series_rotation_plan_free(plan->plans[i]->rotation_plan);
		plan->plans[i]->rotation_plan = rotation_plan;
	}

	return plan;
}


/*
 * Time-domain network correlator
 */


struct sh_series *correlator_network_integrate_power_td(struct sh_series *sky, double **tseries, int tseries_length, double **windows, struct correlator_network_plan_td *plan)
{
	int i, j, k;

	sh_series_zero(sky);
	k = 0;
	for(i = 1; i < instrument_array_len(plan->baselines->baselines[0]->instruments); i++)
		for(j = 0; j < i; j++, k++) {
			struct sh_series *power_2d = correlator_baseline_integrate_power_td(tseries[i], tseries[j], windows[k], tseries_length, plan->plans[k]);
			if(!power_2d)
				return NULL;
			if(!sh_series_add(sky, 1.0 / plan->baselines->n_baselines, power_2d))
				return NULL;
			sh_series_free(power_2d);
		}

	return sky;
}



/*
 * Frequency-domain network correlator
 */


static struct sh_series *correlator_network_integrate_power_fd_internal(struct sh_series *sky, complex double **fseries, const complex double *window, struct correlator_network_plan_fd *plan)
{
	int i, j, k;

	sh_series_zero(sky);
	k = 0;
	for(i = 1; i < instrument_array_len(plan->baselines->baselines[0]->instruments); i++)
		for(j = 0; j < i; j++, k++) {
			struct sh_series *power_2d;
			const complex double *a = plan->swap_frequency_product ? fseries[j] : fseries[i];
			const complex double *b = plan->swap_frequency_product ? fseries[i] : fseries[j];
			if(window)
				power_2d = correlator_baseline_integrate_power_fd_window(a, b, window, plan->plans[k]);
			else
				power_2d = correlator_baseline_integrate_power_fd(a, b, plan->plans[k]);
			if(!power_2d)
				return NULL;
			if(!sh_series_add(sky, 1.0 / plan->baselines->n_baselines, power_2d))
				return NULL;
			sh_series_free(power_2d);
		}

	return sky;
}


struct sh_series *correlator_network_integrate_power_fd(struct sh_series *sky, complex double **fseries, struct correlator_network_plan_fd *plan)
{
	return correlator_network_integrate_power_fd_internal(sky, fseries, NULL, plan);
}


struct sh_series *correlator_network_integrate_power_fd_window(struct sh_series *sky, complex double **fseries, const complex double *window, struct correlator_network_plan_fd *plan)
{
	if(!window)
		return correlator_network_integrate_power_fd(sky, fseries, plan);
	return correlator_network_integrate_power_fd_internal(sky, fseries, window, plan);
}
