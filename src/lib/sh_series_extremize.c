/*
 * Copyright (C) 2026  Kipp C. Cannon
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


#include <assert.h>
#include <complex.h>
#include <float.h>
#include <math.h>
#include <gsl/gsl_multimin.h>
#include <gsl/gsl_vector.h>
#include <sphradiometer/sh_series.h>


/* FIXME: because of the repeated evaluations of Legendre polynomials, this
 * code would benefit from a multi-series evaluator API, something that
 * evaluates an array of functions at the same point returning an array of
 * their values.
 */


/*
 * ============================================================================
 *
 *                                  Helpers
 *
 * ============================================================================
 */


/*
 * the extremizer code doesn't know the rules for the co-ordinates and can
 * cross over wrap-around boundaries if the extremum is near one, so this
 * is used to sanitize co-ordinates before evaluating functions on the
 * sphere.  theta is put into [0, pi] and phi into [0, 2pi).  the values
 * are modified in-place.
 */


static void condition_theta_phi(double *theta, double *phi)
{
	/*fprintf(stderr, "condition_theta_phi(%.16g %.16g) --> ", *theta, *phi);*/
	/* unwind theta into [0, 2pi) */

	*theta = fmod(*theta, 2 * M_PI);
	if(*theta < 0.)
		*theta += 2 * M_PI;

	/* constrain theta to [0, pi], flipping phi to the other side of
	 * the sphere if needed */

	if(*theta > M_PI) {
		*theta = 2. * M_PI - *theta;
		*phi += M_PI;
	}

	/* finally unwind phi into [0, 2pi) */

	*phi = fmod(*phi, 2 * M_PI);
	if(*phi < 0.)
		*phi += 2 * M_PI;
	/*fprintf(stderr, "%.16g %.16g\n", *theta, *phi);*/

	assert(0. <= *theta && *theta <= M_PI);
	assert(0. <= *phi && *phi < 2. * M_PI);
}


/*
 * ============================================================================
 *
 *                   Wrapper Functions for Extremum Finding
 *
 * ============================================================================
 */


/*
 * params structure for use with GSL's "fdf" minimizers
 */


struct fdf_params {
	const struct sh_series *series;
	struct sh_series *u_sintheta;
	struct sh_series *v_sintheta;
};


/*
 * function and gradient evaluators to use GSL's "fdf" minimizers
 */


static double func(const gsl_vector *x, void *params)
{
	struct fdf_params *fdf_params = params;
	double theta = gsl_vector_get(x, 0);
	double phi = gsl_vector_get(x, 1);

	condition_theta_phi(&theta, &phi);

	double val = creal(sh_series_eval(fdf_params->series, theta, phi));

	/*fprintf(stderr, "func(%.16g, %.16g) = %.16g\n", theta, phi, val);*/

	return val;
}


static void dfunc(const gsl_vector *x, void *params, gsl_vector *g)
{
	struct fdf_params *fdf_params = params;
	double theta = gsl_vector_get(x, 0);
	double phi = gsl_vector_get(x, 1);

	condition_theta_phi(&theta, &phi);

	double sin_theta = sin(theta);

	double u = creal(sh_series_eval(fdf_params->u_sintheta, theta, phi)) / sin_theta;
	double v = creal(sh_series_eval(fdf_params->v_sintheta, theta, phi)) / sin_theta;

	/*fprintf(stderr, "grad dfunc(%.16g, %.16g) = %.16g theta + %.16g phi\n", theta, phi, u, v);*/

	gsl_vector_set(g, 0, u);
	gsl_vector_set(g, 1, v);
}


static void func_dfunc(const gsl_vector *x, void *params, double *f, gsl_vector *g)
{
	*f = func(x, params);
	dfunc(x, params, g);
}


/*
 * find a point near the minimum.  the input series is low-pass filtered
 * and transformed to the pixel domain where a brute-force search is used
 * to select the co-ordinates of the pixel with the minimum value.  the
 * result is hopefully within a wavelength of the minimum at full
 * bandwidth.
 */


static double find_near_minimum(const struct sh_series *series, unsigned l_lowpass, double *theta, double *phi)
{
	struct sh_series *low_bandwidth;
	/* needed to work around const'edness.  optimizer should remove */
	const struct sh_series *const_low_bandwidth;
	int ntheta, nphi;
	int i, j;
	double *cos_theta_array;
	complex double *mesh, *m;
	double min;

	/*
	 * for safety
	 */

	*theta = *phi = NAN;

	/*
	 * make a low-bandwidth version of the function
	 */

	if(series->l_max > l_lowpass) {
		low_bandwidth = sh_series_copy(series);
		if(!low_bandwidth)
			return NAN;
		sh_series_resize(low_bandwidth, l_lowpass);
		if(!low_bandwidth)
			return NAN;
		const_low_bandwidth = low_bandwidth;
	} else
		const_low_bandwidth = series;

	/*
	 * evaluate the low-bandwidth version on a mesh
	 */

	mesh = sh_series_to_mesh(const_low_bandwidth, &ntheta, &nphi, &cos_theta_array);
	if(const_low_bandwidth != series)
		sh_series_free(low_bandwidth);
	if(!mesh)
		return NAN;

	/*
	 * select the minimum/maximum Re(pixel)
	 */

	m = mesh;
	min = INFINITY;
	/*fprintf(stderr, "best guess %.16g @ %.16g %.16g\n", min, *theta, *phi);*/
	for(j = 0; j < ntheta; j++) {
		for(i = 0; i < nphi; i++) {
			double val = creal(*m++);
			if(val < min) {
				*theta = acos(cos_theta_array[j]);
				*phi = i * 2 * M_PI / nphi;
				min = val;
				/*fprintf(stderr, "best guess %.16g @ %.16g %.16g\n", min, *theta, *phi);*/
			}
		}
	}

	/*
	 * success
	 */

	free(mesh);
	free(cos_theta_array);

	return min;
}


/*
 * ============================================================================
 *
 *                                  Extrema
 *
 * ============================================================================
 */


/*
 * report the co-ordinates and value of the minimum of the real-valued
 * function on the sphere described by series.  on success the co-ordinates
 * are stored in the addresses pointed to by theta and phi, and the value
 * is returned.  on failure NaN is returned.
 *
 * NOTE:  this is a first attempt at this, it's not working very well, I'm
 * still experimenting.  use at your own risk.
 *
 * NOTE:  this uses GSL's implementation of the
 * Broyden–Fletcher–Goldfarb–Shanno (BFGS) algorithm, which computes an
 * iterative approximation of the inverse of the function's Hessian matrix.
 * I believe we might be able to obtain a closed form expression for that
 * matrix, and if so then this code might be improved by writing a custom
 * gradient descent algorithm that takes advantage of the explicit form of
 * the inverse Hessian.
 */


double sh_series_real_minimum(const struct sh_series *series, double *theta, double *phi)
{
	struct fdf_params fdf_params = {
		.series = series,
	};
	gsl_multimin_function_fdf fdf = {
		.f = func,
		.df = dfunc,
		.fdf = func_dfunc,
		.n = 2,
		.params = &fdf_params
	};
	double val;
	gsl_vector *x = gsl_vector_alloc(2);
	if(!x)
		return NAN;

	/*
	 * find the co-ordinates of the minimum of a low-bandwidth
	 * approximation of series.  this will be the starting point for
	 * the gradient descent
	 */

	if(isnan(find_near_minimum(series, 6, gsl_vector_ptr(x, 0), gsl_vector_ptr(x, 1)))) {
		gsl_vector_free(x);
		return NAN;
	}

	/*
	 * compute sin(theta) * grad of input series.  this completes the
	 * initialization of fdf_params.
	 */

	if(!sh_series_sintheta_grad(series, &fdf_params.u_sintheta, &fdf_params.v_sintheta)) {
		gsl_vector_free(x);
		return NAN;
	}

	/*
	 * initialize the minimizer
	 */

	gsl_multimin_fdfminimizer *extremizer = gsl_multimin_fdfminimizer_alloc(gsl_multimin_fdfminimizer_vector_bfgs2, 2);
	if(!extremizer) {
		gsl_vector_free(x);
		sh_series_free(fdf_params.u_sintheta);
		sh_series_free(fdf_params.v_sintheta);
		return NAN;
	}

	gsl_multimin_fdfminimizer_set(extremizer, &fdf, x, M_PI / 24., 0.001);
	gsl_vector_free(x);

	for(int iter = 0; iter < 100; iter++) {
		int status;

		status = gsl_multimin_fdfminimizer_iterate(extremizer);
		if(status) {
			fprintf(stderr, "iterator failed\n");
			break;
		}

		status = gsl_multimin_test_gradient(extremizer->gradient, 1e-3);
		if(status == GSL_SUCCESS) {
			fprintf(stderr, "iterator converged\n");
			break;
		}

		x = gsl_multimin_fdfminimizer_x(extremizer);
		*theta = gsl_vector_get(x, 0);
		*phi = gsl_vector_get(x, 1);
		val = gsl_multimin_fdfminimizer_minimum(extremizer);
		fprintf(stderr, "iter %d max %.16g @ %.16g %.16g\n", iter, val, *theta, *phi);
	}

	x = gsl_multimin_fdfminimizer_x(extremizer);
	*theta = gsl_vector_get(x, 0);
	*phi = gsl_vector_get(x, 1);
	val = gsl_multimin_fdfminimizer_minimum(extremizer);

	gsl_multimin_fdfminimizer_free(extremizer);
	sh_series_free(fdf_params.u_sintheta);
	sh_series_free(fdf_params.v_sintheta);

	return val;
}
