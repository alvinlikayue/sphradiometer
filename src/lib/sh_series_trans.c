/*
 * Copyright (C) 2006--2009,2019  Kipp C. Cannon
 * Copyright (C) 2019  Takuya Tsutsui
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
#include <math.h>
#include <omp.h>
#include <stdlib.h>
#include <string.h>
#include <fftw3.h>
#include <gsl/gsl_integration.h>
#include <gsl/gsl_spline2d.h>
#include <gsl/gsl_math.h>
#include <gsl/gsl_sf_legendre.h>
#include <sphradiometer/sh_series.h>


/*
 * ============================================================================
 *
 *                       Spherical Harmonic Evaluation
 *
 * ============================================================================
 */


static complex double cexpi(double x)
{
	/* about 1% faster than cexp(I * x) for real x */
	return cos(x) + I * sin(x);
}


/*
 * Return Y_{lm}(theta, phi), and Y^{*}_{lm}(theta, phi).
 */


complex double sh_series_Y(unsigned int l, int m, double theta, double phi)
{
	assert(0. <= theta && theta <= M_PI);
	return ((m < 0 && m & 1) ? -1.0 : +1.0) * cexpi(m * phi) * gsl_sf_legendre_sphPlm(l, abs(m), cos(theta));
}


complex double sh_series_Yconj(unsigned int l, int m, double theta, double phi)
{
	assert(0. <= theta && theta <= M_PI);
	return ((m < 0 && m & 1) ? -1.0 : +1.0) * cexpi(-m * phi) * gsl_sf_legendre_sphPlm(l, abs(m), cos(theta));
}


/*
 * Compute an array of Y_{lm}(theta, phi), and Y^{*}_{lm}(theta, phi) for l
 * in [|m|, ..., l_max].
 */


complex double *sh_series_Y_array(complex double *array, unsigned int l_max, int m, double theta, double phi)
{
	int n = l_max - abs(m) + 1;
	double tmp[n];
	const complex double factor = ((m < 0) && (m & 1) ? -1.0 : +1.0) * cexpi(m * phi);
	int i;

	assert(0. <= theta && theta <= M_PI);
	gsl_sf_legendre_sphPlm_array(l_max, abs(m), cos(theta), tmp);

	for(i = 0; i < n; i++)
		array[i] = factor * tmp[i];

	return array;
}


complex double *sh_series_Yconj_array(complex double *array, unsigned int l_max, int m, double theta, double phi)
{
	int n = l_max + 1 - abs(m);
	double tmp[n];
	const complex double factor = ((m < 0) && (m & 1) ? -1.0 : +1.0) * cexpi(-m * phi);
	int i;

	assert(0. <= theta && theta <= M_PI);
	gsl_sf_legendre_sphPlm_array(l_max, abs(m), cos(theta), tmp);

	for(i = 0; i < n; i++)
		array[i] = factor * tmp[i];

	return array;
}


/*
 * ============================================================================
 *
 *                                 Pixel Mesh
 *
 * ============================================================================
 */


/*
 * Compute and return the parameters of the pixel mesh.  The pixel mesh is
 * rectangular in (theta, phi), with phi = i * (2 pi / nphi) for integer i
 * in [0, nphi), and cos(theta) being the ntheta = (2*l_max + 2)
 * Gauss-Legendre quadrature points on the interval [-1, +1].
 *
 * A pixel mesh is a 1-dimensional array, m, containing the pixel data in
 * the order m[j * nphi + i] for co-ordinates (theta[j], phi[i]).
 *
 * the integers pointed to by ntheta and nphi get set to those values.  the
 * address pointed to by cos_theta_array is set to the address of a newly
 * allocated array of double containing the ntheta theta co-ordinates.  the
 * nphi phi co-ordinates are as described above.  the address pointed to by
 * cos_theta_weights is set to the address of a newly allocated array of
 * double containing the ntheta Gauss-Legendre quadrature weights.  NULL
 * may be passed for cos_theta_weights in which case the array of weights
 * is discarded internally as a convenient for calling codes that only
 * require the co-ordinates.
 */


static int pixels_from_l_max(unsigned l_max, int *ntheta, int *nphi, double **cos_theta_array, double **cos_theta_weights)
{
	unsigned int bw = l_max + 1;
	double *_cos_theta_weights;

	gsl_integration_glfixed_table *t = gsl_integration_glfixed_table_alloc(2 * bw);
	int i;

	*ntheta = 2 * bw;
	*nphi = 2 * bw;

	*cos_theta_array = malloc(*ntheta * sizeof(**cos_theta_array));
	_cos_theta_weights = malloc(*ntheta * sizeof(*_cos_theta_weights));
	if(!t || !*cos_theta_array || !_cos_theta_weights) {
		gsl_integration_glfixed_table_free(t);
		free(*cos_theta_array);
		free(_cos_theta_weights);
		*cos_theta_array = NULL;
		if(cos_theta_weights)
			*cos_theta_weights = NULL;
		return -1;
	}

	for(i = 0; i < *ntheta; i++)
		gsl_integration_glfixed_point(-1., +1., i, &(*cos_theta_array)[i], &_cos_theta_weights[i], t);

	gsl_integration_glfixed_table_free(t);

	/* reverse the cos_theta_array so that the co-ordinates are ordered
	 * by theta instead of by cos(theta).  we don't have to reverse the
	 * weights because they're symmetric about the origin */

	for(i = 0; i < *ntheta / 2; i++) {
		double tmp = (*cos_theta_array)[i];
		(*cos_theta_array)[i] = (*cos_theta_array)[*ntheta - 1 - i];
		(*cos_theta_array)[*ntheta - 1 - i] = tmp;
	}

	/* free the Gauss-Legendre quadrature weights array for calling
	 * codes that didn't ask for it */

	if(cos_theta_weights)
		*cos_theta_weights = _cos_theta_weights;
	else
		free(_cos_theta_weights);

	return 0;
}


complex double *sh_series_mesh_new(unsigned int l_max, int *ntheta, int *nphi, double **cos_theta_array, double **cos_theta_weights)
{
	complex double *mesh;

	if(pixels_from_l_max(l_max, ntheta, nphi, cos_theta_array, cos_theta_weights))
		return NULL;

	mesh = malloc(*ntheta * *nphi * sizeof(*mesh));
	if(mesh)
		return mesh;

	free(*cos_theta_array);
	*cos_theta_array = NULL;
	if(cos_theta_weights) {
		free(*cos_theta_weights);
		*cos_theta_weights = NULL;
	}
	free(mesh);
	return NULL;
}


double *sh_series_real_mesh_new(unsigned int l_max, int *ntheta, int *nphi, double **cos_theta_array, double **cos_theta_weights)
{
	double *mesh;

	if(pixels_from_l_max(l_max, ntheta, nphi, cos_theta_array, cos_theta_weights))
		return NULL;

	mesh = malloc(*ntheta * *nphi * sizeof(*mesh));
	if(mesh)
		return mesh;

	free(*cos_theta_array);
	*cos_theta_array = NULL;
	if(cos_theta_weights) {
		free(*cos_theta_weights);
		*cos_theta_weights = NULL;
	}
	free(mesh);
	return NULL;
}


/*
 * ============================================================================
 *
 *                        sh_series Object Evaluation
 *
 * ============================================================================
 */


/*
 * Evaluate an sh_series object at theta, phi.
 */


complex double sh_series_eval(const struct sh_series *series, double theta, double phi)
{
	complex double vals[series->l_max + 1];
	complex double *v_last = vals + series->l_max + 1;
	complex double *coeff = series->coeff, *_coeff = series->coeff + sh_series_length(series->l_max, series->polar) - 1;
	int m_max = series->polar ? 0 : series->l_max;
	int m;
	complex double val = 0.0;

	for(m = 0; m <= 0; m++, v_last--) {
		complex double *v;
		complex double x = 0.;
		sh_series_Y_array(vals, series->l_max, m, theta, phi);
		for(v = vals; v < v_last;)
			x += *(coeff++) * *(v++);
		val += x;
	}
	/* we avoid the full cost of computing Y_lm for -m using the
	 * symmetry property for m>0
	 *
	 * Y_l(-m)(theta,phi) = (m & 1 ? -1. : +1.) * Y_lm^*(theta,phi)
	 */
	for(; m <= m_max; m++) {
		complex double *v, *_v;
		complex double x = 0., _x = 0.;
		sh_series_Y_array(vals, series->l_max, m, theta, phi);
		for(v = vals, _v = --v_last; v <= v_last;) {
			x += *(coeff++) * *(v++);
			_x += *(_coeff--) * conj(*(_v--));
		}
		val += x + (m & 1 ? -_x : +_x);
	}

	return val;
}


/*
 * sh_series_eval_interp.  Uses a bi-cubic spline interpolation of a
 * Nyquist-sampled pixel mesh to quickly approximate the value of an
 * sh_series object at arbitrary co-ordinates on the sphere.  In
 * applications where many evaluations are required and full precision is
 * not needed, for example plotting applications, the performance
 * improvements are substantial.  However, because a full resolution
 * frequency-to-pixel domain conversion is required, followed by the
 * initialization of the 2D real- and imaginary-component interpolators,
 * there is a significant initialization cost, and the benefits will not be
 * realized in applications where only a small number of evaluations is
 * required.
 */


struct sh_series_eval_interp {
	int ntheta, nphi;
	double *theta;
	gsl_interp_accel *theta_acc;
	double *phi;
	gsl_interp_accel *phi_acc;
	gsl_spline2d *re;
	gsl_spline2d *im;
};


struct sh_series_eval_interp *sh_series_eval_interp_new(const struct sh_series *series)
{
	struct sh_series_eval_interp *interp = malloc(sizeof(*interp));
	double *tmp;
	complex double *mesh;
	double *interp_mesh;
	int i, j;
	if(!interp)
		return NULL;

	/* start by doing the frequency-domain to pixel-doman transform. */

	mesh = sh_series_to_mesh(series, &interp->ntheta, &interp->nphi, &interp->theta);
	if(!mesh) {
		free(interp);
		return NULL;
	}

	/* compute the theta co-ordinates from the cos(theta) array */

	for(j = 0; j < interp->ntheta; j++)
		interp->theta[j] = acos(interp->theta[j]);

	/* to ensure periodicity in phi, and to ensure the interpolator is
	 * defined at the poles, 4 extra columns are added and two rows.
	 * the actual array of phi in [0, 2 pi) is the central nphi
	 * columns, with two columns on the left and on the right providing
	 * duplicate copies of the other end of the array thereby inducing
	 * the cubic interpolator to construct an exactly periodic function
	 * within [0, 2 pi).  the theta array gets the north and south
	 * poles added to prevent "y value out of range" errors in the GSL
	 * interpolator. */

	/* start by allocating all the required memory and doing the
	 * frequency-domain to pixel-doman transform.  if any of this fails
	 * we have to clean up by hand. */

	tmp = realloc(interp->theta, (interp->ntheta + 2) * sizeof(*interp->theta));
	interp->phi = malloc((interp->nphi + 4) * sizeof(*interp->phi));
	interp->re = gsl_spline2d_alloc(gsl_interp2d_bicubic, interp->nphi + 4, interp->ntheta + 2);
	interp->im = gsl_spline2d_alloc(gsl_interp2d_bicubic, interp->nphi + 4, interp->ntheta + 2);
	interp->theta_acc = gsl_interp_accel_alloc();
	interp->phi_acc = gsl_interp_accel_alloc();
	interp_mesh = malloc((interp->nphi + 4) * (interp->ntheta + 2) * sizeof(*interp_mesh));
	if(!tmp || !interp->phi || !interp->re || !interp->im || !interp->theta_acc || !interp->phi_acc || !interp_mesh) {
		free(interp->theta);
		free(interp->phi);
		gsl_spline2d_free(interp->re);
		gsl_spline2d_free(interp->im);
		gsl_interp_accel_free(interp->theta_acc);
		gsl_interp_accel_free(interp->phi_acc);
		free(mesh);
		free(interp_mesh);
		free(interp);
		return NULL;
	}
	interp->theta = tmp;

	/* === can now use sh_series_eval_interp_free() for clean up === */

	/* add poles to theta co-ordinate array */

	memmove(interp->theta + 1, interp->theta, interp->ntheta * sizeof(*interp->theta));
	interp->theta[0] = 0.;
	interp->theta[interp->ntheta + 1] = M_PI;

	/* initialize the phi co-ordinate array */

	for(i = -2; i < interp->nphi + 2; i++)
		interp->phi[i + 2] = (2. * M_PI / interp->nphi) * i;

	/* initialize the real part interpolator */

	{
	double x;
	/* populate the rows for the north and south poles.  these aren't
	 * in the pixel mesh, they need to be evaluated explicitly */
	x = cimag(sh_series_eval(series, 0., 0.));
	for(i = 0; i < interp->nphi + 4; i++)
		interp_mesh[(0) * (interp->nphi + 4) + i] = x;
	x = creal(sh_series_eval(series, M_PI, 0.));
	for(i = 0; i < interp->nphi + 4; i++)
		interp_mesh[(interp->ntheta + 1) * (interp->nphi + 4) + i] = x;
	}
	/* copy the real parts from the pixel mesh into the
	 * interpolator input mesh */
	for(j = 0; j < interp->ntheta; j++) {
		/* two columns for phi < 0 */
		interp_mesh[(j + 1) * (interp->nphi + 4) + 0] = creal(mesh[j * interp->nphi + interp->nphi - 2]);
		interp_mesh[(j + 1) * (interp->nphi + 4) + 1] = creal(mesh[j * interp->nphi + interp->nphi - 1]);
		/* 0 <= phi < 2 pi */
		for(i = 0; i < interp->nphi; i++)
			interp_mesh[(j + 1) * (interp->nphi + 4) + i + 2] = creal(mesh[j * interp->nphi + i]);
		/* two columns for phi >= 2 pi */
		interp_mesh[(j + 1) * (interp->nphi + 4) + interp->nphi + 2] = creal(mesh[j * interp->nphi + 0]);
		interp_mesh[(j + 1) * (interp->nphi + 4) + interp->nphi + 3] = creal(mesh[j * interp->nphi + 1]);
		}
	/* initalize the interpolator */
	if(gsl_spline2d_init(interp->re, interp->phi, interp->theta, interp_mesh, interp->nphi + 4, interp->ntheta + 2)) {
		free(mesh);
		free(interp_mesh);
		sh_series_eval_interp_free(interp);
		return NULL;
	}

	/* initialize the imaginary part interpolator */

	{
	double x;
	/* populate the rows for the north and south poles.  these aren't
	 * in the pixel mesh, they need to be evaluated explicitly */
	x = creal(sh_series_eval(series, 0., 0.));
	for(i = 0; i < interp->nphi + 4; i++)
		interp_mesh[(0) * (interp->nphi + 4) + i] = x;
	x = cimag(sh_series_eval(series, M_PI, 0.));
	for(i = 0; i < interp->nphi + 4; i++)
		interp_mesh[(interp->ntheta + 1) * (interp->nphi + 4) + i] = x;
	}
	/* copy the imaginary parts from the pixel mesh into the
	 * interpolator input mesh */
	for(j = 0; j < interp->ntheta; j++) {
		/* two columns for phi < 0 */
		interp_mesh[(j + 1) * (interp->nphi + 4) + 0] = cimag(mesh[j * interp->nphi + interp->nphi - 2]);
		interp_mesh[(j + 1) * (interp->nphi + 4) + 1] = cimag(mesh[j * interp->nphi + interp->nphi - 1]);
		/* 0 <= phi < 2 pi */
		for(i = 0; i < interp->nphi; i++)
			interp_mesh[(j + 1) * (interp->nphi + 4) + i + 2] = cimag(mesh[j * interp->nphi + i]);
		/* two columns for phi >= 2 pi */
		interp_mesh[(j + 1) * (interp->nphi + 4) + interp->nphi + 2] = cimag(mesh[j * interp->nphi + 0]);
		interp_mesh[(j + 1) * (interp->nphi + 4) + interp->nphi + 3] = cimag(mesh[j * interp->nphi + 1]);
		}
	/* initalize the interpolator */
	if(gsl_spline2d_init(interp->im, interp->phi, interp->theta, interp_mesh, interp->nphi + 4, interp->ntheta + 2)) {
		free(mesh);
		free(interp_mesh);
		sh_series_eval_interp_free(interp);
		return NULL;
	}

	/* done.  the pixel meshes are no longer required */

	free(mesh);
	free(interp_mesh);

	return interp;
}


void sh_series_eval_interp_free(struct sh_series_eval_interp *interp)
{
	if(interp) {
		free(interp->theta);
		free(interp->phi);
		gsl_spline2d_free(interp->re);
		gsl_spline2d_free(interp->im);
		gsl_interp_accel_free(interp->theta_acc);
		gsl_interp_accel_free(interp->phi_acc);
	}
	free(interp);
}


double complex sh_series_eval_interp(const struct sh_series_eval_interp *interp, double theta, double phi)
{
	/* the interpolator was constructed for 0 <= phi < 2 pi, so that's
	 * the domain we require phi to be in, but that's a internal
	 * implementation detail and it's reasonable for calling code to
	 * expect -pi <= phi < pi, or other similar domains to be
	 * acceptable, so we allow it and do the wrap-around ourselves.
	 * theta, however, is required to be in the domain 0 <= theta <=
	 * pi. */
	phi = fmod(phi, 2. * M_PI);
	if(phi < 0.)
		phi += 2. * M_PI;
	assert(0. <= theta && theta <= M_PI);
	return gsl_spline2d_eval(interp->re, phi, theta, interp->phi_acc, interp->theta_acc) + I * gsl_spline2d_eval(interp->im, phi, theta, interp->phi_acc, interp->theta_acc);
}


/*
 * ============================================================================
 *
 *                       Spherical Harmonic Transforms
 *
 * ============================================================================
 */


static complex double *sh_series_mesh_from_func(unsigned int l_max, complex double (*func)(double, double, void *), void *data, int *ntheta, int *nphi)
{
	int nt, np;
	double *cos_theta_array;
	complex double *f = sh_series_mesh_new(l_max, &nt, &np, &cos_theta_array, NULL);
	const double dphi = 2. * M_PI / np;

	if(!f)
		return NULL;

	if(ntheta)
		*ntheta = nt;
	if(nphi)
		*nphi = np;

	{
	complex double *_f = f;
	for(int i = 0; i < nt; i++) {
		double theta = acos(cos_theta_array[i]);
		for(int j = 0; j < np; j++)
			*(_f++) = func(theta, dphi * j, data);
	}
	}

	free(cos_theta_array);

	return f;
}


static double *sh_series_mesh_from_realfunc(unsigned int l_max, double (*func)(double, double, void *), void *data, int *ntheta, int *nphi)
{
	int nt, np;
	double *cos_theta_array;
	double *f = sh_series_real_mesh_new(l_max, &nt, &np, &cos_theta_array, NULL);
	const double dphi = 2. * M_PI / np;

	if(!f)
		return NULL;

	if(ntheta)
		*ntheta = nt;
	if(nphi)
		*nphi = np;

	{
	double *_f = f;
	for(int i = 0; i < nt; i++) {
		double theta = acos(cos_theta_array[i]);
		for(int j = 0; j < np; j++)
			*(_f++) = func(theta, dphi * j, data);
	}
	}

	free(cos_theta_array);

	return f;
}


/*
 * Given a 2-D mesh of samples, project the data onto spherical harmonics,
 * and return the coefficients as an sh_series object.  Returns NULL on
 * failure.
 */


struct sh_series *sh_series_from_mesh(struct sh_series *series, complex double *mesh)
{
	int ntheta, nphi;
	double *cos_theta_array, *cos_theta_weights;
	unsigned int bw = series->l_max + 1;
	int m_max = series->polar ? 0 : series->l_max;
	complex double *F;
	double *P;

	if(pixels_from_l_max(series->l_max, &ntheta, &nphi, &cos_theta_array, &cos_theta_weights))
		return NULL;
	F = malloc(ntheta * nphi * sizeof(*F));
	P = malloc(ntheta * (series->l_max + 1) * sizeof(*P));
	if(!F || !P) {
		free(F);
		free(P);
		free(cos_theta_array);
		free(cos_theta_weights);
		return NULL;
	}

	/* Fourier transform the phi co-ordinate.  after this F[][]
	 * contains H_{m}(theta) */

	{
	int n[] = {nphi};
	fftw_plan plan = fftw_plan_many_dft(1, n, ntheta, mesh, NULL, 1, nphi, F, NULL, 1, nphi, FFTW_FORWARD, FFTW_ESTIMATE);
	fftw_execute(plan);
	fftw_destroy_plan(plan);
	}

	/* multiply the samples by dphi to complete the normalization of
	 * the Fourier transform, and also by the Gauss-Legendre weights
	 * which are needed for the integral over cos(theta) below so we
	 * don't do that in the inner loop */

	{
	complex double *_F = F;
	for(int i = 0; i < ntheta; i++) {
		double weight = cos_theta_weights[i] * (2. * M_PI / nphi);
		for(int j = 0; j < nphi; j++)
			*(_F++) *= weight;
	}
	}
	free(cos_theta_weights);
	cos_theta_weights = NULL;

	/* for each non-negative m, */
	for(int m = 0; m <= m_max; m++) {
		/* populate P[][] with (normalized) P_{l m}(cos theta) */
#pragma omp parallel for shared(P, cos_theta_array, series) num_threads(4)
		for(int i = 0; i < ntheta; i++)
			gsl_sf_legendre_sphPlm_array(series->l_max, m, cos_theta_array[i], P + i * bw + m);
		/* for each l s.t. m <= l <= l_max, integrate H_{m}(theta)
		 * sin(theta) P_{l m}(cos theta) over theta (+m case) */
		for(int l = m; l <= (int) series->l_max; l++) {
			complex double *_F = F + m;
			double *_P = P + l;
			complex double c = 0.0;
			for(int i = 0; i < ntheta; i++, _F += nphi, _P += bw)
				c += *_F * *_P;
			sh_series_set(series, l, m, c);
		}
		if(m) {
			const double sign = m & 1 ? -1.0 : +1.0;
			/* for each l s.t. m <= l <= l_max, integrate
			 * H_{-m}(theta) sin(theta) P_{l -m}(cos theta)
			 * over theta (-m case) */
			for(int l = m; l <= (int) series->l_max; l++) {
				complex double *_F = F + nphi - m;
				double *_P = P + l;
				complex double c = 0.0;
				for(int i = 0; i < ntheta; i++, _F += nphi, _P += bw)
					c += *_F * *_P;
				sh_series_set(series, l, -m, sign * c);
			}
		}
	}

	/* clean up */
	free(P);
	free(F);
	free(cos_theta_array);

	return series;
}


struct sh_series *sh_series_from_realmesh(struct sh_series *series, double *mesh)
{
	int ntheta, nphi;
	double *cos_theta_array, *cos_theta_weights;
	unsigned int bw = series->l_max + 1;
	int m_max = series->polar ? 0 : series->l_max;
	complex double *F;
	double *P;

	if(pixels_from_l_max(series->l_max, &ntheta, &nphi, &cos_theta_array, &cos_theta_weights))
		return NULL;
	F = malloc(ntheta * (nphi / 2 + 1) * sizeof(*F));
	P = malloc(ntheta * (series->l_max + 1) * sizeof(*P));
	if(!F || !P) {
		free(F);
		free(P);
		free(cos_theta_array);
		free(cos_theta_weights);
		return NULL;
	}

	/* Fourier transform the phi co-ordinate.  after this F[][]
	 * contains H_{m}(theta) */

	{
	int n[] = {nphi};
	fftw_plan plan = fftw_plan_many_dft_r2c(1, n, ntheta, mesh, NULL, 1, nphi, F, NULL, 1, nphi / 2 + 1, FFTW_ESTIMATE);
	fftw_execute(plan);
	fftw_destroy_plan(plan);
	}

	/* multiply the samples by dphi to complete the normalization of
	 * the Fourier transform, and also by the Gauss-Legendre weights
	 * which are needed for the integral over cos(theta) below so we
	 * don't do that in the inner loop */

	{
	complex double *_F = F;
	for(int i = 0; i < ntheta; i++) {
		double weight = cos_theta_weights[i] * (2. * M_PI / nphi);
		for(int j = 0; j < (nphi / 2 + 1); j++)
			*(_F++) *= weight;
	}
	}
	free(cos_theta_weights);
	cos_theta_weights = NULL;

	/* for each non-negative m, */
	for(int m = 0; m <= m_max; m++) {
		/* populate P[][] with (normalized) P_{lm}(cos theta) */
#pragma omp parallel for shared(P, cos_theta_array, series) num_threads(4)
		for(int i = 0; i < ntheta; i++)
			gsl_sf_legendre_sphPlm_array(series->l_max, m, cos_theta_array[i], P + i * bw + m);
		/* for each l s.t. m <= l <= l_max, integrate H_{m}(theta)
		 * sin(theta) P_{lm}(cos theta) over theta for +/-m */
		for(int l = m; l <= (int) series->l_max; l++) {
			complex double *_F = F + m;
			double *_P = P + l;
			complex double c = 0.0;
			for(int i = 0; i < ntheta; i++, _F += nphi / 2 + 1, _P += bw)
				c += *_F * *_P;
			sh_series_set(series, l, -m, (m & 1 ? -1 : 1) * conj(sh_series_set(series, l, m, c)));
		}
	}

	/* clean up */
	free(P);
	free(F);
	free(cos_theta_array);

	return series;
}


/*
 * Given a complex valued function of theta, phi, project the function onto
 * spherical harmonics upto and including l = l_max, and return the
 * coefficients as an sh_series object.  Returns NULL on failure.
 *
 * The function's signature is
 *
 * complex double func(double theta, double phi, void *data);
 *
 * where data is the data argument passed to sh_series_from_func()
 */


struct sh_series *sh_series_from_func(struct sh_series *series, complex double (*func)(double, double, void *), void *data)
{
	complex double *mesh = sh_series_mesh_from_func(series->l_max, func, data, NULL, NULL);

	if(!mesh)
		return NULL;

	series = sh_series_from_mesh(series, mesh);
	free(mesh);

	return series;
}


/*
 * Given a real-valued function of theta, phi, project the function onto
 * spherical harmonics upto and including l = l_max, and return the
 * coefficients as an sh_series object.  Returns NULL on failure.
 *
 * The function's signature is
 *
 * double func(double theta, double phi, void *data);
 *
 * where data is the data argument passed to sh_series_from_realfunc()
 */


struct sh_series *sh_series_from_realfunc(struct sh_series *series, double (*func)(double, double, void *), void *data)
{
	double *mesh = sh_series_mesh_from_realfunc(series->l_max, func, data, NULL, NULL);

	if(!mesh)
		return NULL;

	series = sh_series_from_realmesh(series, mesh);
	free(mesh);
	return series;
}


/*
 * Compute a pixel mesh from an sh_series object.  The inverse of
 * sh_series_from_mesh().
 */


complex double *sh_series_to_mesh(const struct sh_series *series, int *ntheta, int *nphi, double **cos_theta_array)
{
	int _ntheta, _nphi;
	double *_cos_theta_array;
	int m_max = series->polar ? 0 : series->l_max;
	complex double *mesh = sh_series_mesh_new(series->l_max, &_ntheta, &_nphi, &_cos_theta_array, NULL);
	complex double *F = calloc(_ntheta * _nphi, sizeof(*F));

	/* check memory allocation */

	if(!mesh || !F) {
		if(ntheta)
			*ntheta = 0;
		if(nphi)
			*nphi = 0;
		if(cos_theta_array)
			*cos_theta_array = NULL;
		free(mesh);
		free(F);
		free(_cos_theta_array);
		return NULL;
	}

	/* pass mesh parameters to calling code if asked */

	if(ntheta)
		*ntheta = _ntheta;
	if(nphi)
		*nphi = _nphi;
	if(cos_theta_array)
		*cos_theta_array = _cos_theta_array;

	/* populate F[] */
#pragma omp parallel for shared(series, _ntheta, _nphi, _cos_theta_array, m_max, F) num_threads(4)
	for(int i = 0; i < _ntheta; i++) {
		/* m = 0 */
		double P[series->l_max + 1];
		complex double *coeff = series->coeff + sh_series_moffset(series->l_max, 0);
		complex double x = 0.;
		/* compute (normalized) P_{lm}(cos theta) */
		gsl_sf_legendre_sphPlm_array(series->l_max, 0, _cos_theta_array[i], P);
		for(int l = 0; l <= (int) series->l_max; l++)
			x += *coeff++ * P[l];
		*(F + i * _nphi) = x;

		for(int m = 1; m <= m_max; m++) {
			complex double *coeffp = series->coeff + sh_series_moffset(series->l_max, +m);
			complex double *coeffm = series->coeff + sh_series_moffset(series->l_max, -m);
			complex double xp = 0.;
			complex double xm = 0.;
			/* compute (normalized) P_{lm}(cos theta) */
			gsl_sf_legendre_sphPlm_array(series->l_max, m, _cos_theta_array[i], P + m);
			for(int l = abs(m); l <= (int) series->l_max; l++) {
				xp += *coeffp++ * P[l];
				xm += *coeffm++ * P[l];
			}
			*(F + i * _nphi + m) = +xp;
			*(F + (i + 1) * _nphi - m) = (m & 1) ? -xm : +xm;
		}
	}

	/* Fourier transform the m co-ordinate to the phi co-ordinate */
	{
	int n[] = {_nphi};
	fftw_plan plan = fftw_plan_many_dft(1, n, _ntheta, F, NULL, 1, _nphi, mesh, NULL, 1, _nphi, FFTW_BACKWARD, FFTW_ESTIMATE);
	fftw_execute(plan);
	fftw_destroy_plan(plan);
	}

	/* clean up */
	free(F);
	if(!cos_theta_array)
		free(_cos_theta_array);

	return mesh;
}


/*
 * generate a band-limited impulse on the sphere, with bandwidth set by
 * l_max.  the peak of the impulse is located at the co-ordinates (theta,
 * phi).  returns a newly allocated sh_series on success, or NULL on
 * failure.  this is the spherical equivalent of the sinc function for 1-D
 * time series analysis, and is obtained in the same manner by projecting a
 * Dirac delta function onto the harmonic basis and retaining only those
 * components at or below the requested l_max.
 */


struct sh_series *sh_series_impulse(unsigned int l_max, double theta, double phi)
{
	struct sh_series *series = sh_series_new(l_max, 0);

	if(!series)
		return NULL;

	/* project a Dirac delta onto the spherical harmonic basis upto and
	 * including order l_max */

	for(int m = -(int) l_max; m <= (int) l_max; m++)
		sh_series_Yconj_array(series->coeff + sh_series_moffset(l_max, m), l_max, m, theta, phi);

	return series;
}
