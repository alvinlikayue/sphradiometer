/*
 * Copyright (C) 2006--2009,2019,2022  Kipp C. Cannon
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


#include <complex.h>
#include <math.h>
#include <sphradiometer/sh_series.h>


/*
 * ============================================================================
 *
 *                                  Helpers
 *
 * ============================================================================
 */


/*
 * Compute the spherical harmonic expansion of cos(theta) * the function
 * given by the spherical harmonic expansion in series.  Returns the result
 * in a newly allocated sh_series object.  Returns NULL on failure.
 *
 * NOTE:  the result's l_max is increased by 1 from input's l_max.
 *
 * NOTE:  sin(theta) has a conical cusp at the poles where it is not
 * differentiable and so that function cannot be represented as a linear
 * combination of a finite number of spherical harmonics, and an equivalent
 * multiplication by sin(theta) is not available here.
 */


struct sh_series *sh_series_times_costheta(const struct sh_series *series)
{
	/* cos theta = 2 \sqrt{pi/3} Y_(1, 0)(theta, phi)
	 *
	 * see below for the derivation showing that
	 *
	 * Y_(1 0) Y_(l m) = \sqrt{3 (2 l + 1) / (4 pi)} (-1)^m [
	 *	\sqrt{2 l - 1} wigner3j(1 l (l-1) 0 m -m) wigner3j(1 l (l-1) 0 0 0) Y_(l-1 m) +
	 *	\sqrt{2 l + 3} wigner3j(1 l (l+1) 0 m -m) wigner3j(1 l (l+1) 0 0 0) Y_(l+1 m)
	 * ]
	 *
	 * using those, we have
	 *
	 * cos theta Y_(l m) = \sqrt{2 l + 1} (-1)^m [
	 *	\sqrt{2 l - 1} wigner3j(1 l (l-1) 0 m -m) wigner3j(1 l (l-1) 0 0 0) Y_(l-1 m) +
	 *	\sqrt{2 l + 3} wigner3j(1 l (l+1) 0 m -m) wigner3j(1 l (l+1) 0 0 0) Y_(l+1 m)
	 * ]
	 */
	struct sh_series *result = sh_series_new_zero(series->l_max + 1, series->polar);
	complex double *c;
	int l, m;

	if(!result)
		return NULL;

	c = result->coeff;
	/* input (l m) = (0 0) */
	c[sh_series_lmoffset(result, 1, 0)] += sh_series_get(series, 0, 0) / sqrt(3);
	/* input l >= 1 */
	for(l = 1; l <= (int) series->l_max; l++) {
		/* m-independent parts */
		/* for (l-1, m) term */
		double g1 = sqrt((2 * l + 1) * (2 * l - 1)) * sh_series_wigner_3j(1, l, l - 1, 0, 0, 0);
		/* for (l+1, m) term */
		double g2 = sqrt((2 * l + 1) * (2 * l + 3)) * sh_series_wigner_3j(1, l, l + 1, 0, 0, 0);

		int m_max = series->polar ? 0 : l;
		for(m = -m_max; m <= m_max; m++) {
			/* input a_lm * (-1)^m */
			complex double alm = m & 1 ? -sh_series_get(series, l, m) : +sh_series_get(series, l, m);
			/* NOTE:  the (l, m) offset calculation is for the
			 * output series.  the output series' l_max is 1
			 * greater than the input series' */
			c[sh_series_lmoffset(result, l - 1, m)] += alm * sh_series_wigner_3j(1, l, l - 1, 0, m, -m) * g1;
			c[sh_series_lmoffset(result, l + 1, m)] += alm * sh_series_wigner_3j(1, l, l + 1, 0, m, -m) * g2;
		}
	}

	return result;
}


/*
 * Compute the spherical harmonic expansion of sin^2(theta) * the function
 * given by the spherical harmonic expansion in series.  Returns the result
 * in a newly allocated sh_series object.  Returns NULL on failure.
 *
 * NOTE:  the result's l_max is increased by 2 from input's l_max.
 */


struct sh_series *sh_series_times_sin2theta(const struct sh_series *series)
{
	/* sin^2 theta
	 *   = (2/3) - (4/3) \sqrt{pi/5} Y_(2 0)
	 *
	 * from below we have
	 *
	 * Y_(2 0) Y_(l m) = \sqrt{5 (2 l + 1) / (4 pi)} (-1)^m [
	 *	\sqrt{2 l - 3} wigner3j(2 l (l-2) 0 m -m) wigner3j(2 l (l-2) 0 0 0) Y_(l-2 m) +
	 *	\sqrt{2 l + 1} wigner3j(2 l l     0 m -m) wigner3j(2 l l     0 0 0) Y_(l m) +
	 *	\sqrt{2 l + 5} wigner3j(2 l (l+2) 0 m -m) wigner3j(2 l (l+2) 0 0 0) Y_(l+2 m)
	 * ]
	 *
	 * using those, we have
	 *
	 * sin^2 theta Y_(l m) = (2/3) Y_(l m) - (2/3) \sqrt{2 l + 1} (-1)^m [
	 *	\sqrt{2 l - 3} wigner3j(2 l (l-2) 0 m -m) wigner3j(2 l (l-2) 0 0 0) Y_(l-2 m) +
	 *	\sqrt{2 l + 1} wigner3j(2 l l     0 m -m) wigner3j(2 l l     0 0 0) Y_(l m) +
	 *	\sqrt{2 l + 5} wigner3j(2 l (l+2) 0 m -m) wigner3j(2 l (l+2) 0 0 0) Y_(l+2 m)
	 * ]
	 */
	struct sh_series *result = sh_series_copy(series);
	complex double *c;
	int l, m;

	if(!result)
		return NULL;

	/*
	 * compute the contribution each Y_(l m) makes to its own term in
	 * the result by multiplying each by
	 * (2/3) (1 - (2l + 1) (-1)^m wigner3j(2 l l 0 m -m) wigner3j(2 l l 0 0 0))
	 *
	 * because 2+l+l is even, wigner3j(2 l l 0 m -m) = wigner3j(2 l l 0 -m m)
	 */

	c = result->coeff;
	for(l = 0; l <= (int) result->l_max; l++) {
		double w3j_2ll000_2lp1 = (2 * l + 1) * sh_series_wigner_3j(2, l, l, 0, 0, 0);
		int m_max = series->polar ? 0 : l;
		for(m = -m_max; m <= +m_max; m++)
			c[sh_series_lmoffset(result, l, m)] *= 2./3. * (1. - (m & 1 ? -sh_series_wigner_3j(2, l, l, 0, m, -m) : sh_series_wigner_3j(2, l, l, 0, m, -m)) * w3j_2ll000_2lp1);
	}

	/*
	 * in-place resize result to l+2
	 */

	if(!sh_series_resize(result, result->l_max + 2)) {
		sh_series_free(result);
		return NULL;
	}

	/*
	 * for each Y_(l m) add its contribution to the Y_(l+/-2 m)
	 * components of the result
	 */

	c = result->coeff;
	/* input (l m) = (0 0).
	 * sh_series_wigner_3j(2, 0, 2, 0, 0, 0) = 1/sqrt(5)
	 */
	c[sh_series_lmoffset(result, 2, 0)] -= 2./3. / sqrt(5.) * sh_series_get(series, 0, 0);

	if(series->l_max < 1)
		return result;

	/* input (l m) = (1 -1), (1 0), (1 1).
	 * wigner_3j(2, 1, 3, 0, 0, 0) * wigner_3j(2, 1, 3, 0, 0, 0) = (9. / 1225.)**.5
	 * wigner_3j(2, 1, 3, 0, -1, 1) * wigner_3j(2, 1, 3, 0, 0, 0) =
	 * 	wigner_3j(2, 1, 3, 0, 1, -1) * wigner_3j(2, 1, 3, 0, 0, 0) = -(6. / 1225.)**.5
	 *
	 * (2+1+3) is even so the wigner3j functions for m=+/-1 are equal.
	 */
	c[sh_series_lmoffset(result, 3,  0)] -= 2./3. * sqrt(21.) * sqrt(9./1225.) * sh_series_get(series, 1, 0);
	if(!series->polar) {
		c[sh_series_lmoffset(result, 3, -1)] += 2./3. * sqrt(21.) * -sqrt(6./1225.) * sh_series_get(series, 1, -1);
		c[sh_series_lmoffset(result, 3, +1)] += 2./3. * sqrt(21.) * -sqrt(6./1225.) * sh_series_get(series, 1, +1);
	}

	/* input l >= 2 */
	for(l = 2; l <= (int) series->l_max; l++) {
		/* m-independent parts */
		/* for (l-2, m) term */
		double g1 = 2./3. * sqrt((2 * l + 1) * (2 * l - 3)) * sh_series_wigner_3j(2, l, l - 2, 0, 0, 0);
		/* for (l+2, m) term */
		double g2 = 2./3. * sqrt((2 * l + 1) * (2 * l + 5)) * sh_series_wigner_3j(2, l, l + 2, 0, 0, 0);

		int m_max = series->polar ? 0 : l;
		for(m = -m_max; m <= m_max; m++) {
			/* input a_lm * (-1)^m */
			complex double alm = m & 1 ? -sh_series_get(series, l, m) : +sh_series_get(series, l, m);
			/* NOTE:  the (l, m) offset calculation is for the
			 * output series.  the output series' l_max is 2
			 * greater than the input series' */
			c[sh_series_lmoffset(result, l - 2, m)] -= alm * sh_series_wigner_3j(2, l, l - 2, 0, m, -m) * g1;
			c[sh_series_lmoffset(result, l + 2, m)] -= alm * sh_series_wigner_3j(2, l, l + 2, 0, m, -m) * g2;
		}
	}

	return result;
}


/*
 * ============================================================================
 *
 *                          First Order Derivatives
 *
 * ============================================================================
 */


/*
 * Compute the spherical harmonic expansion of the 1st derivative with
 * respect to phi of the function on the sphere described by an sh_series
 * object.  This is done in-place.

 * Returns the sh_series object on success.  Does not fail.
 */


struct sh_series *sh_series_d_by_dphi(struct sh_series *series)
{
	int l_max = series->l_max;
	int m_max = series->polar ? 0 : l_max;
	complex double *c;
	int l, m;

	/* Y_lm(theta, phi) \propto exp(i m phi), so the 1st derivative
	 * with respect to phi is obtained my multiplying each component by
	 * (i m). */

	c = series->coeff;
	/* m = 0 */
	for(l = 0; l <= l_max; l++)
		*(c++) = 0.0;
	/* m > 0 */
	for(m = 1; m <= m_max; m++) {
		complex double factor = I * m;
		for(l = m; l <= l_max; l++)
			*(c++) *= factor;
	}
	/* m < 0 */
	for(m = -m_max; m < 0; m++) {
		complex double factor = I * m;
		for(l = -m; l <= l_max; l++)
			*(c++) *= factor;
	}

	return series;
}


/*
 * Compute the spherical harmonic expansion of sin theta times the 1st
 * derivative with respect to theta of the function on the sphere described
 * by an sh_series object.  Returns the result in a newly allocated
 * sh_series object.  Returns NULL on failure.
 *
 * NOTE:  the result's l_max is increased by 1 from the input's l_max.
 *
 * NOTE:  the algorithm relies on GSL's Wigner 3-j function, which is
 * numerically unstable above l of about 70, and so this function should
 * not be relied upon to compute the derivatives of functions with higher
 * bandwidths than that.
 */


struct sh_series *sh_series_sintheta_d_by_dtheta(const struct sh_series *series)
{
	struct sh_series *result = sh_series_new_zero(series->l_max + 1, series->polar);
	complex double *c;
	int l, m;

	if(!result)
		return NULL;

	/* d/dtheta Y_lm(theta, phi)
	 *	= m cot(theta) Y_lm(theta, phi) +
	 *	  \sqrt{(l - m) (l + m + 1)} exp(-i \phi) Y_(l m+1)(theta, phi)
	 *
	 * sin theta d/dtheta Y_lm(theta, phi)
	 *	= m cos(theta) Y_lm(theta, phi) +
	 *	  \sqrt{(l - m) (l + m + 1)} sin theta exp(-i \phi) Y_(l m+1)(theta, phi)
	 *
	 * sin theta d/dtheta Y_lm(theta, phi)
	 *	= 2 m \sqrt{pi/3} Y_(1,0)(theta, phi) Y_lm(theta, phi) +
	 *	  2 \sqrt{2pi/3} \sqrt{(l - m) (l + m + 1)} Y_(1,-1)(theta, phi) Y_(l m+1)(theta, phi)
	 *
	 * using
	 *
	 * Y_(j1 m1) Y_(j2 m2) =
	 *	\sqrt{(2 j1 + 1) (2 j2 + 1) / (4 pi)}
	 *	\sum_j3=0^\infty \sum_m3=-j3^+j3
	 *		(-1)^m3 \sqrt{2 j3 + 1} wigner3j(j1 j2 j3 m1 m2 -m3) wigner3j(j1 j2 j3 0 0 0) Y_(j3 m3)
	 *
	 * and the fact that wigner3j(j1 j2 j3 m1 m2 m3) = 0 unless all of
	 * the following are true
	 *	1: -ji <= mi <= ji
	 *	2: m1 + m2 + m3 = 0
	 *	3: |j1 - j2| <= j3 <= j1 + j2
	 *	4: (j1 + j2 + j3) is an integer, and an even integer if (m1
	 *	= m2 = m3 = 0)
	 *
	 * we find that
	 *
	 * Y_(1 0) Y_(l m) = \sqrt{3 (2 l + 1) / (4 pi)} (-1)^m [
	 *		\sqrt{2 l - 1} wigner3j(1 l (l-1) 0 m -m) wigner3j(1 l (l-1) 0 0 0) Y_(l-1 m) +
	 *		\sqrt{2 l + 3} wigner3j(1 l (l+1) 0 m -m) wigner3j(1 l (l+1) 0 0 0) Y_(l+1 m)
	 * ]
	 *
	 * and
	 *
	 * Y_(1 -1) Y_(l m+1) = \sqrt{3 (2 l + 1) / (4 pi)} (-1)^m [
	 *		\sqrt{2 l - 1} wigner3j(1 l (l-1) -1 m+1 -m) wigner3j(1 l (l-1) 0 0 0) Y_(l-1 m) +
	 *		\sqrt{2 l + 3} wigner3j(1 l (l+1) -1 m+1 -m) wigner3j(1 l (l+1) 0 0 0) Y_(l+1 m)
	 * ]
	 *
	 * and (used elsewhere)
	 *
	 * Y_(2 0) Y_(l m) = \sqrt{5 (2 l + 1) / (4 pi)} (-1)^m [
	 *		\sqrt{2 l - 3} wigner3j(2 l (l-2) 0 m -m) wigner3j(2 l (l-2) 0 0 0) Y_(l-2 m) +
	 *		\sqrt{2 l + 1} wigner3j(2 l l     0 m -m) wigner3j(2 l l     0 0 0) Y_(l m) +
	 *		\sqrt{2 l + 5} wigner3j(2 l (l+2) 0 m -m) wigner3j(2 l (l+2) 0 0 0) Y_(l+2 m)
	 * ]
	 *
	 * so
	 *
	 * sin theta d/dtheta Y_lm(theta, phi) =
	 *	2 m \sqrt{pi/3} \sqrt{3 (2 l + 1) / (4 pi)} (-1)^m [
	 *		\sqrt{2 l - 1} wigner3j(1 l (l-1) 0 m -m) wigner3j(1 l (l-1) 0 0 0) Y_(l-1 m) +
	 *		\sqrt{2 l + 3} wigner3j(1 l (l+1) 0 m -m) wigner3j(1 l (l+1) 0 0 0) Y_(l+1 m)
	 *	] + 2 \sqrt{2pi/3} \sqrt{(l - m) (l + m + 1)} \sqrt{3 (2 l + 1) / (4 pi)} (-1)^m [
	 *		\sqrt{2 l - 1} wigner3j(1 l (l-1) -1 m+1 -m) wigner3j(1 l (l-1) 0 0 0) Y_(l-1 m) +
	 *		\sqrt{2 l + 3} wigner3j(1 l (l+1) -1 m+1 -m) wigner3j(1 l (l+1) 0 0 0) Y_(l+1 m)
	 * ]
	 *
	 * simplifying
	 *
	 * sin theta d/dtheta Y_lm(theta, phi) =
	 *	m \sqrt{2 l + 1} (-1)^m [
	 *		\sqrt{2 l - 1} wigner3j(1 l (l-1) 0 m -m) wigner3j(1 l (l-1) 0 0 0) Y_(l-1 m) +
	 *		\sqrt{2 l + 3} wigner3j(1 l (l+1) 0 m -m) wigner3j(1 l (l+1) 0 0 0) Y_(l+1 m)
	 *	] + \sqrt{2 (l - m) (l + m + 1)} \sqrt{2 l + 1} (-1)^m [
	 *		\sqrt{2 l - 1} wigner3j(1 l (l-1) -1 m+1 -m) wigner3j(1 l (l-1) 0 0 0) Y_(l-1 m) +
	 *		\sqrt{2 l + 3} wigner3j(1 l (l+1) -1 m+1 -m) wigner3j(1 l (l+1) 0 0 0) Y_(l+1 m)
	 * ]
	 *
	 * sin theta d/dtheta Y_lm(theta, phi) = \sqrt{2 l + 1} (-1)^m [
	 *	m \sqrt{2 l - 1} wigner3j(1 l (l-1) 0 m -m) wigner3j(1 l (l-1) 0 0 0) Y_(l-1 m) +
	 *	m \sqrt{2 l + 3} wigner3j(1 l (l+1) 0 m -m) wigner3j(1 l (l+1) 0 0 0) Y_(l+1 m) +
	 *	\sqrt{2 (l - m) (l + m + 1)} \sqrt{2 l - 1} wigner3j(1 l (l-1) -1 m+1 -m) wigner3j(1 l (l-1) 0 0 0) Y_(l-1 m) +
	 *	\sqrt{2 (l - m) (l + m + 1)} \sqrt{2 l + 3} wigner3j(1 l (l+1) -1 m+1 -m) wigner3j(1 l (l+1) 0 0 0) Y_(l+1 m)
	 * ]
	 *
	 * sin theta d/dtheta Y_lm(theta, phi) = \sqrt{2 l + 1} (-1)^m [
	 *	[ m wigner3j(1 l (l-1) 0 m -m) + \sqrt{2 (l - m) (l + m + 1)} wigner3j(1 l (l-1) -1 m+1 -m) ] \sqrt{2 l - 1} wigner3j(1 l (l-1) 0 0 0) Y_(l-1 m) +
	 *	[ m wigner3j(1 l (l+1) 0 m -m) + \sqrt{2 (l - m) (l + m + 1)} wigner3j(1 l (l+1) -1 m+1 -m) ] \sqrt{2 l + 3} wigner3j(1 l (l+1) 0 0 0) Y_(l+1 m)
	 * ]
	 *
	 * sin theta d/dtheta Y_lm(theta, phi) = 2 \sqrt{l + 0.5} (-1)^m [
	 *	[ m wigner3j(1 l (l-1) 0 m -m) + \sqrt{2 (l - m) (l + m + 1)} wigner3j(1 l (l-1) -1 m+1 -m) ] \sqrt{l - 0.5} wigner3j(1 l (l-1) 0 0 0) Y_(l-1 m) +
	 *	[ m wigner3j(1 l (l+1) 0 m -m) + \sqrt{2 (l - m) (l + m + 1)} wigner3j(1 l (l+1) -1 m+1 -m) ] \sqrt{l + 1.5} wigner3j(1 l (l+1) 0 0 0) Y_(l+1 m)
	 * ]
	 *
	 * from the recursion relation for Wigner 3-j functions,
	 *
	 * \sqrt{(l1 -/+ s1)(l1 +/- s1 + 1)} wigner3j(l1 l2 l3 s1+/-1 s2 s3) +
	 * \sqrt{(l2 -/+ s2)(l2 +/- s2 + 1)} wigner3j(l1 l2 l3 s1 s2+/-1 s3) +
	 * \sqrt{(l3 -/+ s3)(l3 +/- s3 + 1)} wigner3j(l1 l2 l3 s1 s2 s3+/-1) = 0,
	 *
	 * setting l1=1, l2=l, l3=l+/-1, s1=0, s2 = m+1, s3 = -m gives
	 *
	 * \sqrt{2} wigner3j(1 l l+/-1 -1 m+1 -m) =
	 *	-\sqrt{(l + m + 1)(l - m)} wigner3j(1 l l+/-1 0 m -m) - \sqrt{(l+/-1 - m)(l+/-1 + m + 1)} wigner3j(1 l l+/-1 0 m+1 -(m+1))
	 *
	 * therefore,
	 *
	 * sin theta d/dtheta Y_lm(theta, phi) = 2 \sqrt{l + 0.5} (-1)^m [
	 *	[ m wigner3j(1 l (l-1) 0 m -m) + \sqrt{(l - m) (l + m + 1)} [ -\sqrt{(l + m + 1)(l - m)} wigner3j(1 l l-1 0 m -m) - \sqrt{(l-1 - m)(l-1 + m + 1)} wigner3j(1 l l-1 0 m+1 -(m+1)) ] ] \sqrt{l - 0.5} wigner3j(1 l (l-1) 0 0 0) Y_(l-1 m) +
	 *	[ m wigner3j(1 l (l+1) 0 m -m) + \sqrt{(l - m) (l + m + 1)} [ -\sqrt{(l + m + 1)(l - m)} wigner3j(1 l l+1 0 m -m) - \sqrt{(l+1 - m)(l+1 + m + 1)} wigner3j(1 l l+1 0 m+1 -(m+1)) ] ] \sqrt{l + 1.5} wigner3j(1 l (l+1) 0 0 0) Y_(l+1 m)
	 * ]
	 *
	 * sin theta d/dtheta Y_lm(theta, phi) = 2 \sqrt{l + 0.5} (-1)^m [
	 *	[ (m - (l - m) (l + m + 1)) wigner3j(1 l (l-1) 0 m -m) - \sqrt{(l - m) (l + m + 1)} \sqrt{(l - m - 1)(l + m)    } wigner3j(1 l (l-1) 0 (m+1) -(m+1)) ] \sqrt{l - 0.5} wigner3j(1 l (l-1) 0 0 0) Y_(l-1 m) +
	 *	[ (m - (l - m) (l + m + 1)) wigner3j(1 l (l+1) 0 m -m) - \sqrt{(l - m) (l + m + 1)} \sqrt{(l - m + 1)(l + m + 2)} wigner3j(1 l (l+1) 0 (m+1) -(m+1)) ] \sqrt{l + 1.5} wigner3j(1 l (l+1) 0 0 0) Y_(l+1 m)
	 * ]
	 *
	 * this last form depends only on wigner3j(1 l (l+/-1) 0 m -m).  in
	 * the inner loop over m, wigner3j(1 l (l+/-1) 0 (m+1) -(m+1)) can
	 * be saved and used for wigner3j(1 l (l+/-1) 0 m -m) in the next
	 * iteration (after m is incremented by 1).  this reduces the
	 * number of evaluations of the Wigner 3-j function by
	 * approximately a factor of 2.  more arithmetic is needed to
	 * compute the coefficients, and without more information about the
	 * efficiency of GSL's Wigner 3-j implementation, it's not clear if
	 * the result is a win or not.  initial benchmarking shows very
	 * little difference in speed between the two approaches.
	 *
	 * another symmetry that helps reduce the wigner3j evaluations in
	 * the loops over m is
	 *
	 *	wigner3j(l1 l2 l3 0 m -m) =
	 *		(-1)^(l1+l2+l3) wigner3j(l1 l2 l3 0 -m m)
	 */

	/* FIXME:  GSL's Wigner 3-j functions are not stable for large l,
	 * but the form we are using has a 1 in the first j position, so we
	 * could potentially switch to an asymptotic approximation for
	 * large l, assuming the asymptotic form is good enough by the time
	 * we enter the regime where GSL's 3-j function isn't */

	c = result->coeff;
	/* l = 0 does not contribute to result.  NOTE:  the loop is over
	 * the l index of the input series. */
	for(l = 1; l <= (int) series->l_max; l++) {
		/* m-independent parts */
		double x = 2. * sqrt(l + 0.5);
		/* for (l-1, m) term */
		double g1 = x * sqrt(l - 0.5) * sh_series_wigner_3j(1, l, l - 1, 0, 0, 0);
		/* for (l+1, m) term */
		double g2 = x * sqrt(l + 1.5) * sh_series_wigner_3j(1, l, l + 1, 0, 0, 0);

		int m_max = series->polar ? 0 : l;
		/* compute the "m" Wigner 3-j functions for the first loop
		 * iteration */
		double w3j_lm1_m = sh_series_wigner_3j(1, l, l - 1, 0, -m_max, m_max);
		double w3j_lp1_m = sh_series_wigner_3j(1, l, l + 1, 0, -m_max, m_max);
		for(m = -m_max; m <= m_max; m++) {
			/* input a_lm * (-1)^m */
			complex double alm = m & 1 ? -sh_series_get(series, l, m) : +sh_series_get(series, l, m);
			/* m-dependent parts */
			int A = l - m;
			int B = l + m;
			int C = A * (B + 1);
			/* compute the "m+1" Wigner 3-j functions */
			double w3j_lm1_mp1 = sh_series_wigner_3j(1, l, l - 1, 0, m + 1, -(m + 1));
			double w3j_lp1_mp1 = sh_series_wigner_3j(1, l, l + 1, 0, m + 1, -(m + 1));
			/* for (l-1, m) term */
			x = m - C;
			double f1 = x * w3j_lm1_m - sqrt(C * (A - 1) * B) * w3j_lm1_mp1;
			/* for (l+1, m) term */
			double f2 = x * w3j_lp1_m - sqrt(C * (A + 1) * (B + 2)) * w3j_lp1_mp1;
			/* use the "m+1" Wigner 3-j functions as the "m"
			 * functions in the next iteration */
			w3j_lm1_m = w3j_lm1_mp1;
			w3j_lp1_m = w3j_lp1_mp1;

			/* NOTE:  the (l, m) offset calculation is for the
			 * output series.  the output series' l_max is 1
			 * greater than the input series' */
			c[sh_series_lmoffset(result, l - 1, m)] += alm * f1 * g1;
			c[sh_series_lmoffset(result, l + 1, m)] += alm * f2 * g2;
		}
	}

	return result;
}


/*
 * Compute the spherical harmonic expansion of the theta and phi components
 * of sin theta times the gradient of the function on the sphere described
 * by an sh_series object.  The theta component is returned in u, and the
 * phi component in v, both of which will be newly allocated.
 *
 * Returns the original sh_series object on success.  On failure, any
 * memory that was allocated is freed, and u and v are reset to NULL, and
 * NULL is returned.
 *
 * NOTES
 *
 * The gradient in spherical polar co-ordinates is
 *
 *             \partial                          1     \partial
 * \hat{theta} --------------  +  \hat{phi} ---------- ------------
 *             \partial theta               sin(theta) \partial phi
 *
 * This is poorly behaved at the poles, where sin(theta) = 0.  Any
 * variation with respect to phi at the poles leads to a divergent result.
 * Normal, physical, functions on the sphere will lose their phi dependence
 * near the poles, but that leads to a 0/0 situation in the gradient where
 * formally the limit might exist and be well behaved but numerically the
 * result is likely unstable.  A similar issue exists in the theta
 * derivative component, although it's hidden.  Interally, differentiating
 * spherical harmonic basis functions with respect to theta results in two
 * terms, one of which includes a cot(theta) factor.  This diverges at the
 * poles for the same reason:  any theta dependence exactly at the pole is
 * equivalent to a conical cusp, for which the derivative is undefined.
 * Again, normal physical functions might be well behaved but numerically
 * the evaluation of the derivative is unstable.
 *
 * In any case, even disregarding the issues with numerical stability,
 * 1/sin(theta) cannot be written as a linear combination of a finite
 * number of spherical harmonic basis functions, which introduces the need
 * to choose some arbitrary cut-off to represent the gradient.
 *
 * For all of these reasons, it is more convenient to compute sin(theta) *
 * grad.  This removes the 1/sin(theta) factor from the phi component and
 * also from the cot(theta) factor inside the derivative of the spherical
 * harmonic basis functions.  As a matrix operator acting on the vector of
 * coefficients describing a function in the spherical harmonic basis,
 * sin(theta) * grad is nearly diagonal whereas the grad operator alone has
 * non-zero components everywhere.  Because the same sin(theta) factor is
 * applied to both components, the direction of the gradient is not
 * affected, only its magnitude, and so, for example, for the purpose of
 * constructing contour fields or root or peak finding the inconvenience
 * should be small.
 */


const struct sh_series *sh_series_sintheta_grad(const struct sh_series *series, struct sh_series **u, struct sh_series **v)
{
	*u = sh_series_sintheta_d_by_dtheta(series);
	*v = sh_series_copy(series);
	if(!*u || !*v) {
		sh_series_free(*u);
		sh_series_free(*v);
		*u = *v = NULL;
		return NULL;
	}
	sh_series_d_by_dphi(*v);
	return series;
}


/*
 * Compute sin theta * the divergence of the vector field whose theta
 * component is in u and whose phi component is in v.
 *
 * The divergence in spherical polar co-ordinates is
 *
 *      1    \partial                            1     \partial
 * --------- -------------- [sin(theta) u] + --------- ------------ v
 * sin theta \partial theta                  sin theta \partial phi
 *
 * See sh_series_sintheta_grad() for discussion of the 1/sin(theta)
 * factors, and why it is more convenient to compute sin(theta) times the
 * divergence of the vector field.  The function returned is
 *
 * sin(theta) * div =
 *	cos(theta) u + sin(theta) d/dtheta u + d/dphi v
 */


struct sh_series *sh_series_sintheta_div(const struct sh_series *u, const struct sh_series *v)
{
	/* compute cos(theta) u + sin(theta) d/dtheta u */
	struct sh_series *result = sh_series_sintheta_d_by_dtheta(u);
	struct sh_series *costheta_u = sh_series_times_costheta(u);
	if(!(result && costheta_u && sh_series_add(result, 1.0, costheta_u))) {
		sh_series_free(result);
		sh_series_free(costheta_u);
		return NULL;
	}
	sh_series_free(costheta_u);

	/* if v has azimuthal symmetry, d/dphi v = 0 and we're done */
	if(v->polar)
		return result;

	/* bring the result to the minimum required order to continue the
	 * calculation */
	if(result->l_max < v->l_max)
		if(!sh_series_resize(result, v->l_max)) {
			sh_series_free(result);
			return NULL;
		}
	if(result->polar)
		if(!sh_series_set_polar(result, 1)) {
			sh_series_free(result);
			return NULL;
		}

	/* add d/dphi v */
	{
	struct sh_series *d_by_dphi_v = sh_series_copy(v);
	if(!d_by_dphi_v) {
		sh_series_free(result);
		return NULL;
	}
	sh_series_d_by_dphi(d_by_dphi_v);
	if(!sh_series_add(result, 1.0, d_by_dphi_v)) {
		sh_series_free(d_by_dphi_v);
		sh_series_free(result);
		return NULL;
	}
	sh_series_free(d_by_dphi_v);
	}

	/* done */
	return result;
}


/*
 * Compute and return sin(theta) * the curl of the vector field whose theta
 * and phi components are in u and v respectively.
 *
 * In 2 dimensions the curl is a scalar.  It consists of what would form
 * the radial component of the result in 3 dimensions.  The curl is
 *
 *     1      [ \partial                          \partial       ]
 * ---------- [ -------------- ( sin(theta) v ) - ------------ u ]
 * sin(theta) [ \partial theta                    \partial phi   ]
 */


struct sh_series *sh_series_sintheta_curl(const struct sh_series *u, const struct sh_series *v)
{
	/* compute cos(theta) v + sin(theta) d/dtheta v */
	struct sh_series *result = sh_series_sintheta_d_by_dtheta(v);
	struct sh_series *costheta_v = sh_series_times_costheta(v);
	if(!(result && costheta_v && sh_series_add(result, 1.0, costheta_v))) {
		sh_series_free(result);
		sh_series_free(costheta_v);
		return NULL;
	}
	sh_series_free(costheta_v);

	/* if u has azimuthal symmetry, d/dphi u = 0 and we're done */
	if(u->polar)
		return result;

	/* bring the result to the minimum required order to continue the
	 * calculation */
	if(result->l_max < u->l_max)
		if(!sh_series_resize(result, u->l_max)) {
			sh_series_free(result);
			return NULL;
		}
	if(result->polar)
		if(!sh_series_set_polar(result, 1)) {
			sh_series_free(result);
			return NULL;
		}

	/* subtract d/dphi u */
	{
	struct sh_series *d_by_dphi_u = sh_series_copy(u);
	if(!d_by_dphi_u) {
		sh_series_free(result);
		return NULL;
	}
	sh_series_d_by_dphi(d_by_dphi_u);
	if(!sh_series_add(result, -1.0, d_by_dphi_u)) {
		sh_series_free(d_by_dphi_u);
		sh_series_free(result);
		return NULL;
	}
	sh_series_free(d_by_dphi_u);
	}

	/* done */
	return result;
}


/*
 * ============================================================================
 *
 *                          Second Order Derivatives
 *
 * ============================================================================
 */


#if 0

/*
 * On the two sphere the four components of the Hessian matrix are
 *
 * H[f] =
 *  (                 d^2 f                      d^2 f       cos(theta)  d f     )
 *  (               ---------                ------------- - ---------- -----    )
 *  (               d theta^2                d theta d phi   sin(theta) d phi    )
 *  (                                                                            )
 *  (     d^2 f       cos(theta)  d f     d^2 f                            d f   )
 *  ( ------------- - ---------- -----   ------- + cos(theta) sin(theta) ------- )
 *  ( d theta d phi   sin(theta) d phi   d phi^2                         d theta )
 *
 *
 * Gradient descent algorithms for finding minima require the evaluation of
 * the inverse of this matrix.  GSL's implementation relies on a numerical
 * approximation of the inverse but if we could obtain an explicit form
 * then the extremization problem could possibly be made faster.
 */

#endif


/*
 * Replace an sh_series object with the Laplacian of the function it
 * describes.  Spherical harmonics are eigenfunctions of the Laplacian,
 *
 * 	\grad^{2} Y_{lm}(\hat{s}) = -l (l + 1) Y_{lm}(\hat{s}).
 *
 * The Laplacian is, therefore, computed by multiplying each coefficient of
 * the input by the corresponding integer eigenvalue.  This is a fast
 * operation.
 *
 * The explicit form of the operator is
 *
 *     1     [ \partial       (           \partial       ) ]
 * --------- [ -------------- ( sin theta -------------- ) ] +
 * sin theta [ \partial theta (           \partial theta ) ]
 *
 *		     1      \partial^2
 *		----------- ---------------
 *		sin^2 theta \partial \phi^2
 *
 * NOTE that because the spherical harmonic basis functions are
 * eigenfunctions of this operator, the 1/sin(theta) factors that appear in
 * this expression do not cause difficulties the way they do when
 * evaluating the divergence or gradient.  In effect, they are guaranteed
 * to cancel terms in the numerator at the poles, and this cancellation
 * occurs term-by-term for each spherical harmonic basis function
 * individually.
 */


struct sh_series *sh_series_laplacian(struct sh_series *series)
{
	complex double *c = series->coeff;
	int l = 0, m = 0;

	while(1) {
		*(c++) *= -l * (l + 1);
		if(++l > (int) series->l_max) {
			if(series->polar)
				break;
			else if(++m > (int) series->l_max)
				m = -(int) series->l_max;
			else if(m == 0)
				break;
			l = abs(m);
		}
	}

	return series;
}


/*
 * Replace an sh_series object with its inverse Laplacian by dividing each
 * coefficient by the corresponding eigenvalue.  The (0,0) coefficient is
 * undefined in this operation (being the arbitrary integration constant),
 * and this funtion sets it to 0.
 */


struct sh_series *sh_series_invlaplacian(struct sh_series *series)
{
	complex double *c = series->coeff;
	int l = 1, m = 0;

	/* l = 0 */
	*(c++) = 0.0;

	/* done? */
	if(series->l_max == 0)
		return series;

	while(1) {
		*(c++) /= -l * (l + 1);
		if(++l > (int) series->l_max) {
			if(series->polar)
				break;
			else if(++m > (int) series->l_max)
				m = -(int) series->l_max;
			else if(m == 0)
				break;
			l = abs(m);
		}
	}

	return series;
}
