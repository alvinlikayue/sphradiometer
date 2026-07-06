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


#include <sphradiometer/correlator.h>
#include <sphradiometer/sh_series.h>
#include <lal/Sequence.h>
#include <lal/TimeSeries.h>


/*
 * ============================================================================
 *
 *                                    API
 *
 * ============================================================================
 */


COMPLEX16TimeSeries *get_complex16series_from_cache(const char *, const char *);
COMPLEX16Sequence *convert_TimeSeries2Sequence(COMPLEX16TimeSeries *);
void preprocess_SNRTimeSeries(COMPLEX16TimeSeries **, COMPLEX16Sequence **, int);
long precalculated_TimeSeries_length(long, double);
double **transpose_matrix(double **, int, int);

struct sh_series *sh_series_log_uniformsky_prior(int);

int correlator_network_plan_mult_by_projection(struct correlator_network_plan_fd *, double, double, double **);
int correlator_network_plan_mult_by_projection_gmst(struct correlator_network_plan_fd *, double, double, double, double **);
int correlator_network_plan_mult_by_projection_gmst_frame(struct correlator_network_plan_fd *, double, double, double, double, double **);
int correlator_network_plan_conj_delay_product(struct correlator_network_plan_fd *);
struct autocorrelator_network_plan_fd *autocorrelator_network_plan_fd_new(const struct instrument_array *, double, double, double **, int, int);
struct autocorrelator_network_plan_fd *autocorrelator_network_plan_fd_set_l(struct autocorrelator_network_plan_fd *, unsigned);
void autocorrelator_network_plan_fd_free(struct autocorrelator_network_plan_fd *);

int generate_alm_skys(struct sh_series **, struct sh_series **, struct correlator_network_plan_fd *, struct correlator_network_plan_fd *, struct autocorrelator_network_plan_fd *, struct autocorrelator_network_plan_fd *, COMPLEX16TimeSeries **, COMPLEX16Sequence **, struct sh_series *);
int generate_alm_skys_options(struct sh_series **, struct sh_series **, struct correlator_network_plan_fd *, struct correlator_network_plan_fd *, struct autocorrelator_network_plan_fd *, struct autocorrelator_network_plan_fd *, COMPLEX16TimeSeries **, COMPLEX16Sequence **, struct sh_series *, double, int, double, double, double, int);

int read_precalc_time_series_length(unsigned int *, char *);
struct sh_series *read_precalc_logprior(char *);
struct correlator_network_baselines *read_precalc_correlator_network_baselines(const struct instrument_array *, char *);
int read_precalc_correlator_network_plan_fd(struct correlator_network_plan_fd *, struct correlator_network_plan_fd *, const struct correlator_network_baselines *, int, char *);
int read_precalc_autocorrelator_network_plan_fd(struct autocorrelator_network_plan_fd *, struct autocorrelator_network_plan_fd *, const struct instrument_array *, int, char *);

int make_precalc_directories(char *, int);
int write_precalc_time_series_length(unsigned int, char *);
int write_precalc_logprior(const struct sh_series *, char *);
int write_precalc_correlator_network_plan_fd(const struct correlator_network_plan_fd *, const struct correlator_network_plan_fd *, char *);
int write_precalc_autocorrelator_network_plan_fd(const struct autocorrelator_network_plan_fd *, const struct autocorrelator_network_plan_fd *, char *);
