AM_CPPFLAGS = -I${srcdir}/../include -I.

bin_PROGRAMS = \
	inspiral_sky_map \
	sphradiometer_demo

dist_bin_SCRIPTS = \
	skymap_workflow/sph_skymap_common.py \
	skymap_workflow/sph_skymap_from_calc_snr.py \
	skymap_workflow/sph_skymap_injection_pipe.py \
	skymap_workflow/sph_skymap_precompute.py \
	skymap_workflow/sph_skymap_precompute_pipe.py \
	skymap_workflow/sph_spherical_harmonic_coeff_summary.py

skymap_workflowdocdir = $(pkgdatadir)/skymap_workflow
dist_skymap_workflowdoc_DATA = \
	skymap_workflow/README.md \
	skymap_workflow/example_injections.ini \
	skymap_workflow/example_manifest.csv \
	skymap_workflow/example_precompute.ini

noinst_PROGRAMS = \
	fastX

sphradiometer_demo_SOURCES = sphradiometer_demo.c
sphradiometer_demo_CFLAGS = $(AM_CFLAGS) $(LAL_CFLAGS) $(FFTW_CFLAGS) $(GSL_CFLAGS)
sphradiometer_demo_LDFLAGS = $(AM_LDFLAGS) $(LAL_LIBS) $(FFTW_LIBS) $(GSL_LIBS)
sphradiometer_demo_LDADD = ../lib/libsphradiometer.la ../lib/libsphradiometerdiag.la

fastX_SOURCES = fastX.c
fastX_CFLAGS = $(AM_CFLAGS) $(LALSIMULATION_CFLAGS) $(LALFRAME_CFLAGS) $(LAL_CFLAGS) $(FFTW_CFLAGS) $(GSL_CFLAGS)
fastX_LDFLAGS = $(AM_LDFLAGS) $(LALSIMULATION_LIBS) $(LALFRAME_LIBS) $(LAL_LIBS) $(FFTW_LIBS) $(GSL_LIBS)
fastX_LDADD = ../lib/libsphradiometer.la

inspiral_sky_map_SOURCES = inspiral_sky_map.c
inspiral_sky_map_CFLAGS = $(AM_CFLAGS) $(LALSIMULATION_CFLAGS) $(LALFRAME_CFLAGS) $(LAL_CFLAGS) $(FFTW_CFLAGS) $(GSL_CFLAGS) $(HDF5_CFLAGS)
inspiral_sky_map_LDFLAGS = $(AM_LDFLAGS) $(LALSIMULATION_LIBS) $(LALFRAME_LIBS) $(LAL_LIBS) $(FFTW_LIBS) $(GSL_LIBS) $(HDF5_LDFLAGS) $(HDF5_LIBS)
inspiral_sky_map_LDADD = ../lib/libsphinspiralskymap.la ../lib/libsphradiometer.la
