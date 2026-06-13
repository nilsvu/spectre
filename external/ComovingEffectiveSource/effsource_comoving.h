/*******************************************************************************
 * Copyright (C) 2025 Barry Wardell and Sam Upton
 ******************************************************************************/

struct coordinate {
  double t;
  double r;
  double theta;
  double phi;
};

double LegendreP(double l, int m, double z);

#ifndef EFFSOURCE_COMOVING_H
#define EFFSOURCE_COMOVING_H

//So that the cpp code can call the C function set_particle
#ifdef __cplusplus
extern "C" {
#endif

void effsource_init(double mass, double spin);
void effsource_set_particle(double rp);
void effsource_calc_m(int m, struct coordinate * x, double *hS_re, double *hS_im,
   double *dhS_dr_re, double *dhS_dr_im, double *dhS_dth_re, double *dhS_dth_im,
   double *dhS_dph_re, double *dhS_dph_im, double *dhS_dt_re, double *dhS_dt_im,
   double *src_re, double *src_im);

#ifdef __cplusplus
}
#endif

#endif

void effsource_hS(struct coordinate x, double *hS);
void effsource_hS_m(int m, struct coordinate x, double * hS_re, double * hS_im);
void effsource_Psi4S_m(int m, struct coordinate x, double * Psi4S_re, double * Psi4S_im);
void effsource_OPsi4S_m(int m, struct coordinate x, double * OPsi4S_re, double * OPsi4S_im);
