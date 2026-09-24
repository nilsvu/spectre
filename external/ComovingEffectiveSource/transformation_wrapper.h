#ifndef CPP_BRIDGE_H
#define CPP_BRIDGE_H

#ifdef __cplusplus
extern "C" {
#endif

void transformation(
    const double * Kinn_h_re, const double * Kinn_h_im,
    const double * dh_dX_re, const double * dh_dX_im,
    const double * dh_dY_re, const double * dh_dY_im,
    const double * src_re_tetrad, const double * src_im_tetrad,
    double * hS_re, double * hS_im,
    double * dhS_dr_re, double * dhS_dr_im,
    double * dhS_dtheta_re, double * dhS_dtheta_im,
    double * src_re, double * src_im,
    double r, double theta, double a, double M, double r0
);

#ifdef __cplusplus
}  // closes extern "C"
#endif

#endif // CPP_BRIDGE_H
