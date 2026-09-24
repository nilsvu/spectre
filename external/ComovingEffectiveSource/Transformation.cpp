// NOLINTBEGIN
#include <cmath>
#include <complex>
#include <array>
#include <span>
#include <gsl/gsl_cblas.h>

#include "transformation_wrapper.h"
//Note that this is setup to transform a symmetric rank 2 tensor with lower
//for which the tetrad components are given with both tetrad indices lowered
//It returns a numerical expression for the BoyerLindquist component expression 
//with both tensor indices lowered

void transformation(
    const double * Kinn_h_re, const double * Kinn_h_im,
    const double * dh_dX_re, const double * dh_dX_im,
    const double * dh_dY_re, const double * dh_dY_im,
    const double * src_re_tetrad, const double * src_im_tetrad,
    double * hS_re, double * hS_im,
    double * dhS_dr_re, double * dhS_dr_im,
    double * dhS_dtheta_re, double * dhS_dtheta_im,
    double * src_re, double * src_im,
    double r, double theta, double a, double M, double r0)
{
    const size_t dim = 4;

    //Construct marix which transforms null tetrad components to BoyerLindquist
    std::array<std::complex<double>, dim*dim> tetrad_trans_matrix = {
        std::complex<double>{-((-a*a + 2*M*r - r*r)/(2*(r*r + a*a*cos(theta)*cos(theta)))),0}, 
        std::complex<double>{1,0},
        std::complex<double>{-(a*a*cos(theta)*sin(theta)/(sqrt(2)*(r*r + a*a*cos(theta)*cos(theta)))), (a*r*sin(theta))/(sqrt(2)*(r*r + a*a*cos(theta)*cos(theta)))}, 
        std::complex<double>{-((a*a*cos(theta)*sin(theta))/(sqrt(2)*(r*r + a*a*cos(theta)*cos(theta)))), -((a*r*sin(theta))/(sqrt(2)*(r*r + a*a*cos(theta)*cos(theta))))},

        std::complex<double>{0.5,0},
        std::complex<double>{-((r*r + a*a*cos(theta)*cos(theta))/(a*a - 2*M*r + r*r)),0},
        std::complex<double>{0,0},
        std::complex<double>{0,0},

        std::complex<double>{0,0},
        std::complex<double>{0,0},
        std::complex<double>{r/sqrt(2),(a*cos(theta))/sqrt(2)},
        std::complex<double>{r/sqrt(2),-(a*cos(theta))/sqrt(2)},

        std::complex<double>{-((a*(a*a + r*(-2*M + r))*sin(theta)*sin(theta))/(2*(r*r + a*a*cos(theta)*cos(theta)))), 0},
        std::complex<double>{-a*sin(theta)*sin(theta), 0},
        std::complex<double>{(a*(a*a + r*r)*cos(theta)*sin(theta))/(sqrt(2)*(r*r + a*a*cos(theta)*cos(theta))), -((r*(a*a + r*r)*sin(theta))/(sqrt(2)*(r*r + a*a*cos(theta)*cos(theta))))},
        std::complex<double>{(a*(a*a + r*r)*cos(theta)*sin(theta))/(sqrt(2)*(r*r + a*a*cos(theta)*cos(theta))), ((r*(a*a + r*r)*sin(theta))/(sqrt(2)*(r*r + a*a*cos(theta)*cos(theta))))}
    };

    // Construct the matrix in tetrad components to transform 
    std::array<std::complex<double>,dim*dim> tetrad_components = {
        std::complex<double>{Kinn_h_re[0], Kinn_h_im[0]},
        std::complex<double>{Kinn_h_re[8], Kinn_h_im[8]}, 
        std::complex<double>{Kinn_h_re[2], Kinn_h_im[2]},
        std::complex<double>{Kinn_h_re[3], Kinn_h_im[3]},

        std::complex<double>{Kinn_h_re[8], Kinn_h_im[8]},
        std::complex<double>{Kinn_h_re[4], Kinn_h_im[4]},
        std::complex<double>{Kinn_h_re[5], Kinn_h_im[5]},
        std::complex<double>{Kinn_h_re[6], Kinn_h_im[6]},

        std::complex<double>{Kinn_h_re[2], Kinn_h_im[2]},
        std::complex<double>{Kinn_h_re[5], Kinn_h_im[5]},
        std::complex<double>{Kinn_h_re[7], Kinn_h_im[7]},
        std::complex<double>{Kinn_h_re[1], Kinn_h_im[1]},

        std::complex<double>{Kinn_h_re[3], Kinn_h_im[3]},
        std::complex<double>{Kinn_h_re[6], Kinn_h_im[6]},
        std::complex<double>{Kinn_h_re[1], Kinn_h_im[1]},
        std::complex<double>{Kinn_h_re[9], Kinn_h_im[9]},
    };

    //Declare the result of C=l*h
    std::array<std::complex<double>, dim*dim> C;
    //Declare result D=C(lTranspose)
    std::array<std::complex<double>, dim*dim> D;

    //alpha and beta must be set to conform to the cblas matrix multiplication function
    const std::complex<double> alpha = {1.0,0};
    const std::complex<double> beta = {0.0,0.0};
    cblas_zgemm(
        CblasRowMajor, CblasNoTrans, CblasNoTrans,
        dim, dim, dim, &alpha,   // alpha
        tetrad_trans_matrix.data(), dim,
        tetrad_components.data(), dim,
        &beta,   // beta
        C.data(), dim
    );
    cblas_zgemm(
        CblasRowMajor, CblasNoTrans, CblasTrans,
        dim, dim, dim, &alpha,   // alpha
        C.data(), dim,
        tetrad_trans_matrix.data(), dim,
        &beta,   // beta
        D.data(), dim
    );

    // Return elements of result with same ordering used by spectre
    hS_re[0] = D[0].real();
    hS_re[1] = D[1].real();
    hS_re[2] = D[2].real();
    hS_re[3] = D[3].real();
    hS_re[4] = D[5].real();
    hS_re[5] = D[6].real();
    hS_re[6] = D[7].real();
    hS_re[7] = D[10].real();
    hS_re[8] = D[11].real();
    hS_re[9] = D[15].real();

    hS_im[0] = D[0].imag();
    hS_im[1] = D[1].imag();
    hS_im[2] = D[2].imag();
    hS_im[3] = D[3].imag();
    hS_im[4] = D[5].imag();
    hS_im[5] = D[6].imag();
    hS_im[6] = D[7].imag();
    hS_im[7] = D[10].imag();
    hS_im[8] = D[11].imag();
    hS_im[9] = D[15].imag();

    // Transform the effective source
    std::array<std::complex<double>,dim*dim> src_tetrad_components = {
        std::complex<double>{src_re_tetrad[0], src_im_tetrad[0]},
        std::complex<double>{src_re_tetrad[1], src_im_tetrad[1]}, 
        std::complex<double>{src_re_tetrad[2], src_im_tetrad[2]},
        std::complex<double>{src_re_tetrad[3], src_im_tetrad[3]},

        std::complex<double>{src_re_tetrad[1], src_im_tetrad[1]},
        std::complex<double>{src_re_tetrad[4], src_im_tetrad[4]},
        std::complex<double>{src_re_tetrad[5], src_im_tetrad[5]},
        std::complex<double>{src_re_tetrad[6], src_im_tetrad[6]},

        std::complex<double>{src_re_tetrad[2], src_im_tetrad[2]},
        std::complex<double>{src_re_tetrad[5], src_im_tetrad[5]},
        std::complex<double>{src_re_tetrad[7], src_im_tetrad[7]},
        std::complex<double>{src_re_tetrad[8], src_im_tetrad[8]},

        std::complex<double>{src_re_tetrad[3], src_im_tetrad[3]},
        std::complex<double>{src_re_tetrad[6], src_im_tetrad[6]},
        std::complex<double>{src_re_tetrad[8], src_im_tetrad[8]},
        std::complex<double>{src_re_tetrad[9], src_im_tetrad[9]},
    };

    //Declare the result of C=l*Effsrc
    std::array<std::complex<double>, dim*dim> l_effsrc;
    //Declare result D=C(lTranspose)
    std::array<std::complex<double>, dim*dim> effsrc_BL;

    //alpha and beta must be set to conform to the cblas matrix multiplication function
    cblas_zgemm(
        CblasRowMajor, CblasNoTrans, CblasNoTrans,
        dim, dim, dim, &alpha,   // alpha
        tetrad_trans_matrix.data(), dim,
        src_tetrad_components.data(), dim,
        &beta,   // beta
        l_effsrc.data(), dim
    );
    cblas_zgemm(
        CblasRowMajor, CblasNoTrans, CblasTrans,
        dim, dim, dim, &alpha,   // alpha
        l_effsrc.data(), dim,
        tetrad_trans_matrix.data(), dim,
        &beta,   // beta
        effsrc_BL.data(), dim
    );

    // Return elements of result with same ordering used by spectre
    src_re[0] = effsrc_BL[0].real();
    src_re[1] = effsrc_BL[1].real();
    src_re[2] = effsrc_BL[2].real();
    src_re[3] = effsrc_BL[3].real();
    src_re[4] = effsrc_BL[5].real();
    src_re[5] = effsrc_BL[6].real();
    src_re[6] = effsrc_BL[7].real();
    src_re[7] = effsrc_BL[10].real();
    src_re[8] = effsrc_BL[11].real();
    src_re[9] = effsrc_BL[15].real();

    src_im[0] = effsrc_BL[0].imag();
    src_im[1] = effsrc_BL[1].imag();
    src_im[2] = effsrc_BL[2].imag();
    src_im[3] = effsrc_BL[3].imag();
    src_im[4] = effsrc_BL[5].imag();
    src_im[5] = effsrc_BL[6].imag();
    src_im[6] = effsrc_BL[7].imag();
    src_im[7] = effsrc_BL[10].imag();
    src_im[8] = effsrc_BL[11].imag();
    src_im[9] = effsrc_BL[15].imag();

    //Construct radial derivative matrix de_dr (derivative of tetrad wrt r)
    std::complex<double> dX_dr = {r0/sqrt(a*a - 2*M*r0 + r0*r0), 0};
    const double Sigma_sq = (r*r + a*a*cos(theta)*cos(theta))*(r*r + a*a*cos(theta)*cos(theta));
    const double Delta = (a*a + r*(-2*M + r));

    std::array<std::complex<double>, dim*dim> radial_deriv_lmatrix = {
        std::complex<double>{(r*(-a*a + M*r) + a*a*(-M + r)*cos(theta)*cos(theta))/Sigma_sq, 0},
        std::complex<double>{0,0},
        std::complex<double>{(sqrt(2)*a*a*r*cos(theta)*sin(theta))/Sigma_sq, a*(-r*r + a*a*cos(theta)*cos(theta))*sin(theta)/(sqrt(2)*Sigma_sq)},
        std::complex<double>{(sqrt(2)*a*a*r*cos(theta)*sin(theta))/Sigma_sq, -a*(-r*r + a*a*cos(theta)*cos(theta))*sin(theta)/(sqrt(2)*Sigma_sq)},

        std::complex<double>{0,0},
        std::complex<double>{(2*r*(-a*a + M*r) + 2*a*a*(-M + r)*cos(theta)*cos(theta))/(Delta*Delta),0},
        std::complex<double>{0,0},
        std::complex<double>{0,0},

        std::complex<double>{0,0},
        std::complex<double>{0,0},
        std::complex<double>{1/sqrt(2),0},
        std::complex<double>{1/sqrt(2),0},

        std::complex<double>{(a*(r*(a*a - M*r) + a*a*(M - r)*cos(theta)*cos(theta))*sin(theta)*sin(theta))/Sigma_sq},
        std::complex<double>{0,0},
        std::complex<double>{(-sqrt(2)*a*a*a*r*cos(theta)*sin(theta)*sin(theta)*sin(theta))/Sigma_sq, (-(-a*a*r*r + r*r*r*r + a*a*(a*a + 3*r*r)*cos(theta)*cos(theta))*sin(theta))/(sqrt(2)*Sigma_sq)},
        std::complex<double>{-(sqrt(2)*a*a*a*r*cos(theta)*sin(theta)*sin(theta)*sin(theta))/Sigma_sq, ((-a*a*r*r + r*r*r*r + a*a*(a*a + 3*r*r)*cos(theta)*cos(theta))*sin(theta))/(sqrt(2)*Sigma_sq)}
    };

    //Construct theta derivative matrix de_dtheta (derivative of tetrad wrt theta)
    std::complex<double> dY_dtheta = {r0*sin(theta), 0};

    std::array<std::complex<double>, dim*dim> theta_deriv_lmatrix = {
        std::complex<double>{a*a*(a*a + r*(-2*M + r))*cos(theta)*sin(theta)/Sigma_sq,0},
        std::complex<double>{0,0},
        std::complex<double>{(a*a*(r*r - (a*a + 2*r*r)*cos(theta)*cos(theta)))/(sqrt(2)*Sigma_sq), a*r*cos(theta)*(2*a*a + r*r - a*a*cos(theta)*cos(theta))/(sqrt(2)*Sigma_sq)},
        std::complex<double>{(a*a*(r*r - (a*a + 2*r*r)*cos(theta)*cos(theta)))/(sqrt(2)*Sigma_sq), -a*r*cos(theta)*(2*a*a + r*r - a*a*cos(theta)*cos(theta))/(sqrt(2)*Sigma_sq)},

        std::complex<double>{0,0},
        std::complex<double>{a*a*sin(2*theta)/Delta,0},
        std::complex<double>{0,0},
        std::complex<double>{0,0},

        std::complex<double>{0,0},
        std::complex<double>{0,0},
        std::complex<double>{0,-((a*sin(theta))/sqrt(2))},
        std::complex<double>{0,((a*sin(theta))/sqrt(2))},

        std::complex<double>{-(a*(a*a + r*r)*(a*a + r*(-2*M + r))*cos(theta)*sin(theta))/Sigma_sq, 0},
        std::complex<double>{-a*sin(2*theta), 0},
        std::complex<double>{(a*(a*a + r*r)*(a*a + (a*a + 2*r*r)*cos(2*theta)))/(2*sqrt(2)*Sigma_sq), (r*(a*a + r*r)*cos(theta)*(-3*a*a - 2*r*r + a*a*cos(2*theta)))/(2*sqrt(2)*Sigma_sq)},
        std::complex<double>{(a*(a*a + r*r)*(a*a + (a*a + 2*r*r)*cos(2*theta)))/(2*sqrt(2)*Sigma_sq), -(r*(a*a + r*r)*cos(theta)*(-3*a*a - 2*r*r + a*a*cos(2*theta)))/(2*sqrt(2)*Sigma_sq)},
    };

    // Construct dh_dX matrix (derivatives of tetrad copmonents of h wrt X)
    std::array<std::complex<double>,dim*dim> dh_dX_matrix = {
        std::complex<double>{dh_dX_re[0], dh_dX_im[0]},
        std::complex<double>{dh_dX_re[8], dh_dX_im[8]}, 
        std::complex<double>{dh_dX_re[2], dh_dX_im[2]},
        std::complex<double>{dh_dX_re[3], dh_dX_im[3]},

        std::complex<double>{dh_dX_re[8], dh_dX_im[8]},
        std::complex<double>{dh_dX_re[4], dh_dX_im[4]},
        std::complex<double>{dh_dX_re[5], dh_dX_im[5]},
        std::complex<double>{dh_dX_re[6], dh_dX_im[6]},

        std::complex<double>{dh_dX_re[2], dh_dX_im[2]},
        std::complex<double>{dh_dX_re[5], dh_dX_im[5]},
        std::complex<double>{dh_dX_re[7], dh_dX_im[7]},
        std::complex<double>{dh_dX_re[1], dh_dX_im[1]},

        std::complex<double>{dh_dX_re[3], dh_dX_im[3]},
        std::complex<double>{dh_dX_re[6], dh_dX_im[6]},
        std::complex<double>{dh_dX_re[1], dh_dX_im[1]},
        std::complex<double>{dh_dX_re[9], dh_dX_im[9]},
    };

    //Construct dh_dY matrix (derivatives of tetrad components of h wrt Y)
    std::array<std::complex<double>,dim*dim> dh_dY_matrix = {
        std::complex<double>{dh_dY_re[0], dh_dY_im[0]},
        std::complex<double>{dh_dY_re[8], dh_dY_im[8]}, 
        std::complex<double>{dh_dY_re[2], dh_dY_im[2]},
        std::complex<double>{dh_dY_re[3], dh_dY_im[3]},

        std::complex<double>{dh_dY_re[8], dh_dY_im[8]},
        std::complex<double>{dh_dY_re[4], dh_dY_im[4]},
        std::complex<double>{dh_dY_re[5], dh_dY_im[5]},
        std::complex<double>{dh_dY_re[6], dh_dY_im[6]},

        std::complex<double>{dh_dY_re[2], dh_dY_im[2]},
        std::complex<double>{dh_dY_re[5], dh_dY_im[5]},
        std::complex<double>{dh_dY_re[7], dh_dY_im[7]},
        std::complex<double>{dh_dY_re[1], dh_dY_im[1]},

        std::complex<double>{dh_dY_re[3], dh_dY_im[3]},
        std::complex<double>{dh_dY_re[6], dh_dY_im[6]},
        std::complex<double>{dh_dY_re[1], dh_dY_im[1]},
        std::complex<double>{dh_dY_re[9], dh_dY_im[9]},
    };

    //Compute (Dl)(h)(lT) + (l)(h)(DlT) + (l)(dX_dr)(dh_dX)(lT)
    // First term
    std::array<std::complex<double>, dim*dim> Dl_h;
    std::array<std::complex<double>, dim*dim> Dl_h_lT;
    cblas_zgemm(
        CblasRowMajor, CblasNoTrans, CblasNoTrans, dim, dim, dim, &alpha,
        radial_deriv_lmatrix.data(), dim, tetrad_components.data(), dim,
        &beta, Dl_h.data(), dim
    ); 
    cblas_zgemm(
        CblasRowMajor, CblasNoTrans, CblasTrans, dim, dim, dim, &alpha,
        Dl_h.data(), dim, tetrad_trans_matrix.data(), dim, &beta,
        Dl_h_lT.data(), dim
    );

    //Second term
    std::array<std::complex<double>, dim*dim> l_h;
    std::array<std::complex<double>, dim*dim> l_h_dlT;
    cblas_zgemm(
        CblasRowMajor, CblasNoTrans, CblasNoTrans, dim, dim, dim, &alpha,
        tetrad_trans_matrix.data(), dim, tetrad_components.data(), dim,
        &beta, l_h.data(), dim
    );
    cblas_zgemm(
        CblasRowMajor, CblasNoTrans, CblasTrans, dim, dim, dim, &alpha,
        l_h.data(), dim, radial_deriv_lmatrix.data(), dim,
        &beta, l_h_dlT.data(), dim
    );

    //Third term
    std::array<std::complex<double>, dim*dim> l_dh_dx;
    std::array<std::complex<double>, dim*dim> l_dh_dx_lT;
    cblas_zgemm(
        CblasRowMajor, CblasNoTrans, CblasNoTrans, dim, dim, dim, &alpha,
        tetrad_trans_matrix.data(), dim, dh_dX_matrix.data(), dim,
        &beta, l_dh_dx.data(), dim
    );
    cblas_zgemm(
        CblasRowMajor, CblasNoTrans, CblasTrans, dim, dim, dim, &dX_dr,
        l_dh_dx.data(), dim, tetrad_trans_matrix.data(), dim,
        &beta, l_dh_dx_lT.data(), dim
    );

    dhS_dr_re[0] = Dl_h_lT[0].real() + l_h_dlT[0].real() + l_dh_dx_lT[0].real();
    dhS_dr_re[1] = Dl_h_lT[1].real() + l_h_dlT[1].real() + l_dh_dx_lT[1].real();
    dhS_dr_re[2] = Dl_h_lT[2].real() + l_h_dlT[2].real() + l_dh_dx_lT[2].real();
    dhS_dr_re[3] = Dl_h_lT[3].real() + l_h_dlT[3].real() + l_dh_dx_lT[3].real();
    dhS_dr_re[4] = Dl_h_lT[5].real() + l_h_dlT[5].real() + l_dh_dx_lT[5].real();
    dhS_dr_re[5] = Dl_h_lT[6].real() + l_h_dlT[6].real() + l_dh_dx_lT[6].real();
    dhS_dr_re[6] = Dl_h_lT[7].real() + l_h_dlT[7].real() + l_dh_dx_lT[7].real();
    dhS_dr_re[7] = Dl_h_lT[10].real() + l_h_dlT[10].real() + l_dh_dx_lT[10].real();
    dhS_dr_re[8] = Dl_h_lT[11].real() + l_h_dlT[11].real() + l_dh_dx_lT[11].real();
    dhS_dr_re[9] = Dl_h_lT[15].real() + l_h_dlT[15].real() + l_dh_dx_lT[15].real();

    dhS_dr_im[0] = Dl_h_lT[0].imag() + l_h_dlT[0].imag() + l_dh_dx_lT[0].imag();
    dhS_dr_im[1] = Dl_h_lT[1].imag() + l_h_dlT[1].imag() + l_dh_dx_lT[1].imag();
    dhS_dr_im[2] = Dl_h_lT[2].imag() + l_h_dlT[2].imag() + l_dh_dx_lT[2].imag();
    dhS_dr_im[3] = Dl_h_lT[3].imag() + l_h_dlT[3].imag() + l_dh_dx_lT[3].imag();
    dhS_dr_im[4] = Dl_h_lT[5].imag() + l_h_dlT[5].imag() + l_dh_dx_lT[5].imag();
    dhS_dr_im[5] = Dl_h_lT[6].imag() + l_h_dlT[6].imag() + l_dh_dx_lT[6].imag();
    dhS_dr_im[6] = Dl_h_lT[7].imag() + l_h_dlT[7].imag() + l_dh_dx_lT[7].imag();
    dhS_dr_im[7] = Dl_h_lT[10].imag() + l_h_dlT[10].imag() + l_dh_dx_lT[10].imag();
    dhS_dr_im[8] = Dl_h_lT[11].imag() + l_h_dlT[11].imag() + l_dh_dx_lT[11].imag();
    dhS_dr_im[9] = Dl_h_lT[15].imag() + l_h_dlT[15].imag() + l_dh_dx_lT[15].imag();

    //Derivative with resepect to theta
    std::array<std::complex<double>, dim*dim> Dlth_h;
    std::array<std::complex<double>, dim*dim> Dlth_h_lT;
    cblas_zgemm(
        CblasRowMajor, CblasNoTrans, CblasNoTrans, dim, dim, dim, &alpha,
        theta_deriv_lmatrix.data(), dim, tetrad_components.data(), dim,
        &beta, Dlth_h.data(), dim
    ); 
    cblas_zgemm(
        CblasRowMajor, CblasNoTrans, CblasTrans, dim, dim, dim, &alpha,
        Dlth_h.data(), dim, tetrad_trans_matrix.data(), dim, &beta,
        Dlth_h_lT.data(), dim
    );

    //Second term
    std::array<std::complex<double>, dim*dim> l_h_dlthT;
    cblas_zgemm(
        CblasRowMajor, CblasNoTrans, CblasTrans, dim, dim, dim, &alpha,
        l_h.data(), dim, theta_deriv_lmatrix.data(), dim,
        &beta, l_h_dlthT.data(), dim
    );

    //Third term
    std::array<std::complex<double>, dim*dim> l_dh_dY;
    std::array<std::complex<double>, dim*dim> l_dh_dY_lT;
    cblas_zgemm(
        CblasRowMajor, CblasNoTrans, CblasNoTrans, dim, dim, dim, &alpha,
        tetrad_trans_matrix.data(), dim, dh_dY_matrix.data(), dim,
        &beta, l_dh_dY.data(), dim
    );
    cblas_zgemm(
        CblasRowMajor, CblasNoTrans, CblasTrans, dim, dim, dim, &dY_dtheta,
        l_dh_dY.data(), dim, tetrad_trans_matrix.data(), dim,
        &beta, l_dh_dY_lT.data(), dim
    );

    dhS_dtheta_re[0] = Dlth_h_lT[0].real() + l_h_dlthT[0].real() + l_dh_dY_lT[0].real();
    dhS_dtheta_re[1] = Dlth_h_lT[1].real() + l_h_dlthT[1].real() + l_dh_dY_lT[1].real();
    dhS_dtheta_re[2] = Dlth_h_lT[2].real() + l_h_dlthT[2].real() + l_dh_dY_lT[2].real();
    dhS_dtheta_re[3] = Dlth_h_lT[3].real() + l_h_dlthT[3].real() + l_dh_dY_lT[3].real();
    dhS_dtheta_re[4] = Dlth_h_lT[5].real() + l_h_dlthT[5].real() + l_dh_dY_lT[5].real();
    dhS_dtheta_re[5] = Dlth_h_lT[6].real() + l_h_dlthT[6].real() + l_dh_dY_lT[6].real();
    dhS_dtheta_re[6] = Dlth_h_lT[7].real() + l_h_dlthT[7].real() + l_dh_dY_lT[7].real();
    dhS_dtheta_re[7] = Dlth_h_lT[10].real() + l_h_dlthT[10].real() + l_dh_dY_lT[10].real();
    dhS_dtheta_re[8] = Dlth_h_lT[11].real() + l_h_dlthT[11].real() + l_dh_dY_lT[11].real();
    dhS_dtheta_re[9] = Dlth_h_lT[15].real() + l_h_dlthT[15].real() + l_dh_dY_lT[15].real();

    dhS_dtheta_im[0] = Dlth_h_lT[0].imag() + l_h_dlthT[0].imag() + l_dh_dY_lT[0].imag();
    dhS_dtheta_im[1] = Dlth_h_lT[1].imag() + l_h_dlthT[1].imag() + l_dh_dY_lT[1].imag();
    dhS_dtheta_im[2] = Dlth_h_lT[2].imag() + l_h_dlthT[2].imag() + l_dh_dY_lT[2].imag();
    dhS_dtheta_im[3] = Dlth_h_lT[3].imag() + l_h_dlthT[3].imag() + l_dh_dY_lT[3].imag();
    dhS_dtheta_im[4] = Dlth_h_lT[5].imag() + l_h_dlthT[5].imag() + l_dh_dY_lT[5].imag();
    dhS_dtheta_im[5] = Dlth_h_lT[6].imag() + l_h_dlthT[6].imag() + l_dh_dY_lT[6].imag();
    dhS_dtheta_im[6] = Dlth_h_lT[7].imag() + l_h_dlthT[7].imag() + l_dh_dY_lT[7].imag();
    dhS_dtheta_im[7] = Dlth_h_lT[10].imag() + l_h_dlthT[10].imag() + l_dh_dY_lT[10].imag();
    dhS_dtheta_im[8] = Dlth_h_lT[11].imag() + l_h_dlthT[11].imag() + l_dh_dY_lT[11].imag();
    dhS_dtheta_im[9] = Dlth_h_lT[15].imag() + l_h_dlthT[15].imag() + l_dh_dY_lT[15].imag();
}
// NOLINTEND
