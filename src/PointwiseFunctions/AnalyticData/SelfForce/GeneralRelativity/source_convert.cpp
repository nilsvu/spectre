int psiTommy(int m, double a, double r, double th, double realBarry[10],  double imagBarry[10], double realTommy[10], double imagTommy[10]){
double factor = r/(2*M_PI);
double rplus = 1 + sqrt(1 - a * a);
double rminus = 1 - sqrt(1 - a * a);
double mdphi = m * a / (rplus - rminus) * log((r - rplus) / (r - rminus));
double cosmdphi = cos(mdphi);
double sinmdphi = sin(mdphi);
realTommy[0] = factor*(realBarry[0]*cosmdphi+imagBarry[0]*sinmdphi);
imagTommy[0] = factor*(imagBarry[0]*cosmdphi-realBarry[0]*sinmdphi);
realTommy[1] = factor*(realBarry[1]*cosmdphi+imagBarry[1]*sinmdphi)*(r*r-2*r+a*a)/(r*r);
imagTommy[1] = factor*(imagBarry[1]*cosmdphi-realBarry[1]*sinmdphi)*(r*r-2*r+a*a)/(r*r);
realTommy[2] = factor*(realBarry[2]*cosmdphi+imagBarry[2]*sinmdphi)/r;
imagTommy[2] = factor*(imagBarry[2]*cosmdphi-realBarry[2]*sinmdphi)/r;
realTommy[3] = factor*(realBarry[3]*cosmdphi+imagBarry[3]*sinmdphi)/(r*sin(th));
imagTommy[3] = factor*(imagBarry[3]*cosmdphi-realBarry[3]*sinmdphi)/(r*sin(th));
realTommy[4] = factor*(realBarry[4]*cosmdphi+imagBarry[4]*sinmdphi)*(r*r-2*r+a*a)*(r*r-2*r+a*a)/(r*r*r*r);
imagTommy[4] = factor*(imagBarry[4]*cosmdphi-realBarry[4]*sinmdphi)*(r*r-2*r+a*a)*(r*r-2*r+a*a)/(r*r*r*r);
realTommy[5] = factor*(realBarry[5]*cosmdphi+imagBarry[5]*sinmdphi)*(r*r-2*r+a*a)/(r*r*r);
imagTommy[5] = factor*(imagBarry[5]*cosmdphi-realBarry[5]*sinmdphi)*(r*r-2*r+a*a)/(r*r*r);
realTommy[6] = factor*(realBarry[6]*cosmdphi+imagBarry[6]*sinmdphi)*(r*r-2*r+a*a)/(r*r*r*sin(th));
imagTommy[6] = factor*(imagBarry[6]*cosmdphi-realBarry[6]*sinmdphi)*(r*r-2*r+a*a)/(r*r*r*sin(th));
realTommy[7] = factor*(realBarry[7]*cosmdphi+imagBarry[7]*sinmdphi)/(r*r);
imagTommy[7] = factor*(imagBarry[7]*cosmdphi-realBarry[7]*sinmdphi)/(r*r);
realTommy[8] = factor*(realBarry[8]*cosmdphi+imagBarry[8]*sinmdphi)/(r*r*sin(th));
imagTommy[8] = factor*(imagBarry[8]*cosmdphi-realBarry[8]*sinmdphi)/(r*r*sin(th));
realTommy[9] = factor*(realBarry[9]*cosmdphi+imagBarry[9]*sinmdphi)/(r*r*sin(th)*sin(th));
imagTommy[9] = factor*(imagBarry[9]*cosmdphi-realBarry[9]*sinmdphi)/(r*r*sin(th)*sin(th));
return 0;
}

int SeffTommy(int m, double a, double r, double th, double realBarry[10],  double imagBarry[10], double realTommy[10], double imagTommy[10]){
double factor = 1.0/(2*M_PI);
double rplus = 1+sqrt(1-a*a);
double rminus = 1-sqrt(1-a*a);
double mdphi = m*a/(rplus-rminus)*log((r-rplus)/(r-rminus));
double cosmdphi = cos(mdphi);
double sinmdphi = sin(mdphi);
realTommy[0] = factor*(realBarry[0]*cosmdphi+imagBarry[0]*sinmdphi)*(-((r*(a*a+(-2+r)*r)*(r*r+a*a*cos(th)*cos(th)))/((a*a+r*r)*(a*a+r*r))));
imagTommy[0] = factor*(imagBarry[0]*cosmdphi-realBarry[0]*sinmdphi)*(-((r*(a*a+(-2+r)*r)*(r*r+a*a*cos(th)*cos(th)))/((a*a+r*r)*(a*a+r*r))));
realTommy[1] = factor*(realBarry[1]*cosmdphi+imagBarry[1]*sinmdphi)*(-0.5*((a*a+(-2+r)*r)*(a*a+(-2+r)*r)*(a*a+2*r*r+a*a*cos(2*th)))/(r*(a*a+r*r)*(a*a+r*r)));
imagTommy[1] = factor*(imagBarry[1]*cosmdphi-realBarry[1]*sinmdphi)*(-0.5*((a*a+(-2+r)*r)*(a*a+(-2+r)*r)*(a*a+2*r*r+a*a*cos(2*th)))/(r*(a*a+r*r)*(a*a+r*r)));
realTommy[2] = factor*(realBarry[2]*cosmdphi+imagBarry[2]*sinmdphi)*(-(((a*a+(-2+r)*r)*(r*r+a*a*cos(th)*cos(th)))/((a*a+r*r)*(a*a+r*r))));
imagTommy[2] = factor*(imagBarry[2]*cosmdphi-realBarry[2]*sinmdphi)*(-(((a*a+(-2+r)*r)*(r*r+a*a*cos(th)*cos(th)))/((a*a+r*r)*(a*a+r*r))));
realTommy[3] = factor*(realBarry[3]*cosmdphi+imagBarry[3]*sinmdphi)*(-(((a*a+(-2+r)*r)*(r*r+a*a*cos(th)*cos(th)))/(sin(th)*(a*a+r*r)*(a*a+r*r))));
imagTommy[3] = factor*(imagBarry[3]*cosmdphi-realBarry[3]*sinmdphi)*(-(((a*a+(-2+r)*r)*(r*r+a*a*cos(th)*cos(th)))/(sin(th)*(a*a+r*r)*(a*a+r*r))));
realTommy[4] = factor*(realBarry[4]*cosmdphi+imagBarry[4]*sinmdphi)*(-0.5*((a*a+(-2+r)*r)*(a*a+(-2+r)*r)*(a*a+(-2+r)*r)*(a*a+2*r*r+a*a*cos(2*th)))/(r*r*r*(a*a+r*r)*(a*a+r*r)));
imagTommy[4] = factor*(imagBarry[4]*cosmdphi-realBarry[4]*sinmdphi)*(-0.5*((a*a+(-2+r)*r)*(a*a+(-2+r)*r)*(a*a+(-2+r)*r)*(a*a+2*r*r+a*a*cos(2*th)))/(r*r*r*(a*a+r*r)*(a*a+r*r)));
realTommy[5] = factor*(realBarry[5]*cosmdphi+imagBarry[5]*sinmdphi)*(-0.5*((a*a+(-2+r)*r)*(a*a+(-2+r)*r)*(a*a+2*r*r+a*a*cos(2*th)))/(r*r*(a*a+r*r)*(a*a+r*r)));
imagTommy[5] = factor*(imagBarry[5]*cosmdphi-realBarry[5]*sinmdphi)*(-0.5*((a*a+(-2+r)*r)*(a*a+(-2+r)*r)*(a*a+2*r*r+a*a*cos(2*th)))/(r*r*(a*a+r*r)*(a*a+r*r)));
realTommy[6] = factor*(realBarry[6]*cosmdphi+imagBarry[6]*sinmdphi)*(-(((a*a+(-2+r)*r)*(a*a+(-2+r)*r)*(r*r+a*a*cos(th)*cos(th)))/(sin(th)*r*r*(a*a+r*r)*(a*a+r*r))));
imagTommy[6] = factor*(imagBarry[6]*cosmdphi-realBarry[6]*sinmdphi)*(-(((a*a+(-2+r)*r)*(a*a+(-2+r)*r)*(r*r+a*a*cos(th)*cos(th)))/(sin(th)*r*r*(a*a+r*r)*(a*a+r*r))));
realTommy[7] = factor*(realBarry[7]*cosmdphi+imagBarry[7]*sinmdphi)*(-(((a*a+(-2+r)*r)*(r*r+a*a*cos(th)*cos(th)))/(r*(a*a+r*r)*(a*a+r*r))));
imagTommy[7] = factor*(imagBarry[7]*cosmdphi-realBarry[7]*sinmdphi)*(-(((a*a+(-2+r)*r)*(r*r+a*a*cos(th)*cos(th)))/(r*(a*a+r*r)*(a*a+r*r))));
realTommy[8] = factor*(realBarry[8]*cosmdphi+imagBarry[8]*sinmdphi)*(-(((a*a+(-2+r)*r)*(r*r+a*a*cos(th)*cos(th)))/(sin(th)*r*(a*a+r*r)*(a*a+r*r))));
imagTommy[8] = factor*(imagBarry[8]*cosmdphi-realBarry[8]*sinmdphi)*(-(((a*a+(-2+r)*r)*(r*r+a*a*cos(th)*cos(th)))/(sin(th)*r*(a*a+r*r)*(a*a+r*r))));
realTommy[9] = factor*(realBarry[9]*cosmdphi+imagBarry[9]*sinmdphi)*(-(((a*a+(-2+r)*r)*(r*r+a*a*cos(th)*cos(th)))/(sin(th)*sin(th)*r*(a*a+r*r)*(a*a+r*r))));
imagTommy[9] = factor*(imagBarry[9]*cosmdphi-realBarry[9]*sinmdphi)*(-(((a*a+(-2+r)*r)*(r*r+a*a*cos(th)*cos(th)))/(sin(th)*sin(th)*r*(a*a+r*r)*(a*a+r*r))));
return 0;
}

int dpsidthetaTommy(int m, double a, double r, double th, double realBarry[10],  double imagBarry[10], double realBarry_dth[10],  double imagBarry_dth[10], double realTommy_dth[10], double imagTommy_dth[10]){
double factor = r/(2*M_PI);
double rplus = 1+sqrt(1-a*a);
double rminus = 1-sqrt(1-a*a);
double mdphi = m*a/(rplus-rminus)*log((r-rplus)/(r-rminus));
double cosmdphi = cos(mdphi);
double sinmdphi = sin(mdphi);
realTommy_dth[0] = factor*(realBarry_dth[0]*cosmdphi+imagBarry_dth[0]*sinmdphi);
imagTommy_dth[0] = factor*(imagBarry_dth[0]*cosmdphi-realBarry_dth[0]*sinmdphi);
realTommy_dth[1] = factor*(realBarry_dth[1]*cosmdphi+imagBarry_dth[1]*sinmdphi)*(r*r-2*r+a*a)/(r*r);
imagTommy_dth[1] = factor*(imagBarry_dth[1]*cosmdphi-realBarry_dth[1]*sinmdphi)*(r*r-2*r+a*a)/(r*r);
realTommy_dth[2] = factor*(realBarry_dth[2]*cosmdphi+imagBarry_dth[2]*sinmdphi)/r;
imagTommy_dth[2] = factor*(imagBarry_dth[2]*cosmdphi-realBarry_dth[2]*sinmdphi)/r;
realTommy_dth[3] = factor*(realBarry_dth[3]*cosmdphi+imagBarry_dth[3]*sinmdphi)/(r*sin(th)) - factor*(realBarry[3]*cosmdphi+imagBarry[3]*sinmdphi)*cos(th)/(r*sin(th)*sin(th));
imagTommy_dth[3] = factor*(imagBarry_dth[3]*cosmdphi-realBarry_dth[3]*sinmdphi)/(r*sin(th)) - factor*(imagBarry[3]*cosmdphi-realBarry[3]*sinmdphi)*cos(th)/(r*sin(th)*sin(th));
realTommy_dth[4] = factor*(realBarry_dth[4]*cosmdphi+imagBarry_dth[4]*sinmdphi)*(r*r-2*r+a*a)*(r*r-2*r+a*a)/(r*r*r*r);
imagTommy_dth[4] = factor*(imagBarry_dth[4]*cosmdphi-realBarry_dth[4]*sinmdphi)*(r*r-2*r+a*a)*(r*r-2*r+a*a)/(r*r*r*r);
realTommy_dth[5] = factor*(realBarry_dth[5]*cosmdphi+imagBarry_dth[5]*sinmdphi)*(r*r-2*r+a*a)/(r*r*r);
imagTommy_dth[5] = factor*(imagBarry_dth[5]*cosmdphi-realBarry_dth[5]*sinmdphi)*(r*r-2*r+a*a)/(r*r*r);
realTommy_dth[6] = factor*(realBarry_dth[6]*cosmdphi+imagBarry_dth[6]*sinmdphi)*(r*r-2*r+a*a)/(r*r*r*sin(th)) - factor*(realBarry[6]*cosmdphi+imagBarry[6]*sinmdphi)*(r*r-2*r+a*a)*cos(th)/(r*r*r*sin(th)*sin(th));
imagTommy_dth[6] = factor*(imagBarry_dth[6]*cosmdphi-realBarry_dth[6]*sinmdphi)*(r*r-2*r+a*a)/(r*r*r*sin(th)) - factor*(imagBarry[6]*cosmdphi-realBarry[6]*sinmdphi)*(r*r-2*r+a*a)*cos(th)/(r*r*r*sin(th)*sin(th));
realTommy_dth[7] = factor*(realBarry_dth[7]*cosmdphi+imagBarry_dth[7]*sinmdphi)/(r*r);
imagTommy_dth[7] = factor*(imagBarry_dth[7]*cosmdphi-realBarry_dth[7]*sinmdphi)/(r*r);
realTommy_dth[8] = factor*(realBarry_dth[8]*cosmdphi+imagBarry_dth[8]*sinmdphi)/(r*r*sin(th)) - factor*(realBarry[8]*cosmdphi+imagBarry[8]*sinmdphi)*cos(th)/(r*r*sin(th)*sin(th));
imagTommy_dth[8] = factor*(imagBarry_dth[8]*cosmdphi-realBarry_dth[8]*sinmdphi)/(r*r*sin(th)) - factor*(imagBarry[8]*cosmdphi-realBarry[8]*sinmdphi)*cos(th)/(r*r*sin(th)*sin(th));
realTommy_dth[9] = factor*(realBarry_dth[9]*cosmdphi+imagBarry_dth[9]*sinmdphi)/(r*r*sin(th)*sin(th)) - factor*(realBarry[9]*cosmdphi+imagBarry[9]*sinmdphi)*2*cos(th)/(r*r*sin(th)*sin(th)*sin(th));
imagTommy_dth[9] = factor*(imagBarry_dth[9]*cosmdphi-realBarry_dth[9]*sinmdphi)/(r*r*sin(th)*sin(th)) - factor*(imagBarry[9]*cosmdphi-realBarry[9]*sinmdphi)*2*cos(th)/(r*r*sin(th)*sin(th)*sin(th));
return 0;
}

/* fixed by Tommy as of 6/2/2024*/
int dpsidrstarTommy(int m, double a, double r, double th, double realBarry[10],  double imagBarry[10], double realBarry_dr[10],  double imagBarry_dr[10], double realTommy_drs[10], double imagTommy_drs[10]){
double factor = (r*r-2*r+a*a)*r/((r*r+a*a)*2*M_PI);
double rplus = 1+sqrt(1-a*a);
double rminus = 1-sqrt(1-a*a);
double mdphi = m*a/(rplus-rminus)*log((r-rplus)/(r-rminus));
double mdphi_dr = m*a/(r*r-2*r+a*a);
double cosmdphi = cos(mdphi);
double sinmdphi = sin(mdphi);
realTommy_drs[0] = factor*(realBarry_dr[0]*cosmdphi+imagBarry_dr[0]*sinmdphi) + factor*(realBarry[0]*cosmdphi+imagBarry[0]*sinmdphi)/r + factor*(imagBarry[0]*cosmdphi-realBarry[0]*sinmdphi)*(mdphi_dr);
imagTommy_drs[0] = factor*(imagBarry_dr[0]*cosmdphi-realBarry_dr[0]*sinmdphi) + factor*(imagBarry[0]*cosmdphi-realBarry[0]*sinmdphi)/r + factor*(-realBarry[0]*cosmdphi-imagBarry[0]*sinmdphi)*(mdphi_dr);
realTommy_drs[1] = factor*(realBarry_dr[1]*cosmdphi+imagBarry_dr[1]*sinmdphi)*(r*r-2*r+a*a)/(r*r) + factor*(realBarry[1]*cosmdphi+imagBarry[1]*sinmdphi)*(r*r-a*a)/(r*r*r) + factor*(imagBarry[1]*cosmdphi-realBarry[1]*sinmdphi)*(mdphi_dr*(r*r-2*r+a*a)/(r*r));
imagTommy_drs[1] = factor*(imagBarry_dr[1]*cosmdphi-realBarry_dr[1]*sinmdphi)*(r*r-2*r+a*a)/(r*r) + factor*(imagBarry[1]*cosmdphi-realBarry[1]*sinmdphi)*(r*r-a*a)/(r*r*r) + factor*(-realBarry[1]*cosmdphi-imagBarry[1]*sinmdphi)*(mdphi_dr*(r*r-2*r+a*a)/(r*r));
realTommy_drs[2] = factor*(realBarry_dr[2]*cosmdphi+imagBarry_dr[2]*sinmdphi)/r + factor*(imagBarry[2]*cosmdphi-realBarry[2]*sinmdphi)*(mdphi_dr/r);
imagTommy_drs[2] = factor*(imagBarry_dr[2]*cosmdphi-realBarry_dr[2]*sinmdphi)/r + factor*(-realBarry[2]*cosmdphi-imagBarry[2]*sinmdphi)*(mdphi_dr/r);
realTommy_drs[3] = factor*(realBarry_dr[3]*cosmdphi+imagBarry_dr[3]*sinmdphi)/(r*sin(th)) + factor*(imagBarry[3]*cosmdphi-realBarry[3]*sinmdphi)*(mdphi_dr/(r*sin(th)));
imagTommy_drs[3] = factor*(imagBarry_dr[3]*cosmdphi-realBarry_dr[3]*sinmdphi)/(r*sin(th)) + factor*(-realBarry[3]*cosmdphi-imagBarry[3]*sinmdphi)*(mdphi_dr/(r*sin(th)));
realTommy_drs[4] = factor*(realBarry_dr[4]*cosmdphi+imagBarry_dr[4]*sinmdphi)*(r*r-2*r+a*a)*(r*r-2*r+a*a)/(r*r*r*r) + factor*(realBarry[4]*cosmdphi+imagBarry[4]*sinmdphi)*(r*r-2*r+a*a)*(r*(2+r)-3*a*a)/(r*r*r*r*r) + factor*(imagBarry[4]*cosmdphi-realBarry[4]*sinmdphi)*(mdphi_dr*(r*r-2*r+a*a)*(r*r-2*r+a*a)/(r*r*r*r));
imagTommy_drs[4] = factor*(imagBarry_dr[4]*cosmdphi-realBarry_dr[4]*sinmdphi)*(r*r-2*r+a*a)*(r*r-2*r+a*a)/(r*r*r*r) + factor*(imagBarry[4]*cosmdphi-realBarry[4]*sinmdphi)*(r*r-2*r+a*a)*(r*(2+r)-3*a*a)/(r*r*r*r*r) + factor*(-realBarry[4]*cosmdphi-imagBarry[4]*sinmdphi)*(mdphi_dr*(r*r-2*r+a*a)*(r*r-2*r+a*a)/(r*r*r*r));
realTommy_drs[5] = factor*(realBarry_dr[5]*cosmdphi+imagBarry_dr[5]*sinmdphi)*(r*r-2*r+a*a)/(r*r*r) + factor*(realBarry[5]*cosmdphi+imagBarry[5]*sinmdphi)*2*(r-a*a)/(r*r*r*r) + factor*(imagBarry[5]*cosmdphi-realBarry[5]*sinmdphi)*(mdphi_dr*(r*r-2*r+a*a)/(r*r*r));
imagTommy_drs[5] = factor*(imagBarry_dr[5]*cosmdphi-realBarry_dr[5]*sinmdphi)*(r*r-2*r+a*a)/(r*r*r) + factor*(imagBarry[5]*cosmdphi-realBarry[5]*sinmdphi)*2*(r-a*a)/(r*r*r*r) + factor*(-realBarry[5]*cosmdphi-imagBarry[5]*sinmdphi)*(mdphi_dr*(r*r-2*r+a*a)/(r*r*r));
realTommy_drs[6] = factor*(realBarry_dr[6]*cosmdphi+imagBarry_dr[6]*sinmdphi)*(r*r-2*r+a*a)/(r*r*r*sin(th)) + factor*(realBarry[6]*cosmdphi+imagBarry[6]*sinmdphi)*2*(r-a*a)/(sin(th)*r*r*r*r) + factor*(imagBarry[6]*cosmdphi-realBarry[6]*sinmdphi)*(mdphi_dr*(r*r-2*r+a*a)/(sin(th)*r*r*r));
imagTommy_drs[6] = factor*(imagBarry_dr[6]*cosmdphi-realBarry_dr[6]*sinmdphi)*(r*r-2*r+a*a)/(r*r*r*sin(th)) + factor*(imagBarry[6]*cosmdphi-realBarry[6]*sinmdphi)*2*(r-a*a)/(sin(th)*r*r*r*r) + factor*(-realBarry[6]*cosmdphi-imagBarry[6]*sinmdphi)*(mdphi_dr*(r*r-2*r+a*a)/(sin(th)*r*r*r));
realTommy_drs[7] = factor*(realBarry_dr[7]*cosmdphi+imagBarry_dr[7]*sinmdphi)/(r*r) + factor*(realBarry[7]*cosmdphi+imagBarry[7]*sinmdphi)/(-r*r*r) + factor*(imagBarry[7]*cosmdphi-realBarry[7]*sinmdphi)*(mdphi_dr/(r*r));
imagTommy_drs[7] = factor*(imagBarry_dr[7]*cosmdphi-realBarry_dr[7]*sinmdphi)/(r*r) + factor*(imagBarry[7]*cosmdphi-realBarry[7]*sinmdphi)/(-r*r*r) + factor*(-realBarry[7]*cosmdphi-imagBarry[7]*sinmdphi)*(mdphi_dr/(r*r));
realTommy_drs[8] = factor*(realBarry_dr[8]*cosmdphi+imagBarry_dr[8]*sinmdphi)/(r*r*sin(th)) + factor*(realBarry[8]*cosmdphi+imagBarry[8]*sinmdphi)/(-r*r*r*sin(th)) + factor*(imagBarry[8]*cosmdphi-realBarry[8]*sinmdphi)*(mdphi_dr/(r*r*sin(th)));
imagTommy_drs[8] = factor*(imagBarry_dr[8]*cosmdphi-realBarry_dr[8]*sinmdphi)/(r*r*sin(th)) + factor*(imagBarry[8]*cosmdphi-realBarry[8]*sinmdphi)/(-r*r*r*sin(th)) + factor*(-realBarry[8]*cosmdphi-imagBarry[8]*sinmdphi)*(mdphi_dr/(r*r*sin(th)));
realTommy_drs[9] = factor*(realBarry_dr[9]*cosmdphi+imagBarry_dr[9]*sinmdphi)/(r*r*sin(th)*sin(th)) + factor*(realBarry[9]*cosmdphi+imagBarry[9]*sinmdphi)/(-r*r*r*sin(th)*sin(th)) + factor*(imagBarry[9]*cosmdphi-realBarry[9]*sinmdphi)*(mdphi_dr/(r*r*sin(th)*sin(th)));
imagTommy_drs[9] = factor*(imagBarry_dr[9]*cosmdphi-realBarry_dr[9]*sinmdphi)/(r*r*sin(th)*sin(th)) + factor*(imagBarry[9]*cosmdphi-realBarry[9]*sinmdphi)/(-r*r*r*sin(th)*sin(th)) + factor*(-realBarry[9]*cosmdphi-imagBarry[9]*sinmdphi)*(mdphi_dr/(r*r*sin(th)*sin(th)));
return 0;
}
