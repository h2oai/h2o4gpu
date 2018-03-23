#ifndef METRICS_METRICS_H_
#define METRICS_METRICS_H_


namespace h2o4gpu {

  double mcc(double tp, double tn, double fp, double fn);
  double f05(double tp, double tn, double fp, double fn);
  double f1(double tp, double tn, double fp, double fn);
  double f2(double tp, double tn, double fp, double fn);
  double acc(double tp, double tn, double fp, double fn);

  double mcc_opt(double *y, int n, double *yhat, int m);
  double mcc_opt(double *y, int n, double *yhat, int m, double* w, int l);

  double f05_opt(double *y, int n, double *yhat, int m);
  double f05_opt(double *y, int n, double *yhat, int m, double* w, int l);

  double f1_opt(double *y, int n, double *yhat, int m);
  double f1_opt(double *y, int n, double *yhat, int m, double* w, int l);

  double f2_opt(double *y, int n, double *yhat, int m);
  double f2_opt(double *y, int n, double *yhat, int m, double* w, int l);

  double acc_opt(double *y, int n, double *yhat, int m);
  double acc_opt(double *y, int n, double *yhat, int m, double* w, int l);

  void confusion_matrices(double *y, int n, double *yhat, int m, double *cm, int k, int j);
  void confusion_matrices(double *y, int n, double *yhat, int m, double* w, int l, double *cm, int k, int j);

}

#endif
