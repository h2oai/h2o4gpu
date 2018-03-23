#include <cmath>
#include <vector>
#include <iostream>
#include <stdlib.h>
#include <numeric>
#include <algorithm>
#include "metrics/metrics.h"

namespace h2o4gpu {

  template <typename T>
    std::vector<size_t> argsort(const std::vector<T> &v) {
    std::vector<size_t> idx(v.size());
    std::iota(idx.begin(), idx.end(), 0);
    sort(idx.begin(), idx.end(), [&v](size_t i1, size_t i2) {return v[i1] < v[i2]; });
    return idx;
  }

  typedef double(*CMMetricFunc)(double, double, double, double);

  #define CM_STATS_COLS 9

  double mcc(double tp, double tn, double fp, double fn) {
    auto n = tp + tn + fp + fn;
    auto s = (tp + fn) / n;
    auto p = (tp + fp) / n;
    auto x = (tp / n) - s * p;
    auto y = sqrt(p * s * (1 - s) * (1 - p));
    return (std::abs(y) < 1E-15) ?  0.0 : x / y;
  }

  double f05(double tp, double tn, double fp, double fn) {
    auto y = 1.25 * tp + fp + 0.25 * fn;
    return (std::abs(y) < 1E-15) ? 0.0 : (1.25 * tp) / y;
  }

  double f1(double tp, double tn, double fp, double fn) {
    auto y = 2 * tp + fp + fn;
    return (std::abs(y) < 1E-15) ? 0.0 : (2 * tp) / y;
  }

  double f2(double tp, double tn, double fp, double fn) {
    auto y = 5 * tp + fp + 4 * fn;
    return (std::abs(y) < 1E-15) ? 0.0 : (5 * tp) / y;
  }

  double acc(double tp, double tn, double fp, double fn) {
    auto y = tp + fp + tn + fn;
    return (std::abs(y) < 1E-15) ? 0.0 : (tp + tn) / y;
  }

  double cm_metric_opt(std::vector<double> y, std::vector<double> yhat,
                       std::vector<double> w, CMMetricFunc metric) {
    auto idx = argsort(yhat);
    int n = static_cast<int>(y.size());
    double tp = 0;
    double tn = 0;
    double fp = 0;
    double fn = 0;
    std::vector<double> y_sorted;
    std::vector<double> w_sorted;
    for (auto &i : idx) {
      y_sorted.push_back(y[i]);  
      w_sorted.push_back(w[i]);
      auto label = static_cast<int>(y[i]);
      tp += w[i] * label;
      fp += w[i] * (1 - label);
    }
    double best_score = 0;
    double prev_proba = -1;
    for (int i = 0; i < n; ++i) {
      auto proba = yhat[idx[i]];
      if (proba != prev_proba) {
        prev_proba = proba;
        best_score = std::max(best_score, metric(tp, tn, fp, fn));
      }
      if (static_cast<int>(y_sorted[i]) == 1) {
          tp -= w_sorted[i];
          fn += w_sorted[i];
      } else {
          tn += w_sorted[i];
          fp -= w_sorted[i];
      }
    }
    return best_score;
  }
  
  void cm_stats(std::vector<double> y, std::vector<double> yhat, std::vector<double> w,
                double cm[][CM_STATS_COLS]) {
    auto idx = argsort(yhat);
    int n = static_cast<int>(y.size());
    double tp = 0;
    double tn = 0;
    double fp = 0;
    double fn = 0;
    std::vector<double> y_sorted;
    std::vector<double> w_sorted;
    for (auto &i : idx) {
      y_sorted.push_back(y[i]);
      w_sorted.push_back(w[i]);
      auto label = static_cast<int>(y[i]);
      tp += w[i] * label;
      fp += w[i] * (1 - label);
    }
    double prev_proba = -1;
    int k = 0;
    for (int i = 0; i < n; ++i) {
      auto proba = yhat[idx[i]];
      if (proba != prev_proba) {
        prev_proba = proba;
        cm[k][0] = proba;
        cm[k][1] = tp;
        cm[k][2] = tn;
        cm[k][3] = fp;
        cm[k][4] = fn;
        cm[k][5] = fp / (fp + tn); // fpr
        cm[k][6] = tp / (tp + fn); // tpr
        cm[k][7] = mcc(tp, tn, fp, fn);
        cm[k][8] = f1(tp, tn, fp, fn);
        k += 1;
      }
      if (static_cast<int>(y_sorted[i]) == 1) {
        tp -= w_sorted[i];
        fn += w_sorted[i];
      }
      else {
        tn += w_sorted[i];
        fp -= w_sorted[i];
      }
    }
  }

  double mcc_opt(double *y, int n, double *yhat, int m) {
    std::vector<double> w(n, 1.0);
    return cm_metric_opt(std::vector<double>(y, y + n), std::vector<double>(yhat, yhat + m),
                         w, mcc);
  }
  
  double mcc_opt(double *y, int n, double *yhat, int m, double *w, int l) {
    return cm_metric_opt(std::vector<double>(y, y + n), std::vector<double>(yhat, yhat + m),
                         std::vector<double>(w, w + l), mcc);
  }

  double f05_opt(double *y, int n, double *yhat, int m) {
    std::vector<double> w(n, 1.0);
    return cm_metric_opt(std::vector<double>(y, y + n), std::vector<double>(yhat, yhat + m),
      w, f05);
  }

  double f05_opt(double *y, int n, double *yhat, int m, double *w, int l) {
    return cm_metric_opt(std::vector<double>(y, y + n), std::vector<double>(yhat, yhat + m),
      std::vector<double>(w, w + l), f05);
  }

  double f1_opt(double *y, int n, double *yhat, int m) {
    std::vector<double> w(n, 1.0);
    return cm_metric_opt(std::vector<double>(y, y + n), std::vector<double>(yhat, yhat + m),
      w, f1);
  }

  double f1_opt(double *y, int n, double *yhat, int m, double *w, int l) {
    return cm_metric_opt(std::vector<double>(y, y + n), std::vector<double>(yhat, yhat + m),
      std::vector<double>(w, w + l), f1);
  }

  double f2_opt(double *y, int n, double *yhat, int m) {
    std::vector<double> w(n, 1.0);
    return cm_metric_opt(std::vector<double>(y, y + n), std::vector<double>(yhat, yhat + m),
      w, f2);
  }

  double f2_opt(double *y, int n, double *yhat, int m, double *w, int l) {
    return cm_metric_opt(std::vector<double>(y, y + n), std::vector<double>(yhat, yhat + m),
      std::vector<double>(w, w + l), f2);
  }

  double acc_opt(double *y, int n, double *yhat, int m) {
    std::vector<double> w(n, 1.0);
    return cm_metric_opt(std::vector<double>(y, y + n), std::vector<double>(yhat, yhat + m),
      w, acc);
  }

  double acc_opt(double *y, int n, double *yhat, int m, double *w, int l) {
    return cm_metric_opt(std::vector<double>(y, y + n), std::vector<double>(yhat, yhat + m),
      std::vector<double>(w, w + l), acc);
  }
  
  void confusion_matrices(double *y, int n, double *yhat, int m, double *cm, int k, int j) {
    std::vector<double> w(n, 1.0);
    cm_stats(std::vector<double>(y, y + n), std::vector<double>(yhat, yhat + m),
             w, reinterpret_cast<double(*)[CM_STATS_COLS]>(cm));
  }

  void confusion_matrices(double *y, int n, double *yhat, int m, double* w, int l, double *cm, int k, int j) {
    cm_stats(std::vector<double>(y, y + n), std::vector<double>(yhat, yhat + m),
             std::vector<double>(w, w + l), reinterpret_cast<double(*)[CM_STATS_COLS]>(cm));
  }
}
