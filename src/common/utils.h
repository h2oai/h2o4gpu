/*!
 * Copyright 2017-2018 H2O.ai, Inc.
 * License   Apache License Version 2.0 (see LICENSE for details)
 */
#pragma once
#include <vector>
#include "cblas/cblas.h"

template<typename T>
void self_dot(std::vector<T> array_in, int n, int dim,
              std::vector<T>& dots);

void compute_distances(std::vector<double> data_in,
                       std::vector<double> centroids_in,
                       std::vector<double> &pairwise_distances,
                       int n, int dim, int k);

void compute_distances(std::vector<float> data_in,
                       std::vector<float> centroids_in,
                       std::vector<float> &pairwise_distances,
                       int n, int dim, int k);
