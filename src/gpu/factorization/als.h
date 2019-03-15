/*
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
/*
 * als.h
 *
 *  Created on: Aug 13, 2015
 *      Authoalsr: weitan
 */

#ifndef SRC_GPU_FACTORIZATION_ALS_H_
#define SRC_GPU_FACTORIZATION_ALS_H_

#define USE_CG
#define CG_ERROR 1e-4
// if cojugate gradient solver generates results in FP16
//#define CUMF_TT_FP16
//#define CUMF_XX_FP16
#define CG_ITER 6

#if GOOGLE_CUDA
#define EIGEN_USE_GPU
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#endif
#include "cuda_utils2.h"
#include "device_utilities.h"
#include "utils.h"
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cusparse.h>
#include <host_defines.h>
// these parameters do not change among different problem size
// our kernels handle the case where F%T==0 and F = 100
#define ALS_T10 10

#ifdef CUMF_USE_HALF
#define SCAN_BATCH 24
#else
#define SCAN_BATCH 28
#endif

#include <iostream>

using namespace std;

#define accumulate_in_registers()                           \
    do                                                      \
    {                                                       \
        temp0 += thetaTemp[tile_x / 2 + k * F / 2].x *      \
                 thetaTemp[tile_y / 2 + k * F / 2].x;       \
        temp1 += thetaTemp[tile_x / 2 + k * F / 2].x *      \
                 thetaTemp[tile_y / 2 + k * F / 2].y;       \
        temp2 += thetaTemp[tile_x / 2 + k * F / 2].x *      \
                 thetaTemp[tile_y / 2 + 1 + k * F / 2].x;   \
        temp3 += thetaTemp[tile_x / 2 + k * F / 2].x *      \
                 thetaTemp[tile_y / 2 + 1 + k * F / 2].y;   \
        temp4 += thetaTemp[tile_x / 2 + k * F / 2].x *      \
                 thetaTemp[tile_y / 2 + 2 + k * F / 2].x;   \
        temp5 += thetaTemp[tile_x / 2 + k * F / 2].x *      \
                 thetaTemp[tile_y / 2 + 2 + k * F / 2].y;   \
        temp6 += thetaTemp[tile_x / 2 + k * F / 2].x *      \
                 thetaTemp[tile_y / 2 + 3 + k * F / 2].x;   \
        temp7 += thetaTemp[tile_x / 2 + k * F / 2].x *      \
                 thetaTemp[tile_y / 2 + 3 + k * F / 2].y;   \
        temp8 += thetaTemp[tile_x / 2 + k * F / 2].x *      \
                 thetaTemp[tile_y / 2 + 4 + k * F / 2].x;   \
        temp9 += thetaTemp[tile_x / 2 + k * F / 2].x *      \
                 thetaTemp[tile_y / 2 + 4 + k * F / 2].y;   \
        temp10 += thetaTemp[tile_x / 2 + k * F / 2].y *     \
                  thetaTemp[tile_y / 2 + k * F / 2].x;      \
        temp11 += thetaTemp[tile_x / 2 + k * F / 2].y *     \
                  thetaTemp[tile_y / 2 + k * F / 2].y;      \
        temp12 += thetaTemp[tile_x / 2 + k * F / 2].y *     \
                  thetaTemp[tile_y / 2 + 1 + k * F / 2].x;  \
        temp13 += thetaTemp[tile_x / 2 + k * F / 2].y *     \
                  thetaTemp[tile_y / 2 + 1 + k * F / 2].y;  \
        temp14 += thetaTemp[tile_x / 2 + k * F / 2].y *     \
                  thetaTemp[tile_y / 2 + 2 + k * F / 2].x;  \
        temp15 += thetaTemp[tile_x / 2 + k * F / 2].y *     \
                  thetaTemp[tile_y / 2 + 2 + k * F / 2].y;  \
        temp16 += thetaTemp[tile_x / 2 + k * F / 2].y *     \
                  thetaTemp[tile_y / 2 + 3 + k * F / 2].x;  \
        temp17 += thetaTemp[tile_x / 2 + k * F / 2].y *     \
                  thetaTemp[tile_y / 2 + 3 + k * F / 2].y;  \
        temp18 += thetaTemp[tile_x / 2 + k * F / 2].y *     \
                  thetaTemp[tile_y / 2 + 4 + k * F / 2].x;  \
        temp19 += thetaTemp[tile_x / 2 + k * F / 2].y *     \
                  thetaTemp[tile_y / 2 + 4 + k * F / 2].y;  \
        temp20 += thetaTemp[tile_x / 2 + 1 + k * F / 2].x * \
                  thetaTemp[tile_y / 2 + k * F / 2].x;      \
        temp21 += thetaTemp[tile_x / 2 + 1 + k * F / 2].x * \
                  thetaTemp[tile_y / 2 + k * F / 2].y;      \
        temp22 += thetaTemp[tile_x / 2 + 1 + k * F / 2].x * \
                  thetaTemp[tile_y / 2 + 1 + k * F / 2].x;  \
        temp23 += thetaTemp[tile_x / 2 + 1 + k * F / 2].x * \
                  thetaTemp[tile_y / 2 + 1 + k * F / 2].y;  \
        temp24 += thetaTemp[tile_x / 2 + 1 + k * F / 2].x * \
                  thetaTemp[tile_y / 2 + 2 + k * F / 2].x;  \
        temp25 += thetaTemp[tile_x / 2 + 1 + k * F / 2].x * \
                  thetaTemp[tile_y / 2 + 2 + k * F / 2].y;  \
        temp26 += thetaTemp[tile_x / 2 + 1 + k * F / 2].x * \
                  thetaTemp[tile_y / 2 + 3 + k * F / 2].x;  \
        temp27 += thetaTemp[tile_x / 2 + 1 + k * F / 2].x * \
                  thetaTemp[tile_y / 2 + 3 + k * F / 2].y;  \
        temp28 += thetaTemp[tile_x / 2 + 1 + k * F / 2].x * \
                  thetaTemp[tile_y / 2 + 4 + k * F / 2].x;  \
        temp29 += thetaTemp[tile_x / 2 + 1 + k * F / 2].x * \
                  thetaTemp[tile_y / 2 + 4 + k * F / 2].y;  \
        temp30 += thetaTemp[tile_x / 2 + 1 + k * F / 2].y * \
                  thetaTemp[tile_y / 2 + k * F / 2].x;      \
        temp31 += thetaTemp[tile_x / 2 + 1 + k * F / 2].y * \
                  thetaTemp[tile_y / 2 + k * F / 2].y;      \
        temp32 += thetaTemp[tile_x / 2 + 1 + k * F / 2].y * \
                  thetaTemp[tile_y / 2 + 1 + k * F / 2].x;  \
        temp33 += thetaTemp[tile_x / 2 + 1 + k * F / 2].y * \
                  thetaTemp[tile_y / 2 + 1 + k * F / 2].y;  \
        temp34 += thetaTemp[tile_x / 2 + 1 + k * F / 2].y * \
                  thetaTemp[tile_y / 2 + 2 + k * F / 2].x;  \
        temp35 += thetaTemp[tile_x / 2 + 1 + k * F / 2].y * \
                  thetaTemp[tile_y / 2 + 2 + k * F / 2].y;  \
        temp36 += thetaTemp[tile_x / 2 + 1 + k * F / 2].y * \
                  thetaTemp[tile_y / 2 + 3 + k * F / 2].x;  \
        temp37 += thetaTemp[tile_x / 2 + 1 + k * F / 2].y * \
                  thetaTemp[tile_y / 2 + 3 + k * F / 2].y;  \
        temp38 += thetaTemp[tile_x / 2 + 1 + k * F / 2].y * \
                  thetaTemp[tile_y / 2 + 4 + k * F / 2].x;  \
        temp39 += thetaTemp[tile_x / 2 + 1 + k * F / 2].y * \
                  thetaTemp[tile_y / 2 + 4 + k * F / 2].y;  \
        temp40 += thetaTemp[tile_x / 2 + 2 + k * F / 2].x * \
                  thetaTemp[tile_y / 2 + k * F / 2].x;      \
        temp41 += thetaTemp[tile_x / 2 + 2 + k * F / 2].x * \
                  thetaTemp[tile_y / 2 + k * F / 2].y;      \
        temp42 += thetaTemp[tile_x / 2 + 2 + k * F / 2].x * \
                  thetaTemp[tile_y / 2 + 1 + k * F / 2].x;  \
        temp43 += thetaTemp[tile_x / 2 + 2 + k * F / 2].x * \
                  thetaTemp[tile_y / 2 + 1 + k * F / 2].y;  \
        temp44 += thetaTemp[tile_x / 2 + 2 + k * F / 2].x * \
                  thetaTemp[tile_y / 2 + 2 + k * F / 2].x;  \
        temp45 += thetaTemp[tile_x / 2 + 2 + k * F / 2].x * \
                  thetaTemp[tile_y / 2 + 2 + k * F / 2].y;  \
        temp46 += thetaTemp[tile_x / 2 + 2 + k * F / 2].x * \
                  thetaTemp[tile_y / 2 + 3 + k * F / 2].x;  \
        temp47 += thetaTemp[tile_x / 2 + 2 + k * F / 2].x * \
                  thetaTemp[tile_y / 2 + 3 + k * F / 2].y;  \
        temp48 += thetaTemp[tile_x / 2 + 2 + k * F / 2].x * \
                  thetaTemp[tile_y / 2 + 4 + k * F / 2].x;  \
        temp49 += thetaTemp[tile_x / 2 + 2 + k * F / 2].x * \
                  thetaTemp[tile_y / 2 + 4 + k * F / 2].y;  \
        temp50 += thetaTemp[tile_x / 2 + 2 + k * F / 2].y * \
                  thetaTemp[tile_y / 2 + k * F / 2].x;      \
        temp51 += thetaTemp[tile_x / 2 + 2 + k * F / 2].y * \
                  thetaTemp[tile_y / 2 + k * F / 2].y;      \
        temp52 += thetaTemp[tile_x / 2 + 2 + k * F / 2].y * \
                  thetaTemp[tile_y / 2 + 1 + k * F / 2].x;  \
        temp53 += thetaTemp[tile_x / 2 + 2 + k * F / 2].y * \
                  thetaTemp[tile_y / 2 + 1 + k * F / 2].y;  \
        temp54 += thetaTemp[tile_x / 2 + 2 + k * F / 2].y * \
                  thetaTemp[tile_y / 2 + 2 + k * F / 2].x;  \
        temp55 += thetaTemp[tile_x / 2 + 2 + k * F / 2].y * \
                  thetaTemp[tile_y / 2 + 2 + k * F / 2].y;  \
        temp56 += thetaTemp[tile_x / 2 + 2 + k * F / 2].y * \
                  thetaTemp[tile_y / 2 + 3 + k * F / 2].x;  \
        temp57 += thetaTemp[tile_x / 2 + 2 + k * F / 2].y * \
                  thetaTemp[tile_y / 2 + 3 + k * F / 2].y;  \
        temp58 += thetaTemp[tile_x / 2 + 2 + k * F / 2].y * \
                  thetaTemp[tile_y / 2 + 4 + k * F / 2].x;  \
        temp59 += thetaTemp[tile_x / 2 + 2 + k * F / 2].y * \
                  thetaTemp[tile_y / 2 + 4 + k * F / 2].y;  \
        temp60 += thetaTemp[tile_x / 2 + 3 + k * F / 2].x * \
                  thetaTemp[tile_y / 2 + k * F / 2].x;      \
        temp61 += thetaTemp[tile_x / 2 + 3 + k * F / 2].x * \
                  thetaTemp[tile_y / 2 + k * F / 2].y;      \
        temp62 += thetaTemp[tile_x / 2 + 3 + k * F / 2].x * \
                  thetaTemp[tile_y / 2 + 1 + k * F / 2].x;  \
        temp63 += thetaTemp[tile_x / 2 + 3 + k * F / 2].x * \
                  thetaTemp[tile_y / 2 + 1 + k * F / 2].y;  \
        temp64 += thetaTemp[tile_x / 2 + 3 + k * F / 2].x * \
                  thetaTemp[tile_y / 2 + 2 + k * F / 2].x;  \
        temp65 += thetaTemp[tile_x / 2 + 3 + k * F / 2].x * \
                  thetaTemp[tile_y / 2 + 2 + k * F / 2].y;  \
        temp66 += thetaTemp[tile_x / 2 + 3 + k * F / 2].x * \
                  thetaTemp[tile_y / 2 + 3 + k * F / 2].x;  \
        temp67 += thetaTemp[tile_x / 2 + 3 + k * F / 2].x * \
                  thetaTemp[tile_y / 2 + 3 + k * F / 2].y;  \
        temp68 += thetaTemp[tile_x / 2 + 3 + k * F / 2].x * \
                  thetaTemp[tile_y / 2 + 4 + k * F / 2].x;  \
        temp69 += thetaTemp[tile_x / 2 + 3 + k * F / 2].x * \
                  thetaTemp[tile_y / 2 + 4 + k * F / 2].y;  \
        temp70 += thetaTemp[tile_x / 2 + 3 + k * F / 2].y * \
                  thetaTemp[tile_y / 2 + k * F / 2].x;      \
        temp71 += thetaTemp[tile_x / 2 + 3 + k * F / 2].y * \
                  thetaTemp[tile_y / 2 + k * F / 2].y;      \
        temp72 += thetaTemp[tile_x / 2 + 3 + k * F / 2].y * \
                  thetaTemp[tile_y / 2 + 1 + k * F / 2].x;  \
        temp73 += thetaTemp[tile_x / 2 + 3 + k * F / 2].y * \
                  thetaTemp[tile_y / 2 + 1 + k * F / 2].y;  \
        temp74 += thetaTemp[tile_x / 2 + 3 + k * F / 2].y * \
                  thetaTemp[tile_y / 2 + 2 + k * F / 2].x;  \
        temp75 += thetaTemp[tile_x / 2 + 3 + k * F / 2].y * \
                  thetaTemp[tile_y / 2 + 2 + k * F / 2].y;  \
        temp76 += thetaTemp[tile_x / 2 + 3 + k * F / 2].y * \
                  thetaTemp[tile_y / 2 + 3 + k * F / 2].x;  \
        temp77 += thetaTemp[tile_x / 2 + 3 + k * F / 2].y * \
                  thetaTemp[tile_y / 2 + 3 + k * F / 2].y;  \
        temp78 += thetaTemp[tile_x / 2 + 3 + k * F / 2].y * \
                  thetaTemp[tile_y / 2 + 4 + k * F / 2].x;  \
        temp79 += thetaTemp[tile_x / 2 + 3 + k * F / 2].y * \
                  thetaTemp[tile_y / 2 + 4 + k * F / 2].y;  \
        temp80 += thetaTemp[tile_x / 2 + 4 + k * F / 2].x * \
                  thetaTemp[tile_y / 2 + k * F / 2].x;      \
        temp81 += thetaTemp[tile_x / 2 + 4 + k * F / 2].x * \
                  thetaTemp[tile_y / 2 + k * F / 2].y;      \
        temp82 += thetaTemp[tile_x / 2 + 4 + k * F / 2].x * \
                  thetaTemp[tile_y / 2 + 1 + k * F / 2].x;  \
        temp83 += thetaTemp[tile_x / 2 + 4 + k * F / 2].x * \
                  thetaTemp[tile_y / 2 + 1 + k * F / 2].y;  \
        temp84 += thetaTemp[tile_x / 2 + 4 + k * F / 2].x * \
                  thetaTemp[tile_y / 2 + 2 + k * F / 2].x;  \
        temp85 += thetaTemp[tile_x / 2 + 4 + k * F / 2].x * \
                  thetaTemp[tile_y / 2 + 2 + k * F / 2].y;  \
        temp86 += thetaTemp[tile_x / 2 + 4 + k * F / 2].x * \
                  thetaTemp[tile_y / 2 + 3 + k * F / 2].x;  \
        temp87 += thetaTemp[tile_x / 2 + 4 + k * F / 2].x * \
                  thetaTemp[tile_y / 2 + 3 + k * F / 2].y;  \
        temp88 += thetaTemp[tile_x / 2 + 4 + k * F / 2].x * \
                  thetaTemp[tile_y / 2 + 4 + k * F / 2].x;  \
        temp89 += thetaTemp[tile_x / 2 + 4 + k * F / 2].x * \
                  thetaTemp[tile_y / 2 + 4 + k * F / 2].y;  \
        temp90 += thetaTemp[tile_x / 2 + 4 + k * F / 2].y * \
                  thetaTemp[tile_y / 2 + k * F / 2].x;      \
        temp91 += thetaTemp[tile_x / 2 + 4 + k * F / 2].y * \
                  thetaTemp[tile_y / 2 + k * F / 2].y;      \
        temp92 += thetaTemp[tile_x / 2 + 4 + k * F / 2].y * \
                  thetaTemp[tile_y / 2 + 1 + k * F / 2].x;  \
        temp93 += thetaTemp[tile_x / 2 + 4 + k * F / 2].y * \
                  thetaTemp[tile_y / 2 + 1 + k * F / 2].y;  \
        temp94 += thetaTemp[tile_x / 2 + 4 + k * F / 2].y * \
                  thetaTemp[tile_y / 2 + 2 + k * F / 2].x;  \
        temp95 += thetaTemp[tile_x / 2 + 4 + k * F / 2].y * \
                  thetaTemp[tile_y / 2 + 2 + k * F / 2].y;  \
        temp96 += thetaTemp[tile_x / 2 + 4 + k * F / 2].y * \
                  thetaTemp[tile_y / 2 + 3 + k * F / 2].x;  \
        temp97 += thetaTemp[tile_x / 2 + 4 + k * F / 2].y * \
                  thetaTemp[tile_y / 2 + 3 + k * F / 2].y;  \
        temp98 += thetaTemp[tile_x / 2 + 4 + k * F / 2].y * \
                  thetaTemp[tile_y / 2 + 4 + k * F / 2].x;  \
        temp99 += thetaTemp[tile_x / 2 + 4 + k * F / 2].y * \
                  thetaTemp[tile_y / 2 + 4 + k * F / 2].y;  \
    } while (0)

#define fill_upper_half_from_registers()                    \
    do                                                      \
    {                                                       \
        tt[index + tile_x + tile_y * F] = temp0;            \
        tt[index + tile_x + (tile_y + 1) * F] = temp1;      \
        tt[index + tile_x + (tile_y + 2) * F] = temp2;      \
        tt[index + tile_x + (tile_y + 3) * F] = temp3;      \
        tt[index + tile_x + (tile_y + 4) * F] = temp4;      \
        tt[index + tile_x + (tile_y + 5) * F] = temp5;      \
        tt[index + tile_x + (tile_y + 6) * F] = temp6;      \
        tt[index + tile_x + (tile_y + 7) * F] = temp7;      \
        tt[index + tile_x + (tile_y + 8) * F] = temp8;      \
        tt[index + tile_x + (tile_y + 9) * F] = temp9;      \
                                                            \
        tt[index + tile_x + 1 + tile_y * F] = temp10;       \
        tt[index + tile_x + 1 + (tile_y + 1) * F] = temp11; \
        tt[index + tile_x + 1 + (tile_y + 2) * F] = temp12; \
        tt[index + tile_x + 1 + (tile_y + 3) * F] = temp13; \
        tt[index + tile_x + 1 + (tile_y + 4) * F] = temp14; \
        tt[index + tile_x + 1 + (tile_y + 5) * F] = temp15; \
        tt[index + tile_x + 1 + (tile_y + 6) * F] = temp16; \
        tt[index + tile_x + 1 + (tile_y + 7) * F] = temp17; \
        tt[index + tile_x + 1 + (tile_y + 8) * F] = temp18; \
        tt[index + tile_x + 1 + (tile_y + 9) * F] = temp19; \
                                                            \
        tt[index + tile_x + 2 + tile_y * F] = temp20;       \
        tt[index + tile_x + 2 + (tile_y + 1) * F] = temp21; \
        tt[index + tile_x + 2 + (tile_y + 2) * F] = temp22; \
        tt[index + tile_x + 2 + (tile_y + 3) * F] = temp23; \
        tt[index + tile_x + 2 + (tile_y + 4) * F] = temp24; \
        tt[index + tile_x + 2 + (tile_y + 5) * F] = temp25; \
        tt[index + tile_x + 2 + (tile_y + 6) * F] = temp26; \
        tt[index + tile_x + 2 + (tile_y + 7) * F] = temp27; \
        tt[index + tile_x + 2 + (tile_y + 8) * F] = temp28; \
        tt[index + tile_x + 2 + (tile_y + 9) * F] = temp29; \
                                                            \
        tt[index + tile_x + 3 + tile_y * F] = temp30;       \
        tt[index + tile_x + 3 + (tile_y + 1) * F] = temp31; \
        tt[index + tile_x + 3 + (tile_y + 2) * F] = temp32; \
        tt[index + tile_x + 3 + (tile_y + 3) * F] = temp33; \
        tt[index + tile_x + 3 + (tile_y + 4) * F] = temp34; \
        tt[index + tile_x + 3 + (tile_y + 5) * F] = temp35; \
        tt[index + tile_x + 3 + (tile_y + 6) * F] = temp36; \
        tt[index + tile_x + 3 + (tile_y + 7) * F] = temp37; \
        tt[index + tile_x + 3 + (tile_y + 8) * F] = temp38; \
        tt[index + tile_x + 3 + (tile_y + 9) * F] = temp39; \
                                                            \
        tt[index + tile_x + 4 + tile_y * F] = temp40;       \
        tt[index + tile_x + 4 + (tile_y + 1) * F] = temp41; \
        tt[index + tile_x + 4 + (tile_y + 2) * F] = temp42; \
        tt[index + tile_x + 4 + (tile_y + 3) * F] = temp43; \
        tt[index + tile_x + 4 + (tile_y + 4) * F] = temp44; \
        tt[index + tile_x + 4 + (tile_y + 5) * F] = temp45; \
        tt[index + tile_x + 4 + (tile_y + 6) * F] = temp46; \
        tt[index + tile_x + 4 + (tile_y + 7) * F] = temp47; \
        tt[index + tile_x + 4 + (tile_y + 8) * F] = temp48; \
        tt[index + tile_x + 4 + (tile_y + 9) * F] = temp49; \
                                                            \
        tt[index + tile_x + 5 + tile_y * F] = temp50;       \
        tt[index + tile_x + 5 + (tile_y + 1) * F] = temp51; \
        tt[index + tile_x + 5 + (tile_y + 2) * F] = temp52; \
        tt[index + tile_x + 5 + (tile_y + 3) * F] = temp53; \
        tt[index + tile_x + 5 + (tile_y + 4) * F] = temp54; \
        tt[index + tile_x + 5 + (tile_y + 5) * F] = temp55; \
        tt[index + tile_x + 5 + (tile_y + 6) * F] = temp56; \
        tt[index + tile_x + 5 + (tile_y + 7) * F] = temp57; \
        tt[index + tile_x + 5 + (tile_y + 8) * F] = temp58; \
        tt[index + tile_x + 5 + (tile_y + 9) * F] = temp59; \
                                                            \
        tt[index + tile_x + 6 + tile_y * F] = temp60;       \
        tt[index + tile_x + 6 + (tile_y + 1) * F] = temp61; \
        tt[index + tile_x + 6 + (tile_y + 2) * F] = temp62; \
        tt[index + tile_x + 6 + (tile_y + 3) * F] = temp63; \
        tt[index + tile_x + 6 + (tile_y + 4) * F] = temp64; \
        tt[index + tile_x + 6 + (tile_y + 5) * F] = temp65; \
        tt[index + tile_x + 6 + (tile_y + 6) * F] = temp66; \
        tt[index + tile_x + 6 + (tile_y + 7) * F] = temp67; \
        tt[index + tile_x + 6 + (tile_y + 8) * F] = temp68; \
        tt[index + tile_x + 6 + (tile_y + 9) * F] = temp69; \
                                                            \
        tt[index + tile_x + 7 + tile_y * F] = temp70;       \
        tt[index + tile_x + 7 + (tile_y + 1) * F] = temp71; \
        tt[index + tile_x + 7 + (tile_y + 2) * F] = temp72; \
        tt[index + tile_x + 7 + (tile_y + 3) * F] = temp73; \
        tt[index + tile_x + 7 + (tile_y + 4) * F] = temp74; \
        tt[index + tile_x + 7 + (tile_y + 5) * F] = temp75; \
        tt[index + tile_x + 7 + (tile_y + 6) * F] = temp76; \
        tt[index + tile_x + 7 + (tile_y + 7) * F] = temp77; \
        tt[index + tile_x + 7 + (tile_y + 8) * F] = temp78; \
        tt[index + tile_x + 7 + (tile_y + 9) * F] = temp79; \
                                                            \
        tt[index + tile_x + 8 + tile_y * F] = temp80;       \
        tt[index + tile_x + 8 + (tile_y + 1) * F] = temp81; \
        tt[index + tile_x + 8 + (tile_y + 2) * F] = temp82; \
        tt[index + tile_x + 8 + (tile_y + 3) * F] = temp83; \
        tt[index + tile_x + 8 + (tile_y + 4) * F] = temp84; \
        tt[index + tile_x + 8 + (tile_y + 5) * F] = temp85; \
        tt[index + tile_x + 8 + (tile_y + 6) * F] = temp86; \
        tt[index + tile_x + 8 + (tile_y + 7) * F] = temp87; \
        tt[index + tile_x + 8 + (tile_y + 8) * F] = temp88; \
        tt[index + tile_x + 8 + (tile_y + 9) * F] = temp89; \
                                                            \
        tt[index + tile_x + 9 + tile_y * F] = temp90;       \
        tt[index + tile_x + 9 + (tile_y + 1) * F] = temp91; \
        tt[index + tile_x + 9 + (tile_y + 2) * F] = temp92; \
        tt[index + tile_x + 9 + (tile_y + 3) * F] = temp93; \
        tt[index + tile_x + 9 + (tile_y + 4) * F] = temp94; \
        tt[index + tile_x + 9 + (tile_y + 5) * F] = temp95; \
        tt[index + tile_x + 9 + (tile_y + 6) * F] = temp96; \
        tt[index + tile_x + 9 + (tile_y + 7) * F] = temp97; \
        tt[index + tile_x + 9 + (tile_y + 8) * F] = temp98; \
        tt[index + tile_x + 9 + (tile_y + 9) * F] = temp99; \
    } while (0)

#define fill_lower_half_from_registers()                    \
    do                                                      \
    {                                                       \
        tt[index + tile_y + 0 + (tile_x + 0) * F] = temp0;  \
        tt[index + tile_y + 1 + (tile_x + 0) * F] = temp1;  \
        tt[index + tile_y + 2 + (tile_x + 0) * F] = temp2;  \
        tt[index + tile_y + 3 + (tile_x + 0) * F] = temp3;  \
        tt[index + tile_y + 4 + (tile_x + 0) * F] = temp4;  \
        tt[index + tile_y + 5 + (tile_x + 0) * F] = temp5;  \
        tt[index + tile_y + 6 + (tile_x + 0) * F] = temp6;  \
        tt[index + tile_y + 7 + (tile_x + 0) * F] = temp7;  \
        tt[index + tile_y + 8 + (tile_x + 0) * F] = temp8;  \
        tt[index + tile_y + 9 + (tile_x + 0) * F] = temp9;  \
                                                            \
        tt[index + tile_y + 0 + (tile_x + 1) * F] = temp10; \
        tt[index + tile_y + 1 + (tile_x + 1) * F] = temp11; \
        tt[index + tile_y + 2 + (tile_x + 1) * F] = temp12; \
        tt[index + tile_y + 3 + (tile_x + 1) * F] = temp13; \
        tt[index + tile_y + 4 + (tile_x + 1) * F] = temp14; \
        tt[index + tile_y + 5 + (tile_x + 1) * F] = temp15; \
        tt[index + tile_y + 6 + (tile_x + 1) * F] = temp16; \
        tt[index + tile_y + 7 + (tile_x + 1) * F] = temp17; \
        tt[index + tile_y + 8 + (tile_x + 1) * F] = temp18; \
        tt[index + tile_y + 9 + (tile_x + 1) * F] = temp19; \
                                                            \
        tt[index + tile_y + 0 + (tile_x + 2) * F] = temp20; \
        tt[index + tile_y + 1 + (tile_x + 2) * F] = temp21; \
        tt[index + tile_y + 2 + (tile_x + 2) * F] = temp22; \
        tt[index + tile_y + 3 + (tile_x + 2) * F] = temp23; \
        tt[index + tile_y + 4 + (tile_x + 2) * F] = temp24; \
        tt[index + tile_y + 5 + (tile_x + 2) * F] = temp25; \
        tt[index + tile_y + 6 + (tile_x + 2) * F] = temp26; \
        tt[index + tile_y + 7 + (tile_x + 2) * F] = temp27; \
        tt[index + tile_y + 8 + (tile_x + 2) * F] = temp28; \
        tt[index + tile_y + 9 + (tile_x + 2) * F] = temp29; \
                                                            \
        tt[index + tile_y + 0 + (tile_x + 3) * F] = temp30; \
        tt[index + tile_y + 1 + (tile_x + 3) * F] = temp31; \
        tt[index + tile_y + 2 + (tile_x + 3) * F] = temp32; \
        tt[index + tile_y + 3 + (tile_x + 3) * F] = temp33; \
        tt[index + tile_y + 4 + (tile_x + 3) * F] = temp34; \
        tt[index + tile_y + 5 + (tile_x + 3) * F] = temp35; \
        tt[index + tile_y + 6 + (tile_x + 3) * F] = temp36; \
        tt[index + tile_y + 7 + (tile_x + 3) * F] = temp37; \
        tt[index + tile_y + 8 + (tile_x + 3) * F] = temp38; \
        tt[index + tile_y + 9 + (tile_x + 3) * F] = temp39; \
                                                            \
        tt[index + tile_y + 0 + (tile_x + 4) * F] = temp40; \
        tt[index + tile_y + 1 + (tile_x + 4) * F] = temp41; \
        tt[index + tile_y + 2 + (tile_x + 4) * F] = temp42; \
        tt[index + tile_y + 3 + (tile_x + 4) * F] = temp43; \
        tt[index + tile_y + 4 + (tile_x + 4) * F] = temp44; \
        tt[index + tile_y + 5 + (tile_x + 4) * F] = temp45; \
        tt[index + tile_y + 6 + (tile_x + 4) * F] = temp46; \
        tt[index + tile_y + 7 + (tile_x + 4) * F] = temp47; \
        tt[index + tile_y + 8 + (tile_x + 4) * F] = temp48; \
        tt[index + tile_y + 9 + (tile_x + 4) * F] = temp49; \
                                                            \
        tt[index + tile_y + 0 + (tile_x + 5) * F] = temp50; \
        tt[index + tile_y + 1 + (tile_x + 5) * F] = temp51; \
        tt[index + tile_y + 2 + (tile_x + 5) * F] = temp52; \
        tt[index + tile_y + 3 + (tile_x + 5) * F] = temp53; \
        tt[index + tile_y + 4 + (tile_x + 5) * F] = temp54; \
        tt[index + tile_y + 5 + (tile_x + 5) * F] = temp55; \
        tt[index + tile_y + 6 + (tile_x + 5) * F] = temp56; \
        tt[index + tile_y + 7 + (tile_x + 5) * F] = temp57; \
        tt[index + tile_y + 8 + (tile_x + 5) * F] = temp58; \
        tt[index + tile_y + 9 + (tile_x + 5) * F] = temp59; \
                                                            \
        tt[index + tile_y + 0 + (tile_x + 6) * F] = temp60; \
        tt[index + tile_y + 1 + (tile_x + 6) * F] = temp61; \
        tt[index + tile_y + 2 + (tile_x + 6) * F] = temp62; \
        tt[index + tile_y + 3 + (tile_x + 6) * F] = temp63; \
        tt[index + tile_y + 4 + (tile_x + 6) * F] = temp64; \
        tt[index + tile_y + 5 + (tile_x + 6) * F] = temp65; \
        tt[index + tile_y + 6 + (tile_x + 6) * F] = temp66; \
        tt[index + tile_y + 7 + (tile_x + 6) * F] = temp67; \
        tt[index + tile_y + 8 + (tile_x + 6) * F] = temp68; \
        tt[index + tile_y + 9 + (tile_x + 6) * F] = temp69; \
                                                            \
        tt[index + tile_y + 0 + (tile_x + 7) * F] = temp70; \
        tt[index + tile_y + 1 + (tile_x + 7) * F] = temp71; \
        tt[index + tile_y + 2 + (tile_x + 7) * F] = temp72; \
        tt[index + tile_y + 3 + (tile_x + 7) * F] = temp73; \
        tt[index + tile_y + 4 + (tile_x + 7) * F] = temp74; \
        tt[index + tile_y + 5 + (tile_x + 7) * F] = temp75; \
        tt[index + tile_y + 6 + (tile_x + 7) * F] = temp76; \
        tt[index + tile_y + 7 + (tile_x + 7) * F] = temp77; \
        tt[index + tile_y + 8 + (tile_x + 7) * F] = temp78; \
        tt[index + tile_y + 9 + (tile_x + 7) * F] = temp79; \
                                                            \
        tt[index + tile_y + 0 + (tile_x + 8) * F] = temp80; \
        tt[index + tile_y + 1 + (tile_x + 8) * F] = temp81; \
        tt[index + tile_y + 2 + (tile_x + 8) * F] = temp82; \
        tt[index + tile_y + 3 + (tile_x + 8) * F] = temp83; \
        tt[index + tile_y + 4 + (tile_x + 8) * F] = temp84; \
        tt[index + tile_y + 5 + (tile_x + 8) * F] = temp85; \
        tt[index + tile_y + 6 + (tile_x + 8) * F] = temp86; \
        tt[index + tile_y + 7 + (tile_x + 8) * F] = temp87; \
        tt[index + tile_y + 8 + (tile_x + 8) * F] = temp88; \
        tt[index + tile_y + 9 + (tile_x + 8) * F] = temp89; \
                                                            \
        tt[index + tile_y + 0 + (tile_x + 9) * F] = temp90; \
        tt[index + tile_y + 1 + (tile_x + 9) * F] = temp91; \
        tt[index + tile_y + 2 + (tile_x + 9) * F] = temp92; \
        tt[index + tile_y + 3 + (tile_x + 9) * F] = temp93; \
        tt[index + tile_y + 4 + (tile_x + 9) * F] = temp94; \
        tt[index + tile_y + 5 + (tile_x + 9) * F] = temp95; \
        tt[index + tile_y + 6 + (tile_x + 9) * F] = temp96; \
        tt[index + tile_y + 7 + (tile_x + 9) * F] = temp97; \
        tt[index + tile_y + 8 + (tile_x + 9) * F] = temp98; \
        tt[index + tile_y + 9 + (tile_x + 9) * F] = temp99; \
    } while (0)

#define fill_lower_half_from_registers_fp16()               \
    do                                                      \
    {                                                       \
        tt[index + tile_y / 2 + 0 + (tile_x + 0) * F / 2] = \
            __floats2half2_rn(temp0, temp1);                \
        tt[index + tile_y / 2 + 1 + (tile_x + 0) * F / 2] = \
            __floats2half2_rn(temp2, temp3);                \
        tt[index + tile_y / 2 + 2 + (tile_x + 0) * F / 2] = \
            __floats2half2_rn(temp4, temp5);                \
        tt[index + tile_y / 2 + 3 + (tile_x + 0) * F / 2] = \
            __floats2half2_rn(temp6, temp7);                \
        tt[index + tile_y / 2 + 4 + (tile_x + 0) * F / 2] = \
            __floats2half2_rn(temp8, temp9);                \
                                                            \
        tt[index + tile_y / 2 + 0 + (tile_x + 1) * F / 2] = \
            __floats2half2_rn(temp10, temp11);              \
        tt[index + tile_y / 2 + 1 + (tile_x + 1) * F / 2] = \
            __floats2half2_rn(temp12, temp13);              \
        tt[index + tile_y / 2 + 2 + (tile_x + 1) * F / 2] = \
            __floats2half2_rn(temp14, temp15);              \
        tt[index + tile_y / 2 + 3 + (tile_x + 1) * F / 2] = \
            __floats2half2_rn(temp16, temp17);              \
        tt[index + tile_y / 2 + 4 + (tile_x + 1) * F / 2] = \
            __floats2half2_rn(temp18, temp19);              \
                                                            \
        tt[index + tile_y / 2 + 0 + (tile_x + 2) * F / 2] = \
            __floats2half2_rn(temp20, temp21);              \
        tt[index + tile_y / 2 + 1 + (tile_x + 2) * F / 2] = \
            __floats2half2_rn(temp22, temp23);              \
        tt[index + tile_y / 2 + 2 + (tile_x + 2) * F / 2] = \
            __floats2half2_rn(temp24, temp25);              \
        tt[index + tile_y / 2 + 3 + (tile_x + 2) * F / 2] = \
            __floats2half2_rn(temp26, temp27);              \
        tt[index + tile_y / 2 + 4 + (tile_x + 2) * F / 2] = \
            __floats2half2_rn(temp28, temp29);              \
                                                            \
        tt[index + tile_y / 2 + 0 + (tile_x + 3) * F / 2] = \
            __floats2half2_rn(temp30, temp31);              \
        tt[index + tile_y / 2 + 1 + (tile_x + 3) * F / 2] = \
            __floats2half2_rn(temp32, temp33);              \
        tt[index + tile_y / 2 + 2 + (tile_x + 3) * F / 2] = \
            __floats2half2_rn(temp34, temp35);              \
        tt[index + tile_y / 2 + 3 + (tile_x + 3) * F / 2] = \
            __floats2half2_rn(temp36, temp37);              \
        tt[index + tile_y / 2 + 4 + (tile_x + 3) * F / 2] = \
            __floats2half2_rn(temp38, temp39);              \
                                                            \
        tt[index + tile_y / 2 + 0 + (tile_x + 4) * F / 2] = \
            __floats2half2_rn(temp40, temp41);              \
        tt[index + tile_y / 2 + 1 + (tile_x + 4) * F / 2] = \
            __floats2half2_rn(temp42, temp43);              \
        tt[index + tile_y / 2 + 2 + (tile_x + 4) * F / 2] = \
            __floats2half2_rn(temp44, temp45);              \
        tt[index + tile_y / 2 + 3 + (tile_x + 4) * F / 2] = \
            __floats2half2_rn(temp46, temp47);              \
        tt[index + tile_y / 2 + 4 + (tile_x + 4) * F / 2] = \
            __floats2half2_rn(temp48, temp49);              \
                                                            \
        tt[index + tile_y / 2 + 0 + (tile_x + 5) * F / 2] = \
            __floats2half2_rn(temp50, temp51);              \
        tt[index + tile_y / 2 + 1 + (tile_x + 5) * F / 2] = \
            __floats2half2_rn(temp52, temp53);              \
        tt[index + tile_y / 2 + 2 + (tile_x + 5) * F / 2] = \
            __floats2half2_rn(temp54, temp55);              \
        tt[index + tile_y / 2 + 3 + (tile_x + 5) * F / 2] = \
            __floats2half2_rn(temp56, temp57);              \
        tt[index + tile_y / 2 + 4 + (tile_x + 5) * F / 2] = \
            __floats2half2_rn(temp58, temp59);              \
                                                            \
        tt[index + tile_y / 2 + 0 + (tile_x + 6) * F / 2] = \
            __floats2half2_rn(temp60, temp61);              \
        tt[index + tile_y / 2 + 1 + (tile_x + 6) * F / 2] = \
            __floats2half2_rn(temp62, temp63);              \
        tt[index + tile_y / 2 + 2 + (tile_x + 6) * F / 2] = \
            __floats2half2_rn(temp64, temp65);              \
        tt[index + tile_y / 2 + 3 + (tile_x + 6) * F / 2] = \
            __floats2half2_rn(temp66, temp67);              \
        tt[index + tile_y / 2 + 4 + (tile_x + 6) * F / 2] = \
            __floats2half2_rn(temp68, temp69);              \
                                                            \
        tt[index + tile_y / 2 + 0 + (tile_x + 7) * F / 2] = \
            __floats2half2_rn(temp70, temp71);              \
        tt[index + tile_y / 2 + 1 + (tile_x + 7) * F / 2] = \
            __floats2half2_rn(temp72, temp73);              \
        tt[index + tile_y / 2 + 2 + (tile_x + 7) * F / 2] = \
            __floats2half2_rn(temp74, temp75);              \
        tt[index + tile_y / 2 + 3 + (tile_x + 7) * F / 2] = \
            __floats2half2_rn(temp76, temp77);              \
        tt[index + tile_y / 2 + 4 + (tile_x + 7) * F / 2] = \
            __floats2half2_rn(temp78, temp79);              \
                                                            \
        tt[index + tile_y / 2 + 0 + (tile_x + 8) * F / 2] = \
            __floats2half2_rn(temp80, temp81);              \
        tt[index + tile_y / 2 + 1 + (tile_x + 8) * F / 2] = \
            __floats2half2_rn(temp82, temp83);              \
        tt[index + tile_y / 2 + 2 + (tile_x + 8) * F / 2] = \
            __floats2half2_rn(temp84, temp85);              \
        tt[index + tile_y / 2 + 3 + (tile_x + 8) * F / 2] = \
            __floats2half2_rn(temp86, temp87);              \
        tt[index + tile_y / 2 + 4 + (tile_x + 8) * F / 2] = \
            __floats2half2_rn(temp88, temp89);              \
                                                            \
        tt[index + tile_y / 2 + 0 + (tile_x + 9) * F / 2] = \
            __floats2half2_rn(temp90, temp91);              \
        tt[index + tile_y / 2 + 1 + (tile_x + 9) * F / 2] = \
            __floats2half2_rn(temp92, temp93);              \
        tt[index + tile_y / 2 + 2 + (tile_x + 9) * F / 2] = \
            __floats2half2_rn(temp94, temp95);              \
        tt[index + tile_y / 2 + 3 + (tile_x + 9) * F / 2] = \
            __floats2half2_rn(temp96, temp97);              \
        tt[index + tile_y / 2 + 4 + (tile_x + 9) * F / 2] = \
            __floats2half2_rn(temp98, temp99);              \
    } while (0)

#define fill_upper_half_from_registers_fp16()               \
    do                                                      \
    {                                                       \
        tt[index + tile_x / 2 + 0 + (tile_y + 0) * F / 2] = \
            __floats2half2_rn(temp0, temp10);               \
        tt[index + tile_x / 2 + 1 + (tile_y + 0) * F / 2] = \
            __floats2half2_rn(temp20, temp30);              \
        tt[index + tile_x / 2 + 2 + (tile_y + 0) * F / 2] = \
            __floats2half2_rn(temp40, temp50);              \
        tt[index + tile_x / 2 + 3 + (tile_y + 0) * F / 2] = \
            __floats2half2_rn(temp60, temp70);              \
        tt[index + tile_x / 2 + 4 + (tile_y + 0) * F / 2] = \
            __floats2half2_rn(temp80, temp90);              \
                                                            \
        tt[index + tile_x / 2 + 0 + (tile_y + 1) * F / 2] = \
            __floats2half2_rn(temp1, temp11);               \
        tt[index + tile_x / 2 + 1 + (tile_y + 1) * F / 2] = \
            __floats2half2_rn(temp21, temp31);              \
        tt[index + tile_x / 2 + 2 + (tile_y + 1) * F / 2] = \
            __floats2half2_rn(temp41, temp51);              \
        tt[index + tile_x / 2 + 3 + (tile_y + 1) * F / 2] = \
            __floats2half2_rn(temp61, temp71);              \
        tt[index + tile_x / 2 + 4 + (tile_y + 1) * F / 2] = \
            __floats2half2_rn(temp81, temp91);              \
                                                            \
        tt[index + tile_x / 2 + 0 + (tile_y + 2) * F / 2] = \
            __floats2half2_rn(temp2, temp12);               \
        tt[index + tile_x / 2 + 1 + (tile_y + 2) * F / 2] = \
            __floats2half2_rn(temp22, temp32);              \
        tt[index + tile_x / 2 + 2 + (tile_y + 2) * F / 2] = \
            __floats2half2_rn(temp42, temp52);              \
        tt[index + tile_x / 2 + 3 + (tile_y + 2) * F / 2] = \
            __floats2half2_rn(temp62, temp72);              \
        tt[index + tile_x / 2 + 4 + (tile_y + 2) * F / 2] = \
            __floats2half2_rn(temp82, temp92);              \
                                                            \
        tt[index + tile_x / 2 + 0 + (tile_y + 3) * F / 2] = \
            __floats2half2_rn(temp3, temp13);               \
        tt[index + tile_x / 2 + 1 + (tile_y + 3) * F / 2] = \
            __floats2half2_rn(temp23, temp33);              \
        tt[index + tile_x / 2 + 2 + (tile_y + 3) * F / 2] = \
            __floats2half2_rn(temp43, temp53);              \
        tt[index + tile_x / 2 + 3 + (tile_y + 3) * F / 2] = \
            __floats2half2_rn(temp63, temp73);              \
        tt[index + tile_x / 2 + 4 + (tile_y + 3) * F / 2] = \
            __floats2half2_rn(temp83, temp93);              \
                                                            \
        tt[index + tile_x / 2 + 0 + (tile_y + 4) * F / 2] = \
            __floats2half2_rn(temp4, temp14);               \
        tt[index + tile_x / 2 + 1 + (tile_y + 4) * F / 2] = \
            __floats2half2_rn(temp24, temp34);              \
        tt[index + tile_x / 2 + 2 + (tile_y + 4) * F / 2] = \
            __floats2half2_rn(temp44, temp54);              \
        tt[index + tile_x / 2 + 3 + (tile_y + 4) * F / 2] = \
            __floats2half2_rn(temp64, temp74);              \
        tt[index + tile_x / 2 + 4 + (tile_y + 4) * F / 2] = \
            __floats2half2_rn(temp84, temp94);              \
                                                            \
        tt[index + tile_x / 2 + 0 + (tile_y + 5) * F / 2] = \
            __floats2half2_rn(temp5, temp15);               \
        tt[index + tile_x / 2 + 1 + (tile_y + 5) * F / 2] = \
            __floats2half2_rn(temp25, temp35);              \
        tt[index + tile_x / 2 + 2 + (tile_y + 5) * F / 2] = \
            __floats2half2_rn(temp45, temp55);              \
        tt[index + tile_x / 2 + 3 + (tile_y + 5) * F / 2] = \
            __floats2half2_rn(temp65, temp75);              \
        tt[index + tile_x / 2 + 4 + (tile_y + 5) * F / 2] = \
            __floats2half2_rn(temp85, temp95);              \
                                                            \
        tt[index + tile_x / 2 + 0 + (tile_y + 6) * F / 2] = \
            __floats2half2_rn(temp6, temp16);               \
        tt[index + tile_x / 2 + 1 + (tile_y + 6) * F / 2] = \
            __floats2half2_rn(temp26, temp36);              \
        tt[index + tile_x / 2 + 2 + (tile_y + 6) * F / 2] = \
            __floats2half2_rn(temp46, temp56);              \
        tt[index + tile_x / 2 + 3 + (tile_y + 6) * F / 2] = \
            __floats2half2_rn(temp66, temp76);              \
        tt[index + tile_x / 2 + 4 + (tile_y + 6) * F / 2] = \
            __floats2half2_rn(temp86, temp96);              \
                                                            \
        tt[index + tile_x / 2 + 0 + (tile_y + 7) * F / 2] = \
            __floats2half2_rn(temp7, temp17);               \
        tt[index + tile_x / 2 + 1 + (tile_y + 7) * F / 2] = \
            __floats2half2_rn(temp27, temp37);              \
        tt[index + tile_x / 2 + 2 + (tile_y + 7) * F / 2] = \
            __floats2half2_rn(temp47, temp57);              \
        tt[index + tile_x / 2 + 3 + (tile_y + 7) * F / 2] = \
            __floats2half2_rn(temp67, temp77);              \
        tt[index + tile_x / 2 + 4 + (tile_y + 7) * F / 2] = \
            __floats2half2_rn(temp87, temp97);              \
                                                            \
        tt[index + tile_x / 2 + 0 + (tile_y + 8) * F / 2] = \
            __floats2half2_rn(temp8, temp18);               \
        tt[index + tile_x / 2 + 1 + (tile_y + 8) * F / 2] = \
            __floats2half2_rn(temp28, temp38);              \
        tt[index + tile_x / 2 + 2 + (tile_y + 8) * F / 2] = \
            __floats2half2_rn(temp48, temp58);              \
        tt[index + tile_x / 2 + 3 + (tile_y + 8) * F / 2] = \
            __floats2half2_rn(temp68, temp78);              \
        tt[index + tile_x / 2 + 4 + (tile_y + 8) * F / 2] = \
            __floats2half2_rn(temp88, temp98);              \
                                                            \
        tt[index + tile_x / 2 + 0 + (tile_y + 9) * F / 2] = \
            __floats2half2_rn(temp9, temp19);               \
        tt[index + tile_x / 2 + 1 + (tile_y + 9) * F / 2] = \
            __floats2half2_rn(temp29, temp39);              \
        tt[index + tile_x / 2 + 2 + (tile_y + 9) * F / 2] = \
            __floats2half2_rn(temp49, temp59);              \
        tt[index + tile_x / 2 + 3 + (tile_y + 9) * F / 2] = \
            __floats2half2_rn(temp69, temp79);              \
        tt[index + tile_x / 2 + 4 + (tile_y + 9) * F / 2] = \
            __floats2half2_rn(temp89, temp99);              \
    } while (0)

#define fill_upper_half_from_registers_float2()             \
    do                                                      \
    {                                                       \
        tt[index + tile_x / 2 + 0 + (tile_y + 0) * F / 2] = \
            make_float2(temp0, temp10);                     \
        tt[index + tile_x / 2 + 1 + (tile_y + 0) * F / 2] = \
            make_float2(temp20, temp30);                    \
        tt[index + tile_x / 2 + 2 + (tile_y + 0) * F / 2] = \
            make_float2(temp40, temp50);                    \
        tt[index + tile_x / 2 + 3 + (tile_y + 0) * F / 2] = \
            make_float2(temp60, temp70);                    \
        tt[index + tile_x / 2 + 4 + (tile_y + 0) * F / 2] = \
            make_float2(temp80, temp90);                    \
                                                            \
        tt[index + tile_x / 2 + 0 + (tile_y + 1) * F / 2] = \
            make_float2(temp1, temp11);                     \
        tt[index + tile_x / 2 + 1 + (tile_y + 1) * F / 2] = \
            make_float2(temp21, temp31);                    \
        tt[index + tile_x / 2 + 2 + (tile_y + 1) * F / 2] = \
            make_float2(temp41, temp51);                    \
        tt[index + tile_x / 2 + 3 + (tile_y + 1) * F / 2] = \
            make_float2(temp61, temp71);                    \
        tt[index + tile_x / 2 + 4 + (tile_y + 1) * F / 2] = \
            make_float2(temp81, temp91);                    \
                                                            \
        tt[index + tile_x / 2 + 0 + (tile_y + 2) * F / 2] = \
            make_float2(temp2, temp12);                     \
        tt[index + tile_x / 2 + 1 + (tile_y + 2) * F / 2] = \
            make_float2(temp22, temp32);                    \
        tt[index + tile_x / 2 + 2 + (tile_y + 2) * F / 2] = \
            make_float2(temp42, temp52);                    \
        tt[index + tile_x / 2 + 3 + (tile_y + 2) * F / 2] = \
            make_float2(temp62, temp72);                    \
        tt[index + tile_x / 2 + 4 + (tile_y + 2) * F / 2] = \
            make_float2(temp82, temp92);                    \
                                                            \
        tt[index + tile_x / 2 + 0 + (tile_y + 3) * F / 2] = \
            make_float2(temp3, temp13);                     \
        tt[index + tile_x / 2 + 1 + (tile_y + 3) * F / 2] = \
            make_float2(temp23, temp33);                    \
        tt[index + tile_x / 2 + 2 + (tile_y + 3) * F / 2] = \
            make_float2(temp43, temp53);                    \
        tt[index + tile_x / 2 + 3 + (tile_y + 3) * F / 2] = \
            make_float2(temp63, temp73);                    \
        tt[index + tile_x / 2 + 4 + (tile_y + 3) * F / 2] = \
            make_float2(temp83, temp93);                    \
                                                            \
        tt[index + tile_x / 2 + 0 + (tile_y + 4) * F / 2] = \
            make_float2(temp4, temp14);                     \
        tt[index + tile_x / 2 + 1 + (tile_y + 4) * F / 2] = \
            make_float2(temp24, temp34);                    \
        tt[index + tile_x / 2 + 2 + (tile_y + 4) * F / 2] = \
            make_float2(temp44, temp54);                    \
        tt[index + tile_x / 2 + 3 + (tile_y + 4) * F / 2] = \
            make_float2(temp64, temp74);                    \
        tt[index + tile_x / 2 + 4 + (tile_y + 4) * F / 2] = \
            make_float2(temp84, temp94);                    \
                                                            \
        tt[index + tile_x / 2 + 0 + (tile_y + 5) * F / 2] = \
            make_float2(temp5, temp15);                     \
        tt[index + tile_x / 2 + 1 + (tile_y + 5) * F / 2] = \
            make_float2(temp25, temp35);                    \
        tt[index + tile_x / 2 + 2 + (tile_y + 5) * F / 2] = \
            make_float2(temp45, temp55);                    \
        tt[index + tile_x / 2 + 3 + (tile_y + 5) * F / 2] = \
            make_float2(temp65, temp75);                    \
        tt[index + tile_x / 2 + 4 + (tile_y + 5) * F / 2] = \
            make_float2(temp85, temp95);                    \
                                                            \
        tt[index + tile_x / 2 + 0 + (tile_y + 6) * F / 2] = \
            make_float2(temp6, temp16);                     \
        tt[index + tile_x / 2 + 1 + (tile_y + 6) * F / 2] = \
            make_float2(temp26, temp36);                    \
        tt[index + tile_x / 2 + 2 + (tile_y + 6) * F / 2] = \
            make_float2(temp46, temp56);                    \
        tt[index + tile_x / 2 + 3 + (tile_y + 6) * F / 2] = \
            make_float2(temp66, temp76);                    \
        tt[index + tile_x / 2 + 4 + (tile_y + 6) * F / 2] = \
            make_float2(temp86, temp96);                    \
                                                            \
        tt[index + tile_x / 2 + 0 + (tile_y + 7) * F / 2] = \
            make_float2(temp7, temp17);                     \
        tt[index + tile_x / 2 + 1 + (tile_y + 7) * F / 2] = \
            make_float2(temp27, temp37);                    \
        tt[index + tile_x / 2 + 2 + (tile_y + 7) * F / 2] = \
            make_float2(temp47, temp57);                    \
        tt[index + tile_x / 2 + 3 + (tile_y + 7) * F / 2] = \
            make_float2(temp67, temp77);                    \
        tt[index + tile_x / 2 + 4 + (tile_y + 7) * F / 2] = \
            make_float2(temp87, temp97);                    \
                                                            \
        tt[index + tile_x / 2 + 0 + (tile_y + 8) * F / 2] = \
            make_float2(temp8, temp18);                     \
        tt[index + tile_x / 2 + 1 + (tile_y + 8) * F / 2] = \
            make_float2(temp28, temp38);                    \
        tt[index + tile_x / 2 + 2 + (tile_y + 8) * F / 2] = \
            make_float2(temp48, temp58);                    \
        tt[index + tile_x / 2 + 3 + (tile_y + 8) * F / 2] = \
            make_float2(temp68, temp78);                    \
        tt[index + tile_x / 2 + 4 + (tile_y + 8) * F / 2] = \
            make_float2(temp88, temp98);                    \
                                                            \
        tt[index + tile_x / 2 + 0 + (tile_y + 9) * F / 2] = \
            make_float2(temp9, temp19);                     \
        tt[index + tile_x / 2 + 1 + (tile_y + 9) * F / 2] = \
            make_float2(temp29, temp39);                    \
        tt[index + tile_x / 2 + 2 + (tile_y + 9) * F / 2] = \
            make_float2(temp49, temp59);                    \
        tt[index + tile_x / 2 + 3 + (tile_y + 9) * F / 2] = \
            make_float2(temp69, temp79);                    \
        tt[index + tile_x / 2 + 4 + (tile_y + 9) * F / 2] = \
            make_float2(temp89, temp99);                    \
    } while (0)

#define fill_lower_half_from_registers_float2()             \
    do                                                      \
    {                                                       \
        tt[index + tile_y / 2 + 0 + (tile_x + 0) * F / 2] = \
            make_float2(temp0, temp1);                      \
        tt[index + tile_y / 2 + 1 + (tile_x + 0) * F / 2] = \
            make_float2(temp2, temp3);                      \
        tt[index + tile_y / 2 + 2 + (tile_x + 0) * F / 2] = \
            make_float2(temp4, temp5);                      \
        tt[index + tile_y / 2 + 3 + (tile_x + 0) * F / 2] = \
            make_float2(temp6, temp7);                      \
        tt[index + tile_y / 2 + 4 + (tile_x + 0) * F / 2] = \
            make_float2(temp8, temp9);                      \
                                                            \
        tt[index + tile_y / 2 + 0 + (tile_x + 1) * F / 2] = \
            make_float2(temp10, temp11);                    \
        tt[index + tile_y / 2 + 1 + (tile_x + 1) * F / 2] = \
            make_float2(temp12, temp13);                    \
        tt[index + tile_y / 2 + 2 + (tile_x + 1) * F / 2] = \
            make_float2(temp14, temp15);                    \
        tt[index + tile_y / 2 + 3 + (tile_x + 1) * F / 2] = \
            make_float2(temp16, temp17);                    \
        tt[index + tile_y / 2 + 4 + (tile_x + 1) * F / 2] = \
            make_float2(temp18, temp19);                    \
                                                            \
        tt[index + tile_y / 2 + 0 + (tile_x + 2) * F / 2] = \
            make_float2(temp20, temp21);                    \
        tt[index + tile_y / 2 + 1 + (tile_x + 2) * F / 2] = \
            make_float2(temp22, temp23);                    \
        tt[index + tile_y / 2 + 2 + (tile_x + 2) * F / 2] = \
            make_float2(temp24, temp25);                    \
        tt[index + tile_y / 2 + 3 + (tile_x + 2) * F / 2] = \
            make_float2(temp26, temp27);                    \
        tt[index + tile_y / 2 + 4 + (tile_x + 2) * F / 2] = \
            make_float2(temp28, temp29);                    \
                                                            \
        tt[index + tile_y / 2 + 0 + (tile_x + 3) * F / 2] = \
            make_float2(temp30, temp31);                    \
        tt[index + tile_y / 2 + 1 + (tile_x + 3) * F / 2] = \
            make_float2(temp32, temp33);                    \
        tt[index + tile_y / 2 + 2 + (tile_x + 3) * F / 2] = \
            make_float2(temp34, temp35);                    \
        tt[index + tile_y / 2 + 3 + (tile_x + 3) * F / 2] = \
            make_float2(temp36, temp37);                    \
        tt[index + tile_y / 2 + 4 + (tile_x + 3) * F / 2] = \
            make_float2(temp38, temp39);                    \
                                                            \
        tt[index + tile_y / 2 + 0 + (tile_x + 4) * F / 2] = \
            make_float2(temp40, temp41);                    \
        tt[index + tile_y / 2 + 1 + (tile_x + 4) * F / 2] = \
            make_float2(temp42, temp43);                    \
        tt[index + tile_y / 2 + 2 + (tile_x + 4) * F / 2] = \
            make_float2(temp44, temp45);                    \
        tt[index + tile_y / 2 + 3 + (tile_x + 4) * F / 2] = \
            make_float2(temp46, temp47);                    \
        tt[index + tile_y / 2 + 4 + (tile_x + 4) * F / 2] = \
            make_float2(temp48, temp49);                    \
                                                            \
        tt[index + tile_y / 2 + 0 + (tile_x + 5) * F / 2] = \
            make_float2(temp50, temp51);                    \
        tt[index + tile_y / 2 + 1 + (tile_x + 5) * F / 2] = \
            make_float2(temp52, temp53);                    \
        tt[index + tile_y / 2 + 2 + (tile_x + 5) * F / 2] = \
            make_float2(temp54, temp55);                    \
        tt[index + tile_y / 2 + 3 + (tile_x + 5) * F / 2] = \
            make_float2(temp56, temp57);                    \
        tt[index + tile_y / 2 + 4 + (tile_x + 5) * F / 2] = \
            make_float2(temp58, temp59);                    \
                                                            \
        tt[index + tile_y / 2 + 0 + (tile_x + 6) * F / 2] = \
            make_float2(temp60, temp61);                    \
        tt[index + tile_y / 2 + 1 + (tile_x + 6) * F / 2] = \
            make_float2(temp62, temp63);                    \
        tt[index + tile_y / 2 + 2 + (tile_x + 6) * F / 2] = \
            make_float2(temp64, temp65);                    \
        tt[index + tile_y / 2 + 3 + (tile_x + 6) * F / 2] = \
            make_float2(temp66, temp67);                    \
        tt[index + tile_y / 2 + 4 + (tile_x + 6) * F / 2] = \
            make_float2(temp68, temp69);                    \
                                                            \
        tt[index + tile_y / 2 + 0 + (tile_x + 7) * F / 2] = \
            make_float2(temp70, temp71);                    \
        tt[index + tile_y / 2 + 1 + (tile_x + 7) * F / 2] = \
            make_float2(temp72, temp73);                    \
        tt[index + tile_y / 2 + 2 + (tile_x + 7) * F / 2] = \
            make_float2(temp74, temp75);                    \
        tt[index + tile_y / 2 + 3 + (tile_x + 7) * F / 2] = \
            make_float2(temp76, temp77);                    \
        tt[index + tile_y / 2 + 4 + (tile_x + 7) * F / 2] = \
            make_float2(temp78, temp79);                    \
                                                            \
        tt[index + tile_y / 2 + 0 + (tile_x + 8) * F / 2] = \
            make_float2(temp80, temp81);                    \
        tt[index + tile_y / 2 + 1 + (tile_x + 8) * F / 2] = \
            make_float2(temp82, temp83);                    \
        tt[index + tile_y / 2 + 2 + (tile_x + 8) * F / 2] = \
            make_float2(temp84, temp85);                    \
        tt[index + tile_y / 2 + 3 + (tile_x + 8) * F / 2] = \
            make_float2(temp86, temp87);                    \
        tt[index + tile_y / 2 + 4 + (tile_x + 8) * F / 2] = \
            make_float2(temp88, temp89);                    \
                                                            \
        tt[index + tile_y / 2 + 0 + (tile_x + 9) * F / 2] = \
            make_float2(temp90, temp91);                    \
        tt[index + tile_y / 2 + 1 + (tile_x + 9) * F / 2] = \
            make_float2(temp92, temp93);                    \
        tt[index + tile_y / 2 + 2 + (tile_x + 9) * F / 2] = \
            make_float2(temp94, temp95);                    \
        tt[index + tile_y / 2 + 3 + (tile_x + 9) * F / 2] = \
            make_float2(temp96, temp97);                    \
        tt[index + tile_y / 2 + 4 + (tile_x + 9) * F / 2] = \
            make_float2(temp98, temp99);                    \
    } while (0)
#define cudacall(call)   \
    do                   \
    {                    \
        CUDACHECK(call); \
    } while (0)

#define cublascall(call)                                                        \
    do                                                                          \
    {                                                                           \
        cublasStatus_t status = (call);                                         \
        if (CUBLAS_STATUS_SUCCESS != status)                                    \
        {                                                                       \
            fprintf(stderr, "CUBLAS Error:\nFile = %s\nLine = %d\nCode = %d\n", \
                    __FILE__, __LINE__, status);                                \
            cudaDeviceReset();                                                  \
            exit(EXIT_FAILURE);                                                 \
        }                                                                       \
    } while (0)

#define cusparsecall(call)                                                        \
    do                                                                            \
    {                                                                             \
        cusparseStatus_t status = (call);                                         \
        if (CUSPARSE_STATUS_SUCCESS != status)                                    \
        {                                                                         \
            fprintf(stderr, "CUSPARSE Error:\nFile = %s\nLine = %d\nCode = %d\n", \
                    __FILE__, __LINE__, status);                                  \
            cudaDeviceReset();                                                    \
            exit(EXIT_FAILURE);                                                   \
        }                                                                         \
    } while (0)

#define cudaCheckError()                                             \
    {                                                                \
        cudaError_t e = cudaGetLastError();                          \
        if (e != cudaSuccess)                                        \
        {                                                            \
            printf("CUDA failure %s:%d: '%s'\n", __FILE__, __LINE__, \
                   cudaGetErrorString(e));                           \
            exit(EXIT_FAILURE);                                      \
        }                                                            \
    }                                                                \
    while (0)

// CG (iterative solve) kernel
// each block solves a A*x=b
__global__ void updateXWithCGKernel(float *A, float *x, float *b,
                                    const int batchSize, const int f,
                                    const float cgIter)
{
    extern __shared__ float smem[];
    float *sharedx = &smem[0];
    float *sharedp = &smem[f];
    float *sharedr = &smem[2 * f];
    float *sharedap = &smem[3 * f];
    float *rsold = &smem[4 * f];
    float *alpha = &smem[4 * f + 1];
    float *rsnew = &smem[4 * f + 2];
    float *beta = &smem[4 * f + 3];

    // sharedx<--x
    sharedx[threadIdx.x] = x[blockIdx.x * blockDim.x + threadIdx.x];
    // sharedx[threadIdx.x] = 0;
    __syncthreads();
    // r=b-A*x;
    float temp = 0;
    for (int i = 0; i < f; i++)
        // this is math correct and coalesced because A is symmetric
        temp += A[blockIdx.x * f * f + f * i + threadIdx.x] * sharedx[i];
    sharedr[threadIdx.x] = b[blockIdx.x * blockDim.x + threadIdx.x] - temp;
    // p=r;
    sharedp[threadIdx.x] = sharedr[threadIdx.x];
    // rsold=r'*r;
    if (threadIdx.x == 0)
    {
        rsold[0] = 0;
    }
    temp = sharedr[threadIdx.x] * sharedr[threadIdx.x];
    blockReduceSumWithAtomics(rsold, temp);
    // temp = blockReduceSum(shared, temp);
    __syncthreads();
#ifdef DEBUG
    if (threadIdx.x == 0)
    {
        printf("***rsold:\n");
        printf("rsold = %f \n", rsold[0]);
        printf("***shared memory content after 1st blockReduceSum:\n");
        for (int i = 0; i < f; i++)
            printf("%f ", sharedp[i]);
        printf("\n");
        for (int i = 0; i < f; i++)
            printf("%f ", sharedr[i]);
        printf("\n");
        for (int i = 0; i < f; i++)
            printf("%f ", sharedap[i]);
        printf("\n");
    }
#endif

    for (int iter = 0; iter < cgIter; iter++)
    {
        // ap=A*p;
        // WARN: set temp to zero since the next operation is +=!
        temp = 0;
        for (int i = 0; i < f; i++)
            // this is math correct and coalesced because A is symmetric
            temp += A[blockIdx.x * f * f + f * i + threadIdx.x] * sharedp[i];
        sharedap[threadIdx.x] = temp;
#ifdef DEBUG
        __syncthreads();
        if (threadIdx.x == 0)
        {
            printf("----------CG iteration %d \n", iter);
            printf("***ap:\n");
            for (int i = 0; i < f; i++)
                printf("%f ", sharedap[i]);
            printf("\n");
            printf("***shared memory content before 2rd blockReduceSum:\n");
            for (int i = 0; i < f; i++)
                printf("%f ", sharedp[i]);
            printf("\n");
            for (int i = 0; i < f; i++)
                printf("%f ", sharedr[i]);
            printf("\n");
            for (int i = 0; i < f; i++)
                printf("%f ", sharedap[i]);
            printf("\n");
        }
#endif
        if (threadIdx.x == 0)
        {
            rsnew[0] = 0;
        }
        // no need to have sync before blockReduce
        // because there is a __syncthreads() in blockReduce
        // pAp=p'*Ap
        temp = sharedp[threadIdx.x] * sharedap[threadIdx.x];
        // temp = blockReduceSum(shared, temp);
        blockReduceSumWithAtomics(rsnew, temp);
        // sync needed, to let all atomicAdd threads completes
        __syncthreads();
        if (threadIdx.x == 0)
        {
            // pAp = temp;
            // alpha=rsold/(p'*Ap); use rsnew to store pAp
            alpha[0] = rsold[0] / rsnew[0];
#ifdef DEBUG
            printf("***rsold:\n");
            printf("rsold = %f \n", rsold[0]);
            printf("***pAp:\n");
            printf("pAp = %f \n", rsnew[0]);
            printf("***alpha:\n");
            printf("alpha = %f \n", alpha[0]);
#endif
            rsnew[0] = 0;
        }
        // needed, aplpha[0] to be used by all threads
        __syncthreads();
        // x=x+alpha*p;
        sharedx[threadIdx.x] =
            sharedx[threadIdx.x] + alpha[0] * sharedp[threadIdx.x];
        // r=r-alpha*Ap;
        sharedr[threadIdx.x] =
            sharedr[threadIdx.x] - alpha[0] * sharedap[threadIdx.x];
        // NOT needed?
        __syncthreads();
#ifdef DEBUG
        __syncthreads();
        if (threadIdx.x == 0)
        {
            printf("***shared memory content before 3rd blockReduceSum:\n");
            for (int i = 0; i < f; i++)
                printf("%f ", sharedp[i]);
            printf("\n");
            for (int i = 0; i < f; i++)
                printf("%f ", sharedr[i]);
            printf("\n");
            for (int i = 0; i < f; i++)
                printf("%f ", sharedap[i]);
            printf("\n");
        }
#endif

        // rsnew=r'*r;
        /*
temp = sharedr[threadIdx.x]*sharedr[threadIdx.x];
temp = blockReduceSum(shared, temp);
__syncthreads();
if(threadIdx.x == 0){
rsnew[0] = temp;
}
*/
        temp = sharedr[threadIdx.x] * sharedr[threadIdx.x];
        blockReduceSumWithAtomics(rsnew, temp);
        // WARN: has to have this sync!
        __syncthreads();

#ifdef DEBUG
        if (threadIdx.x == 0)
        {
            printf("***rsnew:\n");
            printf("rsnew = %f \n", rsnew[0]);
            printf("***shared memory content after 3rd blockReduceSum:\n");
            for (int i = 0; i < f; i++)
                printf("%f ", sharedp[i]);
            printf("\n");
            for (int i = 0; i < f; i++)
                printf("%f ", sharedr[i]);
            printf("\n");
            for (int i = 0; i < f; i++)
                printf("%f ", sharedap[i]);
            printf("\n");
        }
#endif
        if (rsnew[0] < CG_ERROR)
            break;
        // NOT needed?
        //__syncthreads();
        // beta
        if (threadIdx.x == 0)
        {
            beta[0] = rsnew[0] / rsold[0];
            // rsold=rsnew;
            rsold[0] = rsnew[0];
        }
        // need sync since every thread needs beta[0]
        __syncthreads();
        // p=r+(rsnew/rsold)*p;
        sharedp[threadIdx.x] =
            sharedr[threadIdx.x] + beta[0] * sharedp[threadIdx.x];
        // need sync as every thread needs sharedp at the beginning of for
        __syncthreads();
#ifdef DEBUG
        __syncthreads();
        if (threadIdx.x == 0)
        {
            printf("***shared memory content after update p:\n");
            for (int i = 0; i < f; i++)
                printf("%f ", sharedp[i]);
            printf("\n");
            for (int i = 0; i < f; i++)
                printf("%f ", sharedr[i]);
            printf("\n");
            for (int i = 0; i < f; i++)
                printf("%f ", sharedap[i]);
            printf("\n");
        }
        __syncthreads();
#endif
    } // end of CG iterations
    // x<--sharedx
    x[blockIdx.x * blockDim.x + threadIdx.x] = sharedx[threadIdx.x];
}

// CG (iterative solve) kernel
// each block solves a A*x=b and A in fp16
__global__ void updateXWithCGKernel3(half *A, float *x, float *b,
                                     const int batchSize, const int f,
                                     const float cgIter)
{
    extern __shared__ float smem[];
    float *sharedx = &smem[0];
    float *sharedp = &smem[f];
    float *sharedr = &smem[2 * f];
    float *sharedap = &smem[3 * f];
    float *rsold = &smem[4 * f];
    float *alpha = &smem[4 * f + 1];
    float *rsnew = &smem[4 * f + 2];
    float *beta = &smem[4 * f + 3];

    // sharedx<--x
    sharedx[threadIdx.x] = x[blockIdx.x * blockDim.x + threadIdx.x];
    __syncthreads();
    // r=b-A*x;
    float temp = 0;
    for (int i = 0; i < f; i++)
        // this is math correct and coalesced because A is symmetric
        temp +=
            __half2float(A[blockIdx.x * f * f + f * i + threadIdx.x]) * sharedx[i];
    sharedr[threadIdx.x] = b[blockIdx.x * blockDim.x + threadIdx.x] - temp;
    // p=r;
    sharedp[threadIdx.x] = sharedr[threadIdx.x];
    // rsold=r'*r;
    if (threadIdx.x == 0)
    {
        rsold[0] = 0;
    }
    temp = sharedr[threadIdx.x] * sharedr[threadIdx.x];
    blockReduceSumWithAtomics(rsold, temp);
    // temp = blockReduceSum(shared, temp);
    __syncthreads();
#ifdef DEBUG
    if (threadIdx.x == 0)
    {
        printf("***rsold:\n");
        printf("rsold = %f \n", rsold[0]);
        printf("***shared memory content after 1st blockReduceSum:\n");
        for (int i = 0; i < f; i++)
            printf("%f ", sharedp[i]);
        printf("\n");
        for (int i = 0; i < f; i++)
            printf("%f ", sharedr[i]);
        printf("\n");
        for (int i = 0; i < f; i++)
            printf("%f ", sharedap[i]);
        printf("\n");
    }
#endif

    for (int iter = 0; iter < cgIter; iter++)
    {
        // ap=A*p;
        // WARN: set temp to zero since the next operation is +=!
        temp = 0;
        for (int i = 0; i < f; i++)
            // this is math correct and coalesced because A is symmetric
            temp += __half2float(A[blockIdx.x * f * f + f * i + threadIdx.x]) *
                    sharedp[i];
        sharedap[threadIdx.x] = temp;
#ifdef DEBUG
        __syncthreads();
        if (threadIdx.x == 0)
        {
            printf("----------CG iteration %d \n", iter);
            printf("***ap:\n");
            for (int i = 0; i < f; i++)
                printf("%f ", sharedap[i]);
            printf("\n");
            printf("***shared memory content before 2rd blockReduceSum:\n");
            for (int i = 0; i < f; i++)
                printf("%f ", sharedp[i]);
            printf("\n");
            for (int i = 0; i < f; i++)
                printf("%f ", sharedr[i]);
            printf("\n");
            for (int i = 0; i < f; i++)
                printf("%f ", sharedap[i]);
            printf("\n");
        }
#endif
        if (threadIdx.x == 0)
        {
            rsnew[0] = 0;
        }
        // no need to have sync before blockReduce
        // because there is a __syncthreads() in blockReduce
        // pAp=p'*Ap
        temp = sharedp[threadIdx.x] * sharedap[threadIdx.x];
        // temp = blockReduceSum(shared, temp);
        blockReduceSumWithAtomics(rsnew, temp);
        // sync needed, to let all atomicAdd threads completes
        __syncthreads();
        if (threadIdx.x == 0)
        {
            // pAp = temp;
            // alpha=rsold/(p'*Ap); use rsnew to store pAp
            alpha[0] = rsold[0] / rsnew[0];
#ifdef DEBUG
            printf("***rsold:\n");
            printf("rsold = %f \n", rsold[0]);
            printf("***pAp:\n");
            printf("pAp = %f \n", rsnew[0]);
            printf("***alpha:\n");
            printf("alpha = %f \n", alpha[0]);
#endif
            rsnew[0] = 0;
        }
        // needed, aplpha[0] to be used by all threads
        __syncthreads();
        // x=x+alpha*p;
        sharedx[threadIdx.x] =
            sharedx[threadIdx.x] + alpha[0] * sharedp[threadIdx.x];
        // r=r-alpha*Ap;
        sharedr[threadIdx.x] =
            sharedr[threadIdx.x] - alpha[0] * sharedap[threadIdx.x];
// NOT needed?
//__syncthreads();
#ifdef DEBUG
        __syncthreads();
        if (threadIdx.x == 0)
        {
            printf("***shared memory content before 3rd blockReduceSum:\n");
            for (int i = 0; i < f; i++)
                printf("%f ", sharedp[i]);
            printf("\n");
            for (int i = 0; i < f; i++)
                printf("%f ", sharedr[i]);
            printf("\n");
            for (int i = 0; i < f; i++)
                printf("%f ", sharedap[i]);
            printf("\n");
        }
#endif

        // rsnew=r'*r;
        /*
temp = sharedr[threadIdx.x]*sharedr[threadIdx.x];
temp = blockReduceSum(shared, temp);
__syncthreads();
if(threadIdx.x == 0){
rsnew[0] = temp;
}
*/
        temp = sharedr[threadIdx.x] * sharedr[threadIdx.x];
        blockReduceSumWithAtomics(rsnew, temp);
        // WARN: has to have this sync!
        __syncthreads();

#ifdef DEBUG
        if (threadIdx.x == 0)
        {
            printf("***rsnew:\n");
            printf("rsnew = %f \n", rsnew[0]);
            printf("***shared memory content after 3rd blockReduceSum:\n");
            for (int i = 0; i < f; i++)
                printf("%f ", sharedp[i]);
            printf("\n");
            for (int i = 0; i < f; i++)
                printf("%f ", sharedr[i]);
            printf("\n");
            for (int i = 0; i < f; i++)
                printf("%f ", sharedap[i]);
            printf("\n");
        }
#endif
        if (rsnew[0] < CG_ERROR)
            break;
        // NOT needed?
        //__syncthreads();
        // beta
        if (threadIdx.x == 0)
        {
            beta[0] = rsnew[0] / rsold[0];
            // rsold=rsnew;
            rsold[0] = rsnew[0];
        }
        // need sync since every thread needs beta[0]
        __syncthreads();
        // p=r+(rsnew/rsold)*p;
        sharedp[threadIdx.x] =
            sharedr[threadIdx.x] + beta[0] * sharedp[threadIdx.x];
        // need sync as every thread needs sharedp at the beginning of for
        __syncthreads();
#ifdef DEBUG
        __syncthreads();
        if (threadIdx.x == 0)
        {
            printf("***shared memory content after update p:\n");
            for (int i = 0; i < f; i++)
                printf("%f ", sharedp[i]);
            printf("\n");
            for (int i = 0; i < f; i++)
                printf("%f ", sharedr[i]);
            printf("\n");
            for (int i = 0; i < f; i++)
                printf("%f ", sharedap[i]);
            printf("\n");
        }
        __syncthreads();
#endif
    } // end of CG iterations
    // x<--sharedx
    x[blockIdx.x * blockDim.x + threadIdx.x] = sharedx[threadIdx.x];
}

// blockDim.x=64 or 96 (two or three WARPs) instead of 100 -- WARP shuffle seems
// requiring this
__global__ void updateXWithCGKernel2(float *A, float *x, float *b,
                                     const int batchSize, const int f,
                                     const float cgIter)
{
    extern __shared__ float smem[];
    float *sharedx = &smem[0];
    float *sharedp = &smem[f];
    float *sharedr = &smem[2 * f];
    float *sharedap = &smem[3 * f];
    float *rsold = &smem[4 * f];
    float *alpha = &smem[4 * f + 1];
    float *rsnew = &smem[4 * f + 2];
    float *beta = &smem[4 * f + 3];

    // sharedx<--x
    for (int k = threadIdx.x; k < f; k += blockDim.x)
        sharedx[k] = x[blockIdx.x * f + k];
    __syncthreads();
    // r=b-A*x;
    float temp = 0;
    for (int k = threadIdx.x; k < f; k += blockDim.x)
    {
        temp = 0;
        for (int i = 0; i < f; i++)
            temp += A[blockIdx.x * f * f + f * i + k] * sharedx[i];
        sharedr[k] = b[blockIdx.x * f + k] - temp;
        // p=r;
        sharedp[k] = sharedr[k];
    }
    // rsold=r'*r;
    if (threadIdx.x == 0)
    {
        rsold[0] = 0;
    }
    temp = 0;
    for (int k = threadIdx.x; k < f; k += blockDim.x)
    {
        temp += sharedr[k] * sharedr[k];
    }
    blockReduceSumWithAtomics(rsold, temp);
    // temp = blockReduceSum(shared, temp);
    __syncthreads();
#ifdef DEBUG
    if (threadIdx.x == 0)
    {
        printf("***rsold:\n");
        printf("rsold = %f \n", rsold[0]);
        printf("***shared memory content after 1st blockReduceSum:\n");
        for (int i = 0; i < f; i++)
            printf("%f ", sharedp[i]);
        printf("\n");
        for (int i = 0; i < f; i++)
            printf("%f ", sharedr[i]);
        printf("\n");
        for (int i = 0; i < f; i++)
            printf("%f ", sharedap[i]);
        printf("\n");
    }
#endif

    for (int iter = 0; iter < cgIter; iter++)
    {
        // ap=A*p;
        // WARN: set temp to zero since the next operation is +=!
        for (int k = threadIdx.x; k < f; k += blockDim.x)
        {
            temp = 0;
            for (int i = 0; i < f; i++)
                temp += A[blockIdx.x * f * f + f * i + k] * sharedp[i];
            sharedap[k] = temp;
        }
#ifdef DEBUG
        __syncthreads();
        if (threadIdx.x == 0)
        {
            printf("----------CG iteration %d \n", iter);
            printf("***ap:\n");
            for (int i = 0; i < f; i++)
                printf("%f ", sharedap[i]);
            printf("\n");
            printf("***shared memory content before 2rd blockReduceSum:\n");
            for (int i = 0; i < f; i++)
                printf("%f ", sharedp[i]);
            printf("\n");
            for (int i = 0; i < f; i++)
                printf("%f ", sharedr[i]);
            printf("\n");
            for (int i = 0; i < f; i++)
                printf("%f ", sharedap[i]);
            printf("\n");
        }
#endif
        if (threadIdx.x == 0)
        {
            rsnew[0] = 0;
        }
        // no need to have sync before blockReduce
        // because there is a __syncthreads() in blockReduce
        // pAp=p'*Ap
        temp = 0;
        for (int k = threadIdx.x; k < f; k += blockDim.x)
            temp += sharedp[k] * sharedap[k];
        // temp = blockReduceSum(shared, temp);
        blockReduceSumWithAtomics(rsnew, temp);
        // sync needed, to let all atomicAdd threads completes
        __syncthreads();
        if (threadIdx.x == 0)
        {
            // pAp = temp;
            // alpha=rsold/(p'*Ap); use rsnew to store pAp
            alpha[0] = rsold[0] / rsnew[0];
#ifdef DEBUG
            printf("***rsold:\n");
            printf("rsold = %f \n", rsold[0]);
            printf("***pAp:\n");
            printf("pAp = %f \n", rsnew[0]);
            printf("***alpha:\n");
            printf("alpha = %f \n", alpha[0]);
#endif
            rsnew[0] = 0;
        }
        // needed, aplpha[0] to be used by all threads
        __syncthreads();
        for (int k = threadIdx.x; k < f; k += blockDim.x)
        {
            // x=x+alpha*p;
            sharedx[k] = sharedx[k] + alpha[0] * sharedp[k];
            // r=r-alpha*Ap;
            sharedr[k] = sharedr[k] - alpha[0] * sharedap[k];
        }
// NOT needed?
//__syncthreads();
#ifdef DEBUG
        __syncthreads();
        if (threadIdx.x == 0)
        {
            printf("***shared memory content before 3rd blockReduceSum:\n");
            for (int i = 0; i < f; i++)
                printf("%f ", sharedp[i]);
            printf("\n");
            for (int i = 0; i < f; i++)
                printf("%f ", sharedr[i]);
            printf("\n");
            for (int i = 0; i < f; i++)
                printf("%f ", sharedap[i]);
            printf("\n");
        }
#endif

        // rsnew=r'*r;
        /*
temp = sharedr[threadIdx.x]*sharedr[threadIdx.x];
temp = blockReduceSum(shared, temp);
__syncthreads();
if(threadIdx.x == 0){
rsnew[0] = temp;
}
*/
        temp = 0;
        for (int k = threadIdx.x; k < f; k += blockDim.x)
            temp += sharedr[k] * sharedr[k];
        blockReduceSumWithAtomics(rsnew, temp);
        // WARN: has to have this sync!
        __syncthreads();

#ifdef DEBUG
        if (threadIdx.x == 0)
        {
            printf("***rsnew:\n");
            printf("rsnew = %f \n", rsnew[0]);
            printf("***shared memory content after 3rd blockReduceSum:\n");
            for (int i = 0; i < f; i++)
                printf("%f ", sharedp[i]);
            printf("\n");
            for (int i = 0; i < f; i++)
                printf("%f ", sharedr[i]);
            printf("\n");
            for (int i = 0; i < f; i++)
                printf("%f ", sharedap[i]);
            printf("\n");
        }
#endif
        if (rsnew[0] < CG_ERROR)
            break;
        // NOT needed?
        //__syncthreads();
        // beta
        if (threadIdx.x == 0)
        {
            beta[0] = rsnew[0] / rsold[0];
            // rsold=rsnew;
            rsold[0] = rsnew[0];
        }
        // need sync since every thread needs beta[0]
        __syncthreads();
        for (int k = threadIdx.x; k < f; k += blockDim.x)
            // p=r+(rsnew/rsold)*p;
            sharedp[k] = sharedr[k] + beta[0] * sharedp[k];
        // need sync as every thread needs sharedp at the beginning of for
        __syncthreads();
#ifdef DEBUG
        __syncthreads();
        if (threadIdx.x == 0)
        {
            printf("***shared memory content after update p:\n");
            for (int i = 0; i < f; i++)
                printf("%f ", sharedp[i]);
            printf("\n");
            for (int i = 0; i < f; i++)
                printf("%f ", sharedr[i]);
            printf("\n");
            for (int i = 0; i < f; i++)
                printf("%f ", sharedap[i]);
            printf("\n");
        }
        __syncthreads();
#endif
    } // end of CG iterations
    for (int k = threadIdx.x; k < f; k += blockDim.x)
        // x<--sharedx
        x[blockIdx.x * f + k] = sharedx[k];
}

void updateXWithCGHost_tt_fp16(float *A, float *x, float *b,
                               const int batchSize, const int f,
                               const float cgIter)
{
    updateXWithCGKernel3<<<batchSize, f, (4 * f + 4) * sizeof(float)>>>(
        (half *)A, x, b, batchSize, f, cgIter);
    cudaDeviceSynchronize();
    cudaCheckError();

#ifdef DEBUG

    printf("***A[0]:\n");
    float *h_A = new float[f * f];
    float *A_fp32;
    cudacall(cudaMalloc((void **)&A_fp32, f * f * sizeof(A_fp32[0])));
    fp16Array2fp32Array<<<(f * f - 1) / 1024 + 1, 1024>>>(A_fp32, (half *)A,
                                                          f * f);
    cudaDeviceSynchronize();
    cudaCheckError();
    cudacall(
        cudaMemcpy(h_A, A_fp32, f * f * sizeof(float), cudaMemcpyDeviceToHost));
    for (int i = 0; i < f * f; i++)
        printf("%f ", h_A[i]);
    printf("\n");
    delete[] h_A;
    cudacall(cudaFree(A_fp32));

    printf("***x[0]:\n");
    float *h_x = new float[f];
    cudacall(cudaMemcpy(h_x, x, f * sizeof(float), cudaMemcpyDeviceToHost));
    for (int i = 0; i < f; i++)
        printf("%f ", h_x[i]);
    printf("\n");
    delete[] h_x;
/*
        printf("***b[0]:\n");
        float *h_b = new float[f];
        cudacall(cudaMemcpy(h_b, b, f * sizeof(float), cudaMemcpyDeviceToHost));
        for(int i = 0; i < f; i++)
                printf("%f ", h_b[i]);
        printf("\n");
        delete [] h_b;
        */
#endif
}

void updateXWithCGHost(float *A, float *x, float *b, const int batchSize,
                       const int f, const float cgIter)
{
    updateXWithCGKernel<<<batchSize, f, (4 * f + 4) * sizeof(float)>>>
        // updateXWithCGKernel2<<<batchSize, 96, (4*f+4)*sizeof(float)>>>
        (A, x, b, batchSize, f, cgIter);
    OK(cudaDeviceSynchronize());
    cudaCheckError();

#ifdef DEBUG

    printf("***A[0]:\n");
    float *h_A = new float[f * f];
    float *A_fp32;
    cudacall(cudaMalloc((void **)&A_fp32, f * f * sizeof(A_fp32[0])));
    fp16Array2fp32Array<<<(f * f - 1) / 1024 + 1, 1024>>>(A_fp32, (half *)A,
                                                          f * f);
    cudaDeviceSynchronize();
    cudaCheckError();
    cudacall(
        cudaMemcpy(h_A, A_fp32, f * f * sizeof(float), cudaMemcpyDeviceToHost));
    for (int i = 0; i < f * f; i++)
        printf("%f ", h_A[i]);
    printf("\n");
    delete[] h_A;
    cudacall(cudaFree(A_fp32));

    printf("***x[0]:\n");
    float *h_x = new float[f];
    cudacall(cudaMemcpy(h_x, x, f * sizeof(float), cudaMemcpyDeviceToHost));
    for (int i = 0; i < f; i++)
        printf("%f ", h_x[i]);
    printf("\n");
    delete[] h_x;
/*
        printf("***b[0]:\n");
        float *h_b = new float[f];
        cudacall(cudaMemcpy(h_b, b, f * sizeof(float), cudaMemcpyDeviceToHost));
        for(int i = 0; i < f; i++)
                printf("%f ", h_b[i]);
        printf("\n");
        delete [] h_b;
        */
#endif
}

int updateX(const int batch_size, const int batch_offset, float *ythetaT,
            float *tt, float *XT, cublasHandle_t handle, const int m,
            const int n, const int f, float **devPtrTTHost,
            float **devPtrYthetaTHost)
{
#ifdef DEBUG
    float elapsed;
    struct timeval tv0, tv1, tv2;
    gettimeofday(&tv0, NULL);
    printf("*******Batch LU factorization of tt.\n");
#endif
    // pointers needed by batch op
    float **devPtrTT = 0;
    int *INFO;
    for (int k = 0; k < batch_size; k++)
    {
        devPtrTTHost[k] = &tt[k * f * f];
    }
    cudacall(cudaMalloc((void **)&devPtrTT, batch_size * sizeof(*devPtrTT)));
    cudacall(cudaMemcpy(devPtrTT, devPtrTTHost, batch_size * sizeof(*devPtrTT),
                        cudaMemcpyHostToDevice));
    // cudacall( cudaMalloc(&P, f * batch_size * sizeof(int)) );
    cudacall(cudaMalloc(&INFO, batch_size * sizeof(int)));
    cublascall(
        cublasSgetrfBatched(handle, f, devPtrTT, f, NULL, INFO, batch_size));

    cudaThreadSynchronize();
    OK(cudaDeviceSynchronize());
#ifdef DEBUG
    gettimeofday(&tv1, NULL);
    elapsed = (tv1.tv_sec - tv0.tv_sec) + (tv1.tv_usec - tv0.tv_usec) / 1000000.0;
    printf("\t %f seconds. \n", elapsed);

    printf(
        "*******solve: tt * XT = ythetaT use cublas, with LU decomposition.\n");
#endif

    float **devPtrYthetaT = 0;

    for (int k = 0; k < batch_size; k++)
    {
        devPtrYthetaTHost[k] = &ythetaT[batch_offset * f + k * f];
    }
    cudacall(
        cudaMalloc((void **)&devPtrYthetaT, batch_size * sizeof(*devPtrYthetaT)));
    cudacall(cudaMemcpy(devPtrYthetaT, devPtrYthetaTHost,
                        batch_size * sizeof(*devPtrYthetaT),
                        cudaMemcpyHostToDevice));

    int *info2 = (int *)malloc(sizeof(int));
    cublascall(cublasSgetrsBatched(handle, CUBLAS_OP_N, f, 1,
                                   (const float **)devPtrTT, f, NULL,
                                   devPtrYthetaT, f, info2, batch_size));

    cudaThreadSynchronize();
    OK(cudaDeviceSynchronize());
    cudaError_t cudaStat1 = cudaGetLastError();
    if (cudaStat1 != cudaSuccess)
    {
        fprintf(stderr, "Failed to launch cublasSgetrsBatched (error code: %s)!\n",
                cudaGetErrorString(cudaStat1));
        exit(EXIT_FAILURE);
    }

    cudacall(cudaMemcpy(&XT[batch_offset * f], &ythetaT[batch_offset * f],
                        batch_size * f * sizeof(float),
                        cudaMemcpyDeviceToDevice));
#ifdef DEBUG
    gettimeofday(&tv2, NULL);
    elapsed = (tv2.tv_sec - tv1.tv_sec) + (tv2.tv_usec - tv1.tv_usec) / 1000000.0;
    printf("\t %f seconds. \n", elapsed);
#endif

    cudacall(cudaFree(devPtrTT));
    cudacall(cudaFree(INFO));
    free(info2);
    cudacall(cudaFree(devPtrYthetaT));
    return 0;
}

int updateTheta(const int batch_size, const int batch_offset, float *xx,
                float *yTXT, float *thetaT,
                cublasHandle_t handle,
                const int m, const int n, const int f,
                float **devPtrXXHost, float **devPtrYTXTHost)
{
#ifdef DEBUG
    float elapsed;
    struct timeval tv0, tv1, tv2;
    gettimeofday(&tv0, NULL);
    printf("*******LU factorize xx.\n");
#endif
    float **devPtrXX = 0;

    for (int k = 0; k < batch_size; k++)
    {
        devPtrXXHost[k] = &xx[k * f * f];
    }
    cudacall(cudaMalloc((void **)&devPtrXX, batch_size * sizeof(*devPtrXX)));
    cudacall(cudaMemcpy(devPtrXX, devPtrXXHost, batch_size * sizeof(*devPtrXX), cudaMemcpyHostToDevice));
    int *INFO;
    //cudacall(cudaMalloc(&P, f * batch_size * sizeof(int)));
    cudacall(cudaMalloc(&INFO, batch_size * sizeof(int)));
    cublascall(cublasSgetrfBatched(handle, f, devPtrXX, f, NULL, INFO, batch_size));
    cudaThreadSynchronize();
#ifdef DEBUG
    gettimeofday(&tv1, NULL);
    elapsed = (tv1.tv_sec - tv0.tv_sec) + (tv1.tv_usec - tv0.tv_usec) / 1000000.0;
    printf("\t %f seconds. \n", elapsed);

    printf("******* solve xx * thetaT = yTXT with CUDA 7.\n");
#endif
    float **devPtrYTXT = 0;

    for (int k = 0; k < batch_size; k++)
    {
        devPtrYTXTHost[k] = &yTXT[batch_offset * f + k * f];
    }

    cudacall(cudaMalloc((void **)&devPtrYTXT, batch_size * sizeof(*devPtrYTXT)));
    cudacall(cudaMemcpy(devPtrYTXT, devPtrYTXTHost, batch_size * sizeof(*devPtrYTXT), cudaMemcpyHostToDevice));

    int *info2 = (int *)malloc(sizeof(int));
    cublascall(cublasSgetrsBatched(handle, CUBLAS_OP_N, f, 1,
                                   (const float **)devPtrXX, f, NULL, devPtrYTXT, f, info2, batch_size));
    cudaThreadSynchronize();
    cudaError_t cudaStat1 = cudaGetLastError();
    if (cudaStat1 != cudaSuccess)
    {
        fprintf(stderr, "Failed to launch cublasSgetrsBatched (error code: %s)!\n", cudaGetErrorString(cudaStat1));
        exit(EXIT_FAILURE);
    }

    cudacall(cudaMemcpy(&thetaT[batch_offset * f], &yTXT[batch_offset * f],
                        batch_size * f * sizeof(float), cudaMemcpyDeviceToDevice));
#ifdef DEBUG
    gettimeofday(&tv2, NULL);
    elapsed = (tv2.tv_sec - tv1.tv_sec) + (tv2.tv_usec - tv1.tv_usec) / 1000000.0;
    printf("\t %f seconds. \n", elapsed);
#endif

    cudaFree(devPtrXX);
    cudaFree(INFO);
    free(info2);
    cudaFree(devPtrYTXT);
    return 0;
}

__global__ void RMSE(const float *csrVal, const int *cooRowIndex,
                     const int *csrColIndex, const float *__restrict__ thetaT,
                     const float *__restrict__ XT, float *error, const int nnz,
                     const int error_size, const int f)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < nnz)
    {
        int row = cooRowIndex[i];
        int col = csrColIndex[i];
        float e = csrVal[i];
        for (int k = 0; k < f; k++)
        {
            // a and b could be; there are user/item in testing but not training set
            float a = __ldg(&thetaT[f * col + k]);
            float b = __ldg(&XT[f * row + k]);
            // if(isnan(a)||isnan(b))//nan not working in some platform
            if (a != a || b != b)
                break;
            else
                e -= a * b;
        }
        atomicAdd(&error[i % error_size], e * e);
    }
}

// using fp16 as thetaT's format
// using fp16 in computate seems causing register pressure since half intrinsics
// cannot be used. using fp16 in compute also does not converge. not sure if the
// code is incorrect, or ALS cannot tolerate half-precision
__global__ void __launch_bounds__(64, 6)
    get_hermitian100WithHalf(const int batch_offset, float *tt,
                             const int *csrRowIndex, const int *csrColIndex,
                             const float lambda, const int m, const int F,
                             const half *__restrict__ thetaT_fp16)
{
    extern __shared__ float2 thetaTemp[];
    int row = blockIdx.x + batch_offset;
    if (row < m)
    {
        // this block needs to handle end - start thetaT columns
        int start = csrRowIndex[row];
        int end = csrRowIndex[row + 1];
        // slide through [start, end] by window size SCAN_BATCH
        int iterations = (end - start - 1) / SCAN_BATCH + 1;

        float temp0 = 0, temp1 = 0, temp2 = 0, temp3 = 0, temp4 = 0, temp5 = 0,
              temp6 = 0, temp7 = 0, temp8 = 0, temp9 = 0;
        float temp10 = 0, temp11 = 0, temp12 = 0, temp13 = 0, temp14 = 0,
              temp15 = 0, temp16 = 0, temp17 = 0, temp18 = 0, temp19 = 0;
        float temp20 = 0, temp21 = 0, temp22 = 0, temp23 = 0, temp24 = 0,
              temp25 = 0, temp26 = 0, temp27 = 0, temp28 = 0, temp29 = 0;
        float temp30 = 0, temp31 = 0, temp32 = 0, temp33 = 0, temp34 = 0,
              temp35 = 0, temp36 = 0, temp37 = 0, temp38 = 0, temp39 = 0;
        float temp40 = 0, temp41 = 0, temp42 = 0, temp43 = 0, temp44 = 0,
              temp45 = 0, temp46 = 0, temp47 = 0, temp48 = 0, temp49 = 0;
        float temp50 = 0, temp51 = 0, temp52 = 0, temp53 = 0, temp54 = 0,
              temp55 = 0, temp56 = 0, temp57 = 0, temp58 = 0, temp59 = 0;
        float temp60 = 0, temp61 = 0, temp62 = 0, temp63 = 0, temp64 = 0,
              temp65 = 0, temp66 = 0, temp67 = 0, temp68 = 0, temp69 = 0;
        float temp70 = 0, temp71 = 0, temp72 = 0, temp73 = 0, temp74 = 0,
              temp75 = 0, temp76 = 0, temp77 = 0, temp78 = 0, temp79 = 0;
        float temp80 = 0, temp81 = 0, temp82 = 0, temp83 = 0, temp84 = 0,
              temp85 = 0, temp86 = 0, temp87 = 0, temp88 = 0, temp89 = 0;
        float temp90 = 0, temp91 = 0, temp92 = 0, temp93 = 0, temp94 = 0,
              temp95 = 0, temp96 = 0, temp97 = 0, temp98 = 0, temp99 = 0;

        int tile_x = 0;
        int tile_y = 0;

        int tile = F / 10;
        for (int i = 0; i < 10; i++)
        {
            int end = ((20 - i) * (i + 1)) / 2;
            if (threadIdx.x < end)
            {
                tile_x = i * tile;
                tile_y = (10 + threadIdx.x - end) * tile;
                break;
            }
        }
        // iteration: copy gmem-->smem; aggregate smem-->register
        for (int iter = 0; iter < iterations; iter++)
        {
            // float2 theta;
            // copy texture --> smem, and sync
            // two layers: warp divergence unless we split at 32
            // require: 32 >= SCAN_BATCH
            if (threadIdx.x < 2 * 32)
            {
                int index = threadIdx.x - (threadIdx.x / 32) * 32; // 0 to 31;
                if (index < SCAN_BATCH)
                {
                    if (iter * SCAN_BATCH + index < end - start)
                    {
                        // for (int k = 50*(threadIdx.x/32); k < 50*(threadIdx.x/32) + 50; k
                        // += 2){ IMPORTANT: for loop has constant and identical start and
                        // end
                        if (threadIdx.x < 32)
                        {
                            for (int k = 0; k < 50; k += 2)
                            {
                                half2 theta_half2 = __ldg(
                                    (half2 *)&thetaT_fp16
                                        [F * csrColIndex[start + iter * SCAN_BATCH + index] +
                                         k]);
                                thetaTemp[index * F / 2 + k / 2] = __half22float2(theta_half2);
                                // theta.x = __half2float(__ldg(&thetaT_fp16[ F *
                                // csrColIndex[start + iter*SCAN_BATCH + index] + k])); theta.y
                                // =
                                // __half2float(__ldg(&thetaT_fp16[ F * csrColIndex[start +
                                // iter*SCAN_BATCH + index] + k+1])); thetaTemp[index * F/2 +
                                // k/2] = theta;
                            }
                        }
                        else
                        {
                            for (int k = 0; k < 50; k += 2)
                            {
                                half2 theta_half2 = __ldg(
                                    (half2 *)&thetaT_fp16
                                        [F * csrColIndex[start + iter * SCAN_BATCH + index] +
                                         k + 50]);
                                thetaTemp[index * F / 2 + k / 2 + 25] =
                                    __half22float2(theta_half2);
                                // theta.x = __half2float(__ldg(&thetaT_fp16[ F *
                                // csrColIndex[start + iter*SCAN_BATCH + index] + k + 50]));
                                // theta.y = __half2float(__ldg(&thetaT_fp16[ F *
                                // csrColIndex[start + iter*SCAN_BATCH + index] + k + 51]));
                                // thetaTemp[index * F/2 + k/2 + 25] = theta;
                            }
                        }
                    }
                    // must be the last iteration; no need to check
                    // not enough theta to copy, set zero
                    else
                        memset(&thetaTemp[index * F / 2], 0, F * sizeof(float));
                }
            }
            __syncthreads();
            // tile: 10*10
            if (threadIdx.x < 55)
            {
                for (int k = 0; k < SCAN_BATCH; k++)
                {
                    accumulate_in_registers();
                }
            }
        }
        // end of iteration in copying from smem and aggregating in register
        __syncthreads();

        if (threadIdx.x < 55)
        {
            // weighted-lambda regularization
            if (tile_x == tile_y)
            {
                float temp = (end - start) * lambda;
                temp0 += temp;
                temp11 += temp;
                temp22 += temp;
                temp33 += temp;
                temp44 += temp;
                temp55 += temp;
                temp66 += temp;
                temp77 += temp;
                temp88 += temp;
                temp99 += temp;
            }
            // copy output to gmem
            int index = blockIdx.x * F * F;
            fill_lower_half_from_registers();
            // symmetric
            if (tile_x != tile_y)
            {
                fill_upper_half_from_registers();
            }
        }
    }
}

__global__ void __launch_bounds__(64, 6)
    get_hermitian100_tt_fp16(const int batch_offset, half2 *tt,
                             const int *csrRowIndex, const int *csrColIndex,
                             const float lambda, const int m, const int F,
                             const float2 *__restrict__ thetaT)
{
    extern __shared__ float2 thetaTemp[];
    int row = blockIdx.x + batch_offset;
    if (row < m)
    {
        // this block needs to handle end - start thetaT columns
        int start = csrRowIndex[row];
        int end = csrRowIndex[row + 1];
        // slide through [start, end] by window size SCAN_BATCH
        int iterations = (end - start - 1) / SCAN_BATCH + 1;
        float temp0 = 0, temp1 = 0, temp2 = 0, temp3 = 0, temp4 = 0, temp5 = 0,
              temp6 = 0, temp7 = 0, temp8 = 0, temp9 = 0;
        float temp10 = 0, temp11 = 0, temp12 = 0, temp13 = 0, temp14 = 0,
              temp15 = 0, temp16 = 0, temp17 = 0, temp18 = 0, temp19 = 0;
        float temp20 = 0, temp21 = 0, temp22 = 0, temp23 = 0, temp24 = 0,
              temp25 = 0, temp26 = 0, temp27 = 0, temp28 = 0, temp29 = 0;
        float temp30 = 0, temp31 = 0, temp32 = 0, temp33 = 0, temp34 = 0,
              temp35 = 0, temp36 = 0, temp37 = 0, temp38 = 0, temp39 = 0;
        float temp40 = 0, temp41 = 0, temp42 = 0, temp43 = 0, temp44 = 0,
              temp45 = 0, temp46 = 0, temp47 = 0, temp48 = 0, temp49 = 0;
        float temp50 = 0, temp51 = 0, temp52 = 0, temp53 = 0, temp54 = 0,
              temp55 = 0, temp56 = 0, temp57 = 0, temp58 = 0, temp59 = 0;
        float temp60 = 0, temp61 = 0, temp62 = 0, temp63 = 0, temp64 = 0,
              temp65 = 0, temp66 = 0, temp67 = 0, temp68 = 0, temp69 = 0;
        float temp70 = 0, temp71 = 0, temp72 = 0, temp73 = 0, temp74 = 0,
              temp75 = 0, temp76 = 0, temp77 = 0, temp78 = 0, temp79 = 0;
        float temp80 = 0, temp81 = 0, temp82 = 0, temp83 = 0, temp84 = 0,
              temp85 = 0, temp86 = 0, temp87 = 0, temp88 = 0, temp89 = 0;
        float temp90 = 0, temp91 = 0, temp92 = 0, temp93 = 0, temp94 = 0,
              temp95 = 0, temp96 = 0, temp97 = 0, temp98 = 0, temp99 = 0;

        int tile_x = 0;
        int tile_y = 0;

        int tile = F / 10;
        for (int i = 0; i < 10; i++)
        {
            int end = ((20 - i) * (i + 1)) / 2;
            if (threadIdx.x < end)
            {
                tile_x = i * tile;
                tile_y = (10 + threadIdx.x - end) * tile;
                break;
            }
        }
        // iteration: copy gmem-->smem; aggregate smem-->register
        for (int iter = 0; iter < iterations; iter++)
        {
            // copy texture --> smem, and sync
            /*
This is the fastest implementation
thetaT is NOT coalesced loaded but cached by L1 and L2
faster than coalesced version (see the next paragraph
commented out) because it concurrently load multiple thetaT columns two
threads per theta column, e.g., threads 0 & 1 for theta[0], threads 2 &
3 for theta[1] require: blockDim.x (64) >= 2*SCAN_BATCH
*/
            ///*
            if (threadIdx.x < 2 * SCAN_BATCH)
            {
                int anchor = start + iter * SCAN_BATCH + threadIdx.x / 2;
                if (anchor < end)
                {
                    int col = csrColIndex[anchor];
                    // IMPORTANT: for loop has constant and identical start and end
                    for (int k = 0; k < 50; k += 2)
                        // thetaTemp[threadIdx.x*F/4 + k/2] =__ldg(&thetaT[ F/2 * col +
                        // threadIdx.x%2*F/4 + k/2]);
                        thetaTemp[threadIdx.x * F / 4 + k / 2] =
                            thetaT[F / 2 * col + threadIdx.x % 2 * F / 4 + k / 2];
                }
            }
            //*/
            __syncthreads();

            // tile: 10*10
            if (threadIdx.x < 55)
            {
                if (iter < iterations - 1)
                {
                    for (int k = 0; k < SCAN_BATCH; k++)
                        accumulate_in_registers();
                }
                else
                {
                    for (int k = 0; k < end - start - iter * SCAN_BATCH; k++)
                        accumulate_in_registers();
                }
            }
        }
        // end of iteration in copying from smem and aggregating in register
        __syncthreads();
#ifdef DEBUG
// if(threadIdx.x==0)
//	printf("***temp 0~9: %f %f %f %f %f %f %f %f %f %f\n", temp0, temp1,
// temp2, temp3, temp4, temp5, temp6, temp7, temp8, temp9);
#endif
        if (threadIdx.x < 55)
        {
            // weighted-lambda regularization
            if (tile_x == tile_y)
            {
                float temp = (end - start) * lambda;
                temp0 += temp;
                temp11 += temp;
                temp22 += temp;
                temp33 += temp;
                temp44 += temp;
                temp55 += temp;
                temp66 += temp;
                temp77 += temp;
                temp88 += temp;
                temp99 += temp;
            }
            // copy output to gmem
            int index = blockIdx.x * F * F / 2;
            // fill_lower_half_from_registers();
            fill_lower_half_from_registers_fp16();
            // symmetric
            if (tile_x != tile_y)
            {
                // fill_upper_half_from_registers();
                fill_upper_half_from_registers_fp16();
            }
        }
    }
}

__global__ void __launch_bounds__(64)
    get_hermitian100(const int batch_offset, float2 *tt, const int *csrRowIndex,
                     const int *csrColIndex, const float lambda, const int m,
                     const int F, const float2 *__restrict__ thetaT)
{
    extern __shared__ float2 thetaTemp[];
    int row = blockIdx.x + batch_offset;
    if (row < m)
    {
        // this block needs to handle end - start thetaT columns
        int start = csrRowIndex[row];
        int end = csrRowIndex[row + 1];
        // slide through [start, end] by window size SCAN_BATCH
        int iterations = (end - start - 1) / SCAN_BATCH + 1;
        float temp0 = 0, temp1 = 0, temp2 = 0, temp3 = 0, temp4 = 0, temp5 = 0,
              temp6 = 0, temp7 = 0, temp8 = 0, temp9 = 0;
        float temp10 = 0, temp11 = 0, temp12 = 0, temp13 = 0, temp14 = 0,
              temp15 = 0, temp16 = 0, temp17 = 0, temp18 = 0, temp19 = 0;
        float temp20 = 0, temp21 = 0, temp22 = 0, temp23 = 0, temp24 = 0,
              temp25 = 0, temp26 = 0, temp27 = 0, temp28 = 0, temp29 = 0;
        float temp30 = 0, temp31 = 0, temp32 = 0, temp33 = 0, temp34 = 0,
              temp35 = 0, temp36 = 0, temp37 = 0, temp38 = 0, temp39 = 0;
        float temp40 = 0, temp41 = 0, temp42 = 0, temp43 = 0, temp44 = 0,
              temp45 = 0, temp46 = 0, temp47 = 0, temp48 = 0, temp49 = 0;
        float temp50 = 0, temp51 = 0, temp52 = 0, temp53 = 0, temp54 = 0,
              temp55 = 0, temp56 = 0, temp57 = 0, temp58 = 0, temp59 = 0;
        float temp60 = 0, temp61 = 0, temp62 = 0, temp63 = 0, temp64 = 0,
              temp65 = 0, temp66 = 0, temp67 = 0, temp68 = 0, temp69 = 0;
        float temp70 = 0, temp71 = 0, temp72 = 0, temp73 = 0, temp74 = 0,
              temp75 = 0, temp76 = 0, temp77 = 0, temp78 = 0, temp79 = 0;
        float temp80 = 0, temp81 = 0, temp82 = 0, temp83 = 0, temp84 = 0,
              temp85 = 0, temp86 = 0, temp87 = 0, temp88 = 0, temp89 = 0;
        float temp90 = 0, temp91 = 0, temp92 = 0, temp93 = 0, temp94 = 0,
              temp95 = 0, temp96 = 0, temp97 = 0, temp98 = 0, temp99 = 0;

        int tile_x = 0;
        int tile_y = 0;

        int tile = F / 10;
        for (int i = 0; i < 10; i++)
        {
            int end = ((20 - i) * (i + 1)) / 2;
            if (threadIdx.x < end)
            {
                tile_x = i * tile;
                tile_y = (10 + threadIdx.x - end) * tile;
                break;
            }
        }
        // iteration: copy gmem-->smem; aggregate smem-->register
        for (int iter = 0; iter < iterations; iter++)
        {
            // copy texture --> smem, and sync
            /*
This is the fastest implementation
thetaT is NOT coalesced loaded but cached by L1 and L2
faster than coalesced version (see the next paragraph
commented out) because it concurrently load multiple thetaT columns two
threads per theta column, e.g., threads 0 & 1 for theta[0], threads 2 &
3 for theta[1] require: blockDim.x (64) >= 2*SCAN_BATCH
*/
            ///*
            if (threadIdx.x < 2 * SCAN_BATCH)
            {
                int anchor = start + iter * SCAN_BATCH + threadIdx.x / 2;
                if (anchor < end)
                {
                    int col = csrColIndex[anchor];
                    // IMPORTANT: for loop has constant and identical start and end
                    for (int k = 0; k < 50; k += 2)
                        // thetaTemp[threadIdx.x*F/4 + k/2] =__ldg(&thetaT[ F/2 * col +
                        // threadIdx.x%2*F/4 + k/2]);
                        thetaTemp[threadIdx.x * F / 4 + k / 2] =
                            thetaT[F / 2 * col + threadIdx.x % 2 * F / 4 + k / 2];
                }
            }
            //*/

            /*
//coalesced load thetaT, has to load column by column,
less concurrency, worse performance int anchor = start +
iter*SCAN_BATCH + threadIdx.x%32; int col_local; if(anchor < end &&
threadIdx.x%32 < SCAN_BATCH) col_local = csrColIndex[anchor]; int stop
= (end - start - iter*SCAN_BATCH < SCAN_BATCH)? end - start -
iter*SCAN_BATCH: SCAN_BATCH; for (int k = 0; k < stop; k++){
//deal with col_local in lane[k]
int col = __shfl(col_local, k);
//if(blockIdx.x==0 && threadIdx.x==0)
//
printf("iter=%d,k=%d,col=%d,stop=%d,anchor=%d\n", iter,k, col, stop,
anchor);
//this type of for is bad in performance
//for(int i = threadIdx.x; i < F; i += 64)
if(threadIdx.x<F/2)
thetaTemp[k*F/2 + threadIdx.x] =
__ldg(&thetaT[ F/2 * col + threadIdx.x]);
}
*/
            __syncthreads();
            ///*
            // tile: 10*10
            if (threadIdx.x < 55)
            {
                if (iter < iterations - 1)
                {
                    for (int k = 0; k < SCAN_BATCH; k++)
                        accumulate_in_registers();
                }
                else
                {
                    for (int k = 0; k < end - start - iter * SCAN_BATCH; k++)
                        accumulate_in_registers();
                }
            }
             __syncthreads();
            //*/
        }
        // end of iteration in copying from smem and aggregating in register
        __syncthreads();
#ifdef DEBUG
// if(threadIdx.x==0)
//	printf("***temp 0~9: %f %f %f %f %f %f %f %f %f %f\n", temp0, temp1,
// temp2, temp3, temp4, temp5, temp6, temp7, temp8, temp9);
#endif
        if (threadIdx.x < 55)
        {
            // weighted-lambda regularization
            if (tile_x == tile_y)
            {
                float temp = (end - start) * lambda;
                temp0 += temp;
                temp11 += temp;
                temp22 += temp;
                temp33 += temp;
                temp44 += temp;
                temp55 += temp;
                temp66 += temp;
                temp77 += temp;
                temp88 += temp;
                temp99 += temp;
            }
            // copy output to gmem
            int index = blockIdx.x * F * F / 2;
            // fill_lower_half_from_registers();
            fill_lower_half_from_registers_float2();
            // symmetric
            if (tile_x != tile_y)
            {
                // fill_upper_half_from_registers();
                fill_upper_half_from_registers_float2();
            }
        }
         __syncthreads();
    }
}

/*a generic kernel to get the hermitian matrices
 * as the left-hand side of the equations, to update X in ALS
 *examplary F = 100, T = 10
 */
__global__ void get_hermitianT10(const int batch_offset, float *tt,
                                 const int *csrRowIndex, const int *csrColIndex,
                                 const float lambda, const int m, const int F,
                                 const float *__restrict__ thetaT)
{
    extern __shared__ float2 thetaTemp[];
    int row = blockIdx.x + batch_offset;
    if (row < m)
    {
        // this block needs to handle end - start thetaT columns
        int start = csrRowIndex[row];
        int end = csrRowIndex[row + 1];
        // slide through [start, end] by window size SCAN_BATCH
        int iterations = (end - start - 1) / SCAN_BATCH + 1;
        float temp0 = 0, temp1 = 0, temp2 = 0, temp3 = 0, temp4 = 0, temp5 = 0,
              temp6 = 0, temp7 = 0, temp8 = 0, temp9 = 0;
        float temp10 = 0, temp11 = 0, temp12 = 0, temp13 = 0, temp14 = 0,
              temp15 = 0, temp16 = 0, temp17 = 0, temp18 = 0, temp19 = 0;
        float temp20 = 0, temp21 = 0, temp22 = 0, temp23 = 0, temp24 = 0,
              temp25 = 0, temp26 = 0, temp27 = 0, temp28 = 0, temp29 = 0;
        float temp30 = 0, temp31 = 0, temp32 = 0, temp33 = 0, temp34 = 0,
              temp35 = 0, temp36 = 0, temp37 = 0, temp38 = 0, temp39 = 0;
        float temp40 = 0, temp41 = 0, temp42 = 0, temp43 = 0, temp44 = 0,
              temp45 = 0, temp46 = 0, temp47 = 0, temp48 = 0, temp49 = 0;
        float temp50 = 0, temp51 = 0, temp52 = 0, temp53 = 0, temp54 = 0,
              temp55 = 0, temp56 = 0, temp57 = 0, temp58 = 0, temp59 = 0;
        float temp60 = 0, temp61 = 0, temp62 = 0, temp63 = 0, temp64 = 0,
              temp65 = 0, temp66 = 0, temp67 = 0, temp68 = 0, temp69 = 0;
        float temp70 = 0, temp71 = 0, temp72 = 0, temp73 = 0, temp74 = 0,
              temp75 = 0, temp76 = 0, temp77 = 0, temp78 = 0, temp79 = 0;
        float temp80 = 0, temp81 = 0, temp82 = 0, temp83 = 0, temp84 = 0,
              temp85 = 0, temp86 = 0, temp87 = 0, temp88 = 0, temp89 = 0;
        float temp90 = 0, temp91 = 0, temp92 = 0, temp93 = 0, temp94 = 0,
              temp95 = 0, temp96 = 0, temp97 = 0, temp98 = 0, temp99 = 0;

        int N = F / ALS_T10; // N = 100/10=10; for F = 100 and T = 10
        int effective_block_size = N * (N + 1) / 2;
        // get the x and y coordinate
        int tile_x = 0;
        int tile_y = 0;
        for (int i = 0; i < N; i++)
        {
            int end = ((2 * N - i) * (i + 1)) / 2;
            if (threadIdx.x < end)
            {
                tile_x = i * ALS_T10;
                tile_y = (N + threadIdx.x - end) * ALS_T10;
                break;
            }
        }
        int index = blockIdx.x * F * F;
        // iteration: copy gmem-->smem; aggregate smem-->register
        for (int iter = 0; iter < iterations; iter++)
        {
            // phase 1 in iteration: gmem --> smem

            // REQ: blockDim.x >= F/2
            if (threadIdx.x < F / 2)
            {
                for (int k = 0; k < SCAN_BATCH; k++)
                {
                    if (iter * SCAN_BATCH + k < end - start)
                    {
                        float2 theta;
                        theta.x =
                            __ldg(&thetaT[F * csrColIndex[start + iter * SCAN_BATCH + k] +
                                          2 * threadIdx.x]);
                        theta.y =
                            __ldg(&thetaT[F * csrColIndex[start + iter * SCAN_BATCH + k] +
                                          2 * threadIdx.x + 1]);
                        thetaTemp[k * F / 2 + threadIdx.x] = theta;
                        // this simpler statement is slower.
                        // thetaTemp[k * F/2 + threadIdx.x] = __ldg((float2*)&thetaT[F *
                        // csrColIndex[start + iter*SCAN_BATCH + k] + 2*threadIdx.x]);
                    }
                    // not enough theta to copy, set zero
                    else
                        memset(&thetaTemp[k * F / 2 + threadIdx.x], 0, 2 * sizeof(float));
                }
            }
            __syncthreads();

            // phase 2 in iteration: smem --> register
            if (threadIdx.x < effective_block_size)
            { // this redundant "if" seems
                // improving kernel performance
                for (int k = 0; k < SCAN_BATCH; k++)
                {
                    accumulate_in_registers();
                }
            }
            __syncthreads();
        }
        // end of iteration in copying from smem and aggregating in register
        __syncthreads();

        // phase 3, after iteration: register --> gmem
        if (threadIdx.x < effective_block_size)
        {
            fill_lower_half_from_registers();

            // symmetric
            if (tile_x != tile_y)
            {
                fill_upper_half_from_registers();
            }
            // regularization
            if (tile_x == tile_y)
            {
                for (int k = 0; k < ALS_T10; k++)
                    tt[index + (tile_x + k) * (1 + F)] += (end - start) * lambda;
            }
        }
    }
}

template <typename T>
class ALSFactorization
{
  public:
    ALSFactorization(const int m, const int n, const int f, const T lambda,
                     T *thetaTDevice, T *XTDevice)
        : m(m), n(n), f(f), lambda(lambda), thetaT(thetaTDevice), XT(XTDevice)
    {
        // initialize cublas, cusparse
        cublascall(cublasCreate(&handle));
        cusparsecall(cusparseCreate(&cushandle));
        cusparsecall(cusparseCreateMatDescr(&descr));
        cusparseSetMatType(descr, CUSPARSE_MATRIX_TYPE_GENERAL);
        cusparseSetMatIndexBase(descr, CUSPARSE_INDEX_BASE_ZERO);
    }
    ~ALSFactorization()
    {
        cublasDestroy(handle);
        cusparsecall(cusparseDestroy(cushandle));
        cusparsecall(cusparseDestroyMatDescr(descr));
    }

    void Iter(const int *csrRowIndex, const int *csrColIndex, const T *csrVal,
              const int *cscRowIndex, const int *cscColIndex, const T *cscVal,
              const long nnz, const int X_BATCH, const int THETA_BATCH)
    {
        // ---------------------------ALS iteration %d, update
        // X.----------------------------------
        float *tt = 0;
        float *ytheta = 0;
        float *ythetaT = 0;
        cudacall(cudaMalloc((void **)&ytheta, f * m * sizeof(ytheta[0])));
        cudacall(cudaMalloc((void **)&ythetaT, f * m * sizeof(ythetaT[0])));
        cudacall(cudaMemset(ytheta, 0, f * m * sizeof(*ytheta)));
        cudacall(cudaMemset(ythetaT, 0, f * m * sizeof(*ythetaT)));

        const float alpha = 1.0f;
        const float beta = 0.0f;
        // ytheta = R * (thetaT).T
        cusparsecall(csrmm(cushandle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                           CUSPARSE_OPERATION_TRANSPOSE, m, f, n, nnz, &alpha,
                           descr, csrVal, csrRowIndex, csrColIndex, thetaT, f,
                           &beta, ytheta, m));
        OK(cudaDeviceSynchronize());

        // printf("*******transpose ytheta use cublas.\n");
        // ytheta: m*f; need ythetaT = (ytheta).T = f*m
        cublascall(geam(handle, CUBLAS_OP_T, CUBLAS_OP_N, f, m, &alpha,
                        (const float *)ytheta, m, &beta, ythetaT, f, ythetaT, f));
        OK(cudaDeviceSynchronize());

        cudaCheckError();
        cudacall(cudaFree(ytheta));
        float *errors_train = 0;

        int block_dim = f / ALS_T10 * (f / ALS_T10 + 1) / 2;
        if (block_dim < f / 2)
            block_dim = f / 2;

        for (int batch_id = 0; batch_id < X_BATCH; batch_id++)
        {
            // #ifdef DEBUG
            // printf("*******batch %d / %d.*******\n", batch_id, X_BATCH);
            // #endif
            int batch_size = 0;
            if (batch_id != X_BATCH - 1)
                batch_size = m / X_BATCH;
            else
                batch_size = m - batch_id * (m / X_BATCH);
            int batch_offset = batch_id * (m / X_BATCH);
            cudacall(cudaMalloc((void **)&tt, f * f * batch_size * sizeof(float)));
            cudacall(cudaMemset(tt, 0, f * f * batch_size * sizeof(float)));
            if (f == 100)
                get_hermitian100<<<batch_size, 64,
                                   SCAN_BATCH * f / 2 * sizeof(float2)>>>(
                    batch_offset, (float2 *)tt, csrRowIndex, csrColIndex, lambda, m, f,
                    (float2 *)thetaT);
            else
                get_hermitianT10<<<batch_size, block_dim,
                                   SCAN_BATCH * f / 2 * sizeof(float2)>>>(
                    batch_offset, tt, csrRowIndex, csrColIndex, lambda, m, f, thetaT);
            OK(cudaDeviceSynchronize());

            cudaCheckError();
#ifdef USE_CG // use CG iterative solver
            updateXWithCGHost(tt, &XT[batch_offset * f], &ythetaT[batch_offset * f],
                              batch_size, f, CG_ITER);
#else //use LU solver instead \
      //host pointers for cublas batch operations
            float **devPtrTTHost = 0;
            cudacall(cudaMallocHost((void **)&devPtrTTHost, batch_size * sizeof(*devPtrTTHost)));
            float **devPtrYthetaTHost = 0;
            cudacall(cudaMallocHost((void **)&devPtrYthetaTHost, batch_size * sizeof(*devPtrYthetaTHost)));
            updateX(batch_size, batch_offset, ythetaT, tt, XT, handle, m, n, f, devPtrTTHost, devPtrYthetaTHost);
            cudacall(cudaFreeHost(devPtrTTHost));
            cudacall(cudaFreeHost(devPtrYthetaTHost));
#endif
            cudacall(cudaFree(tt));
        }
        cudacall(cudaFree(ythetaT));

        float *yTX = 0;
        float *yTXT = 0;
        cudacall(cudaMalloc((void **)&yTXT, f * n * sizeof(*yTXT)));
        cudacall(cudaMalloc((void **)&yTX, n * f * sizeof(*yTX)));
        cudacall(cudaMemset(yTXT, 0, f * n * sizeof(*yTXT)));
        cudacall(cudaMemset(yTX, 0, f * n * sizeof(*yTX)));

        cusparsecall(cusparseScsrmm2(cushandle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                                     CUSPARSE_OPERATION_TRANSPOSE, n, f, m, nnz,
                                     &alpha, descr, cscVal, cscColIndex,
                                     cscRowIndex, XT, f, &beta, yTX, n));
        // cudaDeviceSynchronize();
        // printf("*******transpose yTX \n");
        // yTX: n*f; need yTXT = (yTX).T = f*n
        cublascall(cublasSgeam(handle, CUBLAS_OP_T, CUBLAS_OP_N, f, n, &alpha,
                               (const float *)yTX, n, &beta, yTXT, f, yTXT, f));
        cudaDeviceSynchronize();
        cudacall(cudaFree(yTX));
        // in batches, when N is huge
        for (int batch_id = 0; batch_id < THETA_BATCH; batch_id++)
        {
            int batch_size = 0;
            if (batch_id != THETA_BATCH - 1)
                batch_size = n / THETA_BATCH;
            else
                batch_size = n - batch_id * (n / THETA_BATCH);
            int batch_offset = batch_id * (n / THETA_BATCH);

            float *xx = 0;
            cudacall(cudaMalloc((void **)&xx, f * f * batch_size * sizeof(xx[0])));
            cudacall(cudaMemset(xx, 0, f * f * batch_size * sizeof(float)));
            if (f == 100)
                get_hermitian100<<<batch_size, 64,
                                   SCAN_BATCH * f / 2 * sizeof(float2)>>>(
                    batch_offset, (float2 *)xx, cscColIndex, cscRowIndex, lambda, n, f,
                    (float2 *)XT);
            else
                get_hermitianT10<<<batch_size, block_dim,
                                   SCAN_BATCH * f * sizeof(float)>>>(
                    batch_offset, xx, cscColIndex, cscRowIndex, lambda, n, f, XT);
            OK(cudaDeviceSynchronize());
            cudaCheckError();

#ifdef USE_CG
            updateXWithCGHost(xx, &thetaT[batch_offset * f], &yTXT[batch_offset * f],
                              batch_size, f, CG_ITER);
#else
            float **devPtrXXHost = 0;
            cudacall(cudaMallocHost((void **)&devPtrXXHost, batch_size * sizeof(*devPtrXXHost)));
            float **devPtrYTXTHost = 0;
            cudacall(cudaMallocHost((void **)&devPtrYTXTHost, batch_size * sizeof(*devPtrYTXTHost)));
            updateTheta(batch_size, batch_offset, xx, yTXT, thetaT, handle, m, n, f,
                        devPtrXXHost, devPtrYTXTHost);
#endif
            cudacall(cudaFree(xx));
        }
        cudacall(cudaFree(yTXT));
    }
    T Score(const int *cooRowIndex, const int *cooColIndex, const T *cooData,
            const long nnz)
    {
        constexpr int error_size = 1000;
        T *errors_test = 0;
        cudacall(
            cudaMalloc((void **)&errors_test, error_size * sizeof(errors_test[0])));
        cudacall(cudaMemset(errors_test, 0, error_size * sizeof(float)));
        constexpr int block = 256;
        RMSE<<<(nnz + block - 1) / block, block>>>(cooData, cooRowIndex,
                                                   cooColIndex, thetaT, XT,
                                                   errors_test, nnz, error_size, f);
        OK(cudaDeviceSynchronize());
        cudaCheckError();

        T *rmse = (T *)malloc(sizeof(T));
        cublascall(cublasSasum(handle, error_size, errors_test, 1, rmse));
        cudaDeviceSynchronize();
        T final_rmse = sqrt((*rmse) / nnz);
        cudacall(cudaFree(errors_test));
        return final_rmse;
    }

  public:
    const int m;
    const int n;
    const int f;
    const T lambda;

  private:
    cublasHandle_t handle;
    cusparseHandle_t cushandle;
    cusparseMatDescr_t descr;
    T *thetaT;
    T *XT;
};

// fused kernel, use thetaT to update XT
__global__ void __launch_bounds__(64)
    alsUpdateFeature100(const int batch_offset, const int *csrRowIndex,
                        const int *csrColIndex, const float lambda, const int m,
                        const int F, const float *thetaT, float *XT,
                        float *ythetaT, int cgIter)
{
    extern __shared__ float2 thetaTemp[];
    int row = blockIdx.x + batch_offset;
    if (row < m)
    {
        // this block needs to handle end - start thetaT columns
        int start = csrRowIndex[row];
        int end = csrRowIndex[row + 1];
        // slide through [start, end] by window size SCAN_BATCH
        int iterations = (end - start - 1) / SCAN_BATCH + 1;
        float temp0 = 0, temp1 = 0, temp2 = 0, temp3 = 0, temp4 = 0, temp5 = 0,
              temp6 = 0, temp7 = 0, temp8 = 0, temp9 = 0;
        float temp10 = 0, temp11 = 0, temp12 = 0, temp13 = 0, temp14 = 0,
              temp15 = 0, temp16 = 0, temp17 = 0, temp18 = 0, temp19 = 0;
        float temp20 = 0, temp21 = 0, temp22 = 0, temp23 = 0, temp24 = 0,
              temp25 = 0, temp26 = 0, temp27 = 0, temp28 = 0, temp29 = 0;
        float temp30 = 0, temp31 = 0, temp32 = 0, temp33 = 0, temp34 = 0,
              temp35 = 0, temp36 = 0, temp37 = 0, temp38 = 0, temp39 = 0;
        float temp40 = 0, temp41 = 0, temp42 = 0, temp43 = 0, temp44 = 0,
              temp45 = 0, temp46 = 0, temp47 = 0, temp48 = 0, temp49 = 0;
        float temp50 = 0, temp51 = 0, temp52 = 0, temp53 = 0, temp54 = 0,
              temp55 = 0, temp56 = 0, temp57 = 0, temp58 = 0, temp59 = 0;
        float temp60 = 0, temp61 = 0, temp62 = 0, temp63 = 0, temp64 = 0,
              temp65 = 0, temp66 = 0, temp67 = 0, temp68 = 0, temp69 = 0;
        float temp70 = 0, temp71 = 0, temp72 = 0, temp73 = 0, temp74 = 0,
              temp75 = 0, temp76 = 0, temp77 = 0, temp78 = 0, temp79 = 0;
        float temp80 = 0, temp81 = 0, temp82 = 0, temp83 = 0, temp84 = 0,
              temp85 = 0, temp86 = 0, temp87 = 0, temp88 = 0, temp89 = 0;
        float temp90 = 0, temp91 = 0, temp92 = 0, temp93 = 0, temp94 = 0,
              temp95 = 0, temp96 = 0, temp97 = 0, temp98 = 0, temp99 = 0;

        int tile_x = 0;
        int tile_y = 0;

        int tile = F / 10;
        for (int i = 0; i < 10; i++)
        {
            int end = ((20 - i) * (i + 1)) / 2;
            if (threadIdx.x < end)
            {
                tile_x = i * tile;
                tile_y = (10 + threadIdx.x - end) * tile;
                // i    end    t_x     t_y
                // 0    10     0       0, 3, 6, 9, etc
                // 1    19     3
                break;
            }
        }
        // iteration: copy gmem-->smem; aggregate smem-->register
        for (int iter = 0; iter < iterations; iter++)
        {
            float2 theta;
            // copy texture --> smem, and sync

            // two layers: warp divergence unless we split at 32
            // require 32 >= SCAN_BATCH
            if (threadIdx.x < 2 * 32)
            {
                // int index = threadIdx.x;
                int index = threadIdx.x - (threadIdx.x / 32) * 32; // 0 to 31;
                if (index < SCAN_BATCH)
                {
                    if (iter * SCAN_BATCH + index < end - start)
                    {
                        // for (int k = 50*(threadIdx.x/32); k < 50*(threadIdx.x/32) + 50; k
                        // += 2){ IMPORTANT: for loop has constant and identical start and
                        // end
                        if (threadIdx.x < 32)
                        {
                            for (int k = 0; k < 50; k += 2)
                            {
                                theta.x = __ldg(
                                    &thetaT[F * csrColIndex[start + iter * SCAN_BATCH + index] +
                                            k]);
                                theta.y = __ldg(
                                    &thetaT[F * csrColIndex[start + iter * SCAN_BATCH + index] +
                                            k + 1]);
                                thetaTemp[index * F / 2 + k / 2] = theta;
                            }
                        }
                        else
                        {
                            for (int k = 0; k < 50; k += 2)
                            {
                                theta.x = __ldg(
                                    &thetaT[F * csrColIndex[start + iter * SCAN_BATCH + index] +
                                            k + 50]);
                                theta.y = __ldg(
                                    &thetaT[F * csrColIndex[start + iter * SCAN_BATCH + index] +
                                            k + 51]);
                                thetaTemp[index * F / 2 + k / 2 + 25] = theta;
                            }
                        }
                    }
                    // must be the last iteration; no need to check
                    // not enough theta to copy, set zero
                    else
                        memset(&thetaTemp[index * F / 2], 0, F * sizeof(float));
                }
            }
            __syncthreads();

            // tile: 10*10
            if (threadIdx.x < 55)
            {
                for (int k = 0; k < SCAN_BATCH; k++)
                {
                    accumulate_in_registers();
                }
            }
        }
        // end of iteration in copying from smem and aggregating in register
        __syncthreads();

#ifdef DEBUG
        if (blockIdx.x == 0 && threadIdx.x == 0)
        {
            printf("***temp 0~9: %f %f %f %f %f %f %f %f %f %f\n", temp0, temp1,
                   temp2, temp3, temp4, temp5, temp6, temp7, temp8, temp9);
        }
#endif

        // newly added CG phase
        // reuse the abundant shared memory
        float *sharedx = (float *)&thetaTemp[0];
        float *sharedp = (float *)&thetaTemp[50];
        float *sharedr = (float *)&thetaTemp[100];
        float *sharedap = (float *)&thetaTemp[150];
        float *sharedax = (float *)&thetaTemp[200];

        float *rsold = (float *)&thetaTemp[250];
        float *alpha = (float *)&thetaTemp[251];
        float *rsnew = (float *)&thetaTemp[252];
        float *beta = (float *)&thetaTemp[253];
        // sharedx<--x
        for (int k = threadIdx.x; k < F; k += 64)
        {
            sharedx[k] = XT[blockIdx.x * F + k];
            sharedax[k] = 0;
        }
        __syncthreads();
        float temp = 0;
        // only uses 55 threads for A*p and A*x
        if (threadIdx.x < 55)
        {
            // add regularization
            if (tile_x == tile_y)
            {
                temp = (end - start) * lambda;
                temp0 += temp;
                temp11 += temp;
                temp22 += temp;
                temp33 += temp;
                temp44 += temp;
                temp55 += temp;
                temp66 += temp;
                temp77 += temp;
                temp88 += temp;
                temp99 += temp;
            }
#ifdef DEBUG
            if (blockIdx.x == 0 && threadIdx.x == 0)
            {
                printf("***temp 0~9: %f %f %f %f %f %f %f %f %f %f\n", temp0, temp1,
                       temp2, temp3, temp4, temp5, temp6, temp7, temp8, temp9);
            }
#endif
            // r=b-A*x;
            // step1: ax=A*x

            atomicAdd(
                &sharedax[tile_y],
                temp0 * sharedx[tile_x] + temp10 * sharedx[tile_x + 1] +
                    temp20 * sharedx[tile_x + 2] + temp30 * sharedx[tile_x + 3] +
                    temp40 * sharedx[tile_x + 4] + temp50 * sharedx[tile_x + 5] +
                    temp60 * sharedx[tile_x + 6] + temp70 * sharedx[tile_x + 7] +
                    temp80 * sharedx[tile_x + 8] + temp90 * sharedx[tile_x + 9]);

            atomicAdd(
                &sharedax[tile_y + 1],
                temp1 * sharedx[tile_x] + temp11 * sharedx[tile_x + 1] +
                    temp21 * sharedx[tile_x + 2] + temp31 * sharedx[tile_x + 3] +
                    temp41 * sharedx[tile_x + 4] + temp51 * sharedx[tile_x + 5] +
                    temp61 * sharedx[tile_x + 6] + temp71 * sharedx[tile_x + 7] +
                    temp81 * sharedx[tile_x + 8] + temp91 * sharedx[tile_x + 9]);
            atomicAdd(
                &sharedax[tile_y + 2],
                temp2 * sharedx[tile_x] + temp12 * sharedx[tile_x + 1] +
                    temp22 * sharedx[tile_x + 2] + temp32 * sharedx[tile_x + 3] +
                    temp42 * sharedx[tile_x + 4] + temp52 * sharedx[tile_x + 5] +
                    temp62 * sharedx[tile_x + 6] + temp72 * sharedx[tile_x + 7] +
                    temp82 * sharedx[tile_x + 8] + temp92 * sharedx[tile_x + 9]);
            atomicAdd(
                &sharedax[tile_y + 3],
                temp3 * sharedx[tile_x] + temp13 * sharedx[tile_x + 1] +
                    temp23 * sharedx[tile_x + 2] + temp33 * sharedx[tile_x + 3] +
                    temp43 * sharedx[tile_x + 4] + temp53 * sharedx[tile_x + 5] +
                    temp63 * sharedx[tile_x + 6] + temp73 * sharedx[tile_x + 7] +
                    temp83 * sharedx[tile_x + 8] + temp93 * sharedx[tile_x + 9]);
            atomicAdd(
                &sharedax[tile_y + 4],
                temp4 * sharedx[tile_x] + temp14 * sharedx[tile_x + 1] +
                    temp24 * sharedx[tile_x + 2] + temp34 * sharedx[tile_x + 3] +
                    temp44 * sharedx[tile_x + 4] + temp54 * sharedx[tile_x + 5] +
                    temp64 * sharedx[tile_x + 6] + temp74 * sharedx[tile_x + 7] +
                    temp84 * sharedx[tile_x + 8] + temp94 * sharedx[tile_x + 9]);
            atomicAdd(
                &sharedax[tile_y + 5],
                temp5 * sharedx[tile_x] + temp15 * sharedx[tile_x + 1] +
                    temp25 * sharedx[tile_x + 2] + temp35 * sharedx[tile_x + 3] +
                    temp45 * sharedx[tile_x + 4] + temp55 * sharedx[tile_x + 5] +
                    temp65 * sharedx[tile_x + 6] + temp75 * sharedx[tile_x + 7] +
                    temp85 * sharedx[tile_x + 8] + temp95 * sharedx[tile_x + 9]);
            atomicAdd(
                &sharedax[tile_y + 6],
                temp6 * sharedx[tile_x] + temp16 * sharedx[tile_x + 1] +
                    temp26 * sharedx[tile_x + 2] + temp36 * sharedx[tile_x + 3] +
                    temp46 * sharedx[tile_x + 4] + temp56 * sharedx[tile_x + 5] +
                    temp66 * sharedx[tile_x + 6] + temp76 * sharedx[tile_x + 7] +
                    temp86 * sharedx[tile_x + 8] + temp96 * sharedx[tile_x + 9]);
            atomicAdd(
                &sharedax[tile_y + 7],
                temp7 * sharedx[tile_x] + temp17 * sharedx[tile_x + 1] +
                    temp27 * sharedx[tile_x + 2] + temp37 * sharedx[tile_x + 3] +
                    temp47 * sharedx[tile_x + 4] + temp57 * sharedx[tile_x + 5] +
                    temp67 * sharedx[tile_x + 6] + temp77 * sharedx[tile_x + 7] +
                    temp87 * sharedx[tile_x + 8] + temp97 * sharedx[tile_x + 9]);
            atomicAdd(
                &sharedax[tile_y + 8],
                temp8 * sharedx[tile_x] + temp18 * sharedx[tile_x + 1] +
                    temp28 * sharedx[tile_x + 2] + temp38 * sharedx[tile_x + 3] +
                    temp48 * sharedx[tile_x + 4] + temp58 * sharedx[tile_x + 5] +
                    temp68 * sharedx[tile_x + 6] + temp78 * sharedx[tile_x + 7] +
                    temp88 * sharedx[tile_x + 8] + temp98 * sharedx[tile_x + 9]);
            atomicAdd(
                &sharedax[tile_y + 9],
                temp9 * sharedx[tile_x] + temp19 * sharedx[tile_x + 1] +
                    temp29 * sharedx[tile_x + 2] + temp39 * sharedx[tile_x + 3] +
                    temp49 * sharedx[tile_x + 4] + temp59 * sharedx[tile_x + 5] +
                    temp69 * sharedx[tile_x + 6] + temp79 * sharedx[tile_x + 7] +
                    temp89 * sharedx[tile_x + 8] + temp99 * sharedx[tile_x + 9]);

            if (tile_x != tile_y)
            {
                atomicAdd(
                    &sharedax[tile_x],
                    temp0 * sharedx[tile_y] + temp1 * sharedx[tile_y + 1] +
                        temp2 * sharedx[tile_y + 2] + temp3 * sharedx[tile_y + 3] +
                        temp4 * sharedx[tile_y + 4] + temp5 * sharedx[tile_y + 5] +
                        temp6 * sharedx[tile_y + 6] + temp7 * sharedx[tile_y + 7] +
                        temp8 * sharedx[tile_y + 8] + temp9 * sharedx[tile_y + 9]);

                atomicAdd(
                    &sharedax[tile_x + 1],
                    temp10 * sharedx[tile_y] + temp11 * sharedx[tile_y + 1] +
                        temp12 * sharedx[tile_y + 2] + temp13 * sharedx[tile_y + 3] +
                        temp14 * sharedx[tile_y + 4] + temp15 * sharedx[tile_y + 5] +
                        temp16 * sharedx[tile_y + 6] + temp17 * sharedx[tile_y + 7] +
                        temp18 * sharedx[tile_y + 8] + temp19 * sharedx[tile_y + 9]);
                atomicAdd(
                    &sharedax[tile_x + 2],
                    temp20 * sharedx[tile_y] + temp21 * sharedx[tile_y + 1] +
                        temp22 * sharedx[tile_y + 2] + temp23 * sharedx[tile_y + 3] +
                        temp24 * sharedx[tile_y + 4] + temp25 * sharedx[tile_y + 5] +
                        temp26 * sharedx[tile_y + 6] + temp27 * sharedx[tile_y + 7] +
                        temp28 * sharedx[tile_y + 8] + temp29 * sharedx[tile_y + 9]);
                atomicAdd(
                    &sharedax[tile_x + 3],
                    temp30 * sharedx[tile_y] + temp31 * sharedx[tile_y + 1] +
                        temp32 * sharedx[tile_y + 2] + temp33 * sharedx[tile_y + 3] +
                        temp34 * sharedx[tile_y + 4] + temp35 * sharedx[tile_y + 5] +
                        temp36 * sharedx[tile_y + 6] + temp37 * sharedx[tile_y + 7] +
                        temp38 * sharedx[tile_y + 8] + temp39 * sharedx[tile_y + 9]);
                atomicAdd(
                    &sharedax[tile_x + 4],
                    temp40 * sharedx[tile_y] + temp41 * sharedx[tile_y + 1] +
                        temp42 * sharedx[tile_y + 2] + temp43 * sharedx[tile_y + 3] +
                        temp44 * sharedx[tile_y + 4] + temp45 * sharedx[tile_y + 5] +
                        temp46 * sharedx[tile_y + 6] + temp47 * sharedx[tile_y + 7] +
                        temp48 * sharedx[tile_y + 8] + temp49 * sharedx[tile_y + 9]);
                atomicAdd(
                    &sharedax[tile_x + 5],
                    temp50 * sharedx[tile_y] + temp51 * sharedx[tile_y + 1] +
                        temp52 * sharedx[tile_y + 2] + temp53 * sharedx[tile_y + 3] +
                        temp54 * sharedx[tile_y + 4] + temp55 * sharedx[tile_y + 5] +
                        temp56 * sharedx[tile_y + 6] + temp57 * sharedx[tile_y + 7] +
                        temp58 * sharedx[tile_y + 8] + temp59 * sharedx[tile_y + 9]);
                atomicAdd(
                    &sharedax[tile_x + 6],
                    temp60 * sharedx[tile_y] + temp61 * sharedx[tile_y + 1] +
                        temp62 * sharedx[tile_y + 2] + temp63 * sharedx[tile_y + 3] +
                        temp64 * sharedx[tile_y + 4] + temp65 * sharedx[tile_y + 5] +
                        temp66 * sharedx[tile_y + 6] + temp67 * sharedx[tile_y + 7] +
                        temp68 * sharedx[tile_y + 8] + temp69 * sharedx[tile_y + 9]);
                atomicAdd(
                    &sharedax[tile_x + 7],
                    temp70 * sharedx[tile_y] + temp71 * sharedx[tile_y + 1] +
                        temp72 * sharedx[tile_y + 2] + temp73 * sharedx[tile_y + 3] +
                        temp74 * sharedx[tile_y + 4] + temp75 * sharedx[tile_y + 5] +
                        temp76 * sharedx[tile_y + 6] + temp77 * sharedx[tile_y + 7] +
                        temp78 * sharedx[tile_y + 8] + temp79 * sharedx[tile_y + 9]);
                atomicAdd(
                    &sharedax[tile_x + 8],
                    temp80 * sharedx[tile_y] + temp81 * sharedx[tile_y + 1] +
                        temp82 * sharedx[tile_y + 2] + temp83 * sharedx[tile_y + 3] +
                        temp84 * sharedx[tile_y + 4] + temp85 * sharedx[tile_y + 5] +
                        temp86 * sharedx[tile_y + 6] + temp87 * sharedx[tile_y + 7] +
                        temp88 * sharedx[tile_y + 8] + temp89 * sharedx[tile_y + 9]);
                atomicAdd(
                    &sharedax[tile_x + 9],
                    temp90 * sharedx[tile_y] + temp91 * sharedx[tile_y + 1] +
                        temp92 * sharedx[tile_y + 2] + temp93 * sharedx[tile_y + 3] +
                        temp94 * sharedx[tile_y + 4] + temp95 * sharedx[tile_y + 5] +
                        temp96 * sharedx[tile_y + 6] + temp97 * sharedx[tile_y + 7] +
                        temp98 * sharedx[tile_y + 8] + temp99 * sharedx[tile_y + 9]);
            }
        }
        __syncthreads();

#ifdef DEBUG
        if (blockIdx.x == 0 && threadIdx.x == 0)
        {
            printf("***x:\n");
            for (int i = 0; i < 100; i++)
                printf("%f ", sharedx[i]);
            printf("\n\n");
            printf("***r=Ax:\n");
            for (int i = 0; i < 100; i++)
                printf("%f ", sharedax[i]);
            printf("\n\n");
        }
#endif
        for (int k = threadIdx.x; k < F; k += 64)
        {
            // r=b-Ax
            sharedr[k] = ythetaT[blockIdx.x * blockDim.x + k] - sharedax[k];
            // p=r;
            sharedp[k] = sharedr[k];
        }
        // rsold=r'*r;
        if (threadIdx.x == 0)
        {
            rsold[0] = 0;
        }
        for (int k = threadIdx.x; k < F; k += 64)
        {
            temp += sharedr[k] * sharedr[k];
        }
        blockReduceSumWithAtomics(rsold, temp);
        __syncthreads();
#ifdef DEBUG
        if (blockIdx.x == 0 && threadIdx.x == 0)
        {
            printf("***rsold:\n");
            printf("rsold = %f \n", rsold[0]);
            printf("***shared memory content after 1st blockReduceSum:\n");
            for (int i = 0; i < 100; i++)
                printf("%f ", sharedx[i]);
            printf("\n");
            for (int i = 0; i < 100; i++)
                printf("%f ", sharedax[i]);
            printf("\n\n");

            for (int i = 0; i < 100; i++)
                printf("%f ", sharedp[i]);
            printf("\n");
            for (int i = 0; i < 100; i++)
                printf("%f ", sharedr[i]);
            printf("\n");
            for (int i = 0; i < 100; i++)
                printf("%f ", sharedap[i]);
            printf("\n");
        }
#endif
        ///*
        // CG iterations
        for (int iter = 0; iter < cgIter; iter++)
        {
            // ap=A*p;
            for (int k = threadIdx.x; k < F; k += 64)
                sharedap[k] = 0;
            __syncthreads();
            // only uses 55 threads for A*p and A*x
            if (threadIdx.x < 55)
            {
                atomicAdd(
                    &sharedap[tile_y],
                    temp0 * sharedp[tile_x] + temp10 * sharedp[tile_x + 1] +
                        temp20 * sharedp[tile_x + 2] + temp30 * sharedp[tile_x + 3] +
                        temp40 * sharedp[tile_x + 4] + temp50 * sharedp[tile_x + 5] +
                        temp60 * sharedp[tile_x + 6] + temp70 * sharedp[tile_x + 7] +
                        temp80 * sharedp[tile_x + 8] + temp90 * sharedp[tile_x + 9]);

                atomicAdd(
                    &sharedap[tile_y + 1],
                    temp1 * sharedp[tile_x] + temp11 * sharedp[tile_x + 1] +
                        temp21 * sharedp[tile_x + 2] + temp31 * sharedp[tile_x + 3] +
                        temp41 * sharedp[tile_x + 4] + temp51 * sharedp[tile_x + 5] +
                        temp61 * sharedp[tile_x + 6] + temp71 * sharedp[tile_x + 7] +
                        temp81 * sharedp[tile_x + 8] + temp91 * sharedp[tile_x + 9]);
                atomicAdd(
                    &sharedap[tile_y + 2],
                    temp2 * sharedp[tile_x] + temp12 * sharedp[tile_x + 1] +
                        temp22 * sharedp[tile_x + 2] + temp32 * sharedp[tile_x + 3] +
                        temp42 * sharedp[tile_x + 4] + temp52 * sharedp[tile_x + 5] +
                        temp62 * sharedp[tile_x + 6] + temp72 * sharedp[tile_x + 7] +
                        temp82 * sharedp[tile_x + 8] + temp92 * sharedp[tile_x + 9]);
                atomicAdd(
                    &sharedap[tile_y + 3],
                    temp3 * sharedp[tile_x] + temp13 * sharedp[tile_x + 1] +
                        temp23 * sharedp[tile_x + 2] + temp33 * sharedp[tile_x + 3] +
                        temp43 * sharedp[tile_x + 4] + temp53 * sharedp[tile_x + 5] +
                        temp63 * sharedp[tile_x + 6] + temp73 * sharedp[tile_x + 7] +
                        temp83 * sharedp[tile_x + 8] + temp93 * sharedp[tile_x + 9]);
                atomicAdd(
                    &sharedap[tile_y + 4],
                    temp4 * sharedp[tile_x] + temp14 * sharedp[tile_x + 1] +
                        temp24 * sharedp[tile_x + 2] + temp34 * sharedp[tile_x + 3] +
                        temp44 * sharedp[tile_x + 4] + temp54 * sharedp[tile_x + 5] +
                        temp64 * sharedp[tile_x + 6] + temp74 * sharedp[tile_x + 7] +
                        temp84 * sharedp[tile_x + 8] + temp94 * sharedp[tile_x + 9]);
                atomicAdd(
                    &sharedap[tile_y + 5],
                    temp5 * sharedp[tile_x] + temp15 * sharedp[tile_x + 1] +
                        temp25 * sharedp[tile_x + 2] + temp35 * sharedp[tile_x + 3] +
                        temp45 * sharedp[tile_x + 4] + temp55 * sharedp[tile_x + 5] +
                        temp65 * sharedp[tile_x + 6] + temp75 * sharedp[tile_x + 7] +
                        temp85 * sharedp[tile_x + 8] + temp95 * sharedp[tile_x + 9]);
                atomicAdd(
                    &sharedap[tile_y + 6],
                    temp6 * sharedp[tile_x] + temp16 * sharedp[tile_x + 1] +
                        temp26 * sharedp[tile_x + 2] + temp36 * sharedp[tile_x + 3] +
                        temp46 * sharedp[tile_x + 4] + temp56 * sharedp[tile_x + 5] +
                        temp66 * sharedp[tile_x + 6] + temp76 * sharedp[tile_x + 7] +
                        temp86 * sharedp[tile_x + 8] + temp96 * sharedp[tile_x + 9]);
                atomicAdd(
                    &sharedap[tile_y + 7],
                    temp7 * sharedp[tile_x] + temp17 * sharedp[tile_x + 1] +
                        temp27 * sharedp[tile_x + 2] + temp37 * sharedp[tile_x + 3] +
                        temp47 * sharedp[tile_x + 4] + temp57 * sharedp[tile_x + 5] +
                        temp67 * sharedp[tile_x + 6] + temp77 * sharedp[tile_x + 7] +
                        temp87 * sharedp[tile_x + 8] + temp97 * sharedp[tile_x + 9]);
                atomicAdd(
                    &sharedap[tile_y + 8],
                    temp8 * sharedp[tile_x] + temp18 * sharedp[tile_x + 1] +
                        temp28 * sharedp[tile_x + 2] + temp38 * sharedp[tile_x + 3] +
                        temp48 * sharedp[tile_x + 4] + temp58 * sharedp[tile_x + 5] +
                        temp68 * sharedp[tile_x + 6] + temp78 * sharedp[tile_x + 7] +
                        temp88 * sharedp[tile_x + 8] + temp98 * sharedp[tile_x + 9]);
                atomicAdd(
                    &sharedap[tile_y + 9],
                    temp9 * sharedp[tile_x] + temp19 * sharedp[tile_x + 1] +
                        temp29 * sharedp[tile_x + 2] + temp39 * sharedp[tile_x + 3] +
                        temp49 * sharedp[tile_x + 4] + temp59 * sharedp[tile_x + 5] +
                        temp69 * sharedp[tile_x + 6] + temp79 * sharedp[tile_x + 7] +
                        temp89 * sharedp[tile_x + 8] + temp99 * sharedp[tile_x + 9]);

                if (tile_x != tile_y)
                {
                    atomicAdd(
                        &sharedap[tile_x],
                        temp0 * sharedp[tile_y] + temp1 * sharedp[tile_y + 1] +
                            temp2 * sharedp[tile_y + 2] + temp3 * sharedp[tile_y + 3] +
                            temp4 * sharedp[tile_y + 4] + temp5 * sharedp[tile_y + 5] +
                            temp6 * sharedp[tile_y + 6] + temp7 * sharedp[tile_y + 7] +
                            temp8 * sharedp[tile_y + 8] + temp9 * sharedp[tile_y + 9]);

                    atomicAdd(
                        &sharedap[tile_x + 1],
                        temp10 * sharedp[tile_y] + temp11 * sharedp[tile_y + 1] +
                            temp12 * sharedp[tile_y + 2] + temp13 * sharedp[tile_y + 3] +
                            temp14 * sharedp[tile_y + 4] + temp15 * sharedp[tile_y + 5] +
                            temp16 * sharedp[tile_y + 6] + temp17 * sharedp[tile_y + 7] +
                            temp18 * sharedp[tile_y + 8] + temp19 * sharedp[tile_y + 9]);
                    atomicAdd(
                        &sharedap[tile_x + 2],
                        temp20 * sharedp[tile_y] + temp21 * sharedp[tile_y + 1] +
                            temp22 * sharedp[tile_y + 2] + temp23 * sharedp[tile_y + 3] +
                            temp24 * sharedp[tile_y + 4] + temp25 * sharedp[tile_y + 5] +
                            temp26 * sharedp[tile_y + 6] + temp27 * sharedp[tile_y + 7] +
                            temp28 * sharedp[tile_y + 8] + temp29 * sharedp[tile_y + 9]);
                    atomicAdd(
                        &sharedap[tile_x + 3],
                        temp30 * sharedp[tile_y] + temp31 * sharedp[tile_y + 1] +
                            temp32 * sharedp[tile_y + 2] + temp33 * sharedp[tile_y + 3] +
                            temp34 * sharedp[tile_y + 4] + temp35 * sharedp[tile_y + 5] +
                            temp36 * sharedp[tile_y + 6] + temp37 * sharedp[tile_y + 7] +
                            temp38 * sharedp[tile_y + 8] + temp39 * sharedp[tile_y + 9]);
                    atomicAdd(
                        &sharedap[tile_x + 4],
                        temp40 * sharedp[tile_y] + temp41 * sharedp[tile_y + 1] +
                            temp42 * sharedp[tile_y + 2] + temp43 * sharedp[tile_y + 3] +
                            temp44 * sharedp[tile_y + 4] + temp45 * sharedp[tile_y + 5] +
                            temp46 * sharedp[tile_y + 6] + temp47 * sharedp[tile_y + 7] +
                            temp48 * sharedp[tile_y + 8] + temp49 * sharedp[tile_y + 9]);
                    atomicAdd(
                        &sharedap[tile_x + 5],
                        temp50 * sharedp[tile_y] + temp51 * sharedp[tile_y + 1] +
                            temp52 * sharedp[tile_y + 2] + temp53 * sharedp[tile_y + 3] +
                            temp54 * sharedp[tile_y + 4] + temp55 * sharedp[tile_y + 5] +
                            temp56 * sharedp[tile_y + 6] + temp57 * sharedp[tile_y + 7] +
                            temp58 * sharedp[tile_y + 8] + temp59 * sharedp[tile_y + 9]);
                    atomicAdd(
                        &sharedap[tile_x + 6],
                        temp60 * sharedp[tile_y] + temp61 * sharedp[tile_y + 1] +
                            temp62 * sharedp[tile_y + 2] + temp63 * sharedp[tile_y + 3] +
                            temp64 * sharedp[tile_y + 4] + temp65 * sharedp[tile_y + 5] +
                            temp66 * sharedp[tile_y + 6] + temp67 * sharedp[tile_y + 7] +
                            temp68 * sharedp[tile_y + 8] + temp69 * sharedp[tile_y + 9]);
                    atomicAdd(
                        &sharedap[tile_x + 7],
                        temp70 * sharedp[tile_y] + temp71 * sharedp[tile_y + 1] +
                            temp72 * sharedp[tile_y + 2] + temp73 * sharedp[tile_y + 3] +
                            temp74 * sharedp[tile_y + 4] + temp75 * sharedp[tile_y + 5] +
                            temp76 * sharedp[tile_y + 6] + temp77 * sharedp[tile_y + 7] +
                            temp78 * sharedp[tile_y + 8] + temp79 * sharedp[tile_y + 9]);
                    atomicAdd(
                        &sharedap[tile_x + 8],
                        temp80 * sharedp[tile_y] + temp81 * sharedp[tile_y + 1] +
                            temp82 * sharedp[tile_y + 2] + temp83 * sharedp[tile_y + 3] +
                            temp84 * sharedp[tile_y + 4] + temp85 * sharedp[tile_y + 5] +
                            temp86 * sharedp[tile_y + 6] + temp87 * sharedp[tile_y + 7] +
                            temp88 * sharedp[tile_y + 8] + temp89 * sharedp[tile_y + 9]);
                    atomicAdd(
                        &sharedap[tile_x + 9],
                        temp90 * sharedp[tile_y] + temp91 * sharedp[tile_y + 1] +
                            temp92 * sharedp[tile_y + 2] + temp93 * sharedp[tile_y + 3] +
                            temp94 * sharedp[tile_y + 4] + temp95 * sharedp[tile_y + 5] +
                            temp96 * sharedp[tile_y + 6] + temp97 * sharedp[tile_y + 7] +
                            temp98 * sharedp[tile_y + 8] + temp99 * sharedp[tile_y + 9]);
                }
            }
            __syncthreads();
#ifdef DEBUG
            if (blockIdx.x == 0 && threadIdx.x == 0)
            {
                printf("----------CG iteration %d \n", iter);
                printf("***ap:\n");
                for (int i = 0; i < F; i++)
                    printf("%f ", sharedap[i]);
                printf("\n\n");
                printf("***shared memory content before 2rd blockReduceSum:\n");
                for (int i = 0; i < F; i++)
                    printf("%f ", sharedp[i]);
                printf("\n\n");
                for (int i = 0; i < F; i++)
                    printf("%f ", sharedr[i]);
                printf("\n\n");
                for (int i = 0; i < F; i++)
                    printf("%f ", sharedap[i]);
                printf("\n\n");
            }
#endif
            if (threadIdx.x == 0)
            {
                rsnew[0] = 0;
            }
            // no need to have sync before blockReduce
            // because there is a __syncthreads() in blockReduce
            // pAp=p'*Ap
            temp = 0;
            for (int k = threadIdx.x; k < F; k += 64)
                temp += sharedp[k] * sharedap[k];
            // temp = blockReduceSum(shared, temp);
            blockReduceSumWithAtomics(rsnew, temp);
            // sync needed, to let all atomicAdd threads completes
            __syncthreads();
            if (threadIdx.x == 0)
            {
                // pAp = temp;
                // alpha=rsold/(p'*Ap); use rsnew to store pAp
                alpha[0] = rsold[0] / rsnew[0];
#ifdef DEBUG
                if (blockIdx.x == 0)
                {
                    printf("***rsold:\n");
                    printf("rsold = %f \n", rsold[0]);
                    printf("***pAp:\n");
                    printf("pAp = %f \n", rsnew[0]);
                    printf("***alpha:\n");
                    printf("alpha = %f \n", alpha[0]);
                }
#endif
                rsnew[0] = 0;
            }
            // needed, aplpha[0] to be used by all threads
            __syncthreads();
            for (int k = threadIdx.x; k < F; k += 64)
            {
                // x=x+alpha*p;
                sharedx[k] = sharedx[k] + alpha[0] * sharedp[k];
                // r=r-alpha*Ap;
                sharedr[k] = sharedr[k] - alpha[0] * sharedap[k];
                // NOT needed?
                //__syncthreads();
            }
            __syncthreads();
#ifdef DEBUG
            if (blockIdx.x == 0 && threadIdx.x == 0)
            {
                printf("***shared memory content before 3rd blockReduceSum:\n");
                for (int i = 0; i < F; i++)
                    printf("%f ", sharedp[i]);
                printf("\n");
                for (int i = 0; i < F; i++)
                    printf("%f ", sharedr[i]);
                printf("\n");
                for (int i = 0; i < F; i++)
                    printf("%f ", sharedap[i]);
                printf("\n");
            }
#endif
            // rsnew=r'*r;
            temp = 0;
            for (int k = threadIdx.x; k < F; k += 64)
                temp += sharedr[k] * sharedr[k];
            blockReduceSumWithAtomics(rsnew, temp);
            // WARN: has to have this sync!
            __syncthreads();

#ifdef DEBUG
            if (blockIdx.x == 0 && threadIdx.x == 0)
            {
                printf("***rsnew:\n");
                printf("rsnew = %f \n", rsnew[0]);
                printf("***shared memory content after 3rd blockReduceSum:\n");
                for (int i = 0; i < F; i++)
                    printf("%f ", sharedp[i]);
                printf("\n");
                for (int i = 0; i < F; i++)
                    printf("%f ", sharedr[i]);
                printf("\n");
                for (int i = 0; i < F; i++)
                    printf("%f ", sharedap[i]);
                printf("\n");
            }
#endif
            if (rsnew[0] < CG_ERROR)
                break;
            // NOT needed?
            //__syncthreads();
            // beta
            if (threadIdx.x == 0)
            {
                beta[0] = rsnew[0] / rsold[0];
                // rsold=rsnew;
                rsold[0] = rsnew[0];
            }
            // need sync since every thread needs beta[0]
            __syncthreads();
            // p=r+(rsnew/rsold)*p;
            for (int k = threadIdx.x; k < F; k += 64)
                sharedp[k] = sharedr[k] + beta[0] * sharedp[k];
            // need sync as every thread needs sharedp at the beginning of for
            __syncthreads();
#ifdef DEBUX_BATCH, THETA_BATCHG
            __syncthreads();
            if (blockIdx.x == 0 && threadIdx.x == 0)
            {
                printf("***shared memory content after update p:\n");
                for (int i = 0; i < F; i++)
                    printf("%f ", sharedp[i]);
                printf("\n");
                for (int i = 0; i < F; i++)
                    printf("%f ", sharedr[i]);
                printf("\n");
                for (int i = 0; i < F; i++)
                    printf("%f ", sharedap[i]);
                printf("\n");
            }
            __syncthreads();
#endif
        } // end of CG iterations
        // x<--sharedx
        for (int k = threadIdx.x; k < F; k += 64)
            XT[blockIdx.x * F + k] = sharedx[k];
        //*/
    }
}
void alsUpdateFeature100Host(const int batch_offset, const int *csrRowIndex,
                             const int *csrColIndex, const float lambda,
                             const int m, const int F,
                             const float *__restrict__ thetaT, float *XT,
                             float *ythetaT, int cgIter)
{
    alsUpdateFeature100<<<m, 64, SCAN_BATCH * F / 2 * sizeof(float2)>>>(
        batch_offset, csrRowIndex, csrColIndex, lambda, m, F, thetaT, XT, ythetaT,
        cgIter);
    cudaDeviceSynchronize();
    cudaCheckError();
}

#endif /* ALS_H_ */
