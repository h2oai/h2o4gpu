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
 *      Author: weitan
 */

#ifndef ALS_H_
#define ALS_H_

#if GOOGLE_CUDA
#define EIGEN_USE_GPU
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#endif
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cusparse.h>
#include <host_defines.h>
//these parameters do not change among different problem size
//our kernels handle the case where F%T==0 and F = 100
#define T10 10

#define accumulate_in_registers()\
	do\
    {\
	temp0 += thetaTemp[tile_x/2 + k*F/2].x * thetaTemp[tile_y/2 + k*F/2].x;\
	temp1 += thetaTemp[tile_x/2 + k*F/2].x * thetaTemp[tile_y/2 + k*F/2].y;\
	temp2 += thetaTemp[tile_x/2 + k*F/2].x * thetaTemp[tile_y/2 +1 + k*F/2].x;\
	temp3 += thetaTemp[tile_x/2 + k*F/2].x * thetaTemp[tile_y/2 +1 + k*F/2].y;\
	temp4 += thetaTemp[tile_x/2 + k*F/2].x * thetaTemp[tile_y/2 +2 + k*F/2].x;\
	temp5 += thetaTemp[tile_x/2 + k*F/2].x * thetaTemp[tile_y/2 +2 + k*F/2].y;\
	temp6 += thetaTemp[tile_x/2 + k*F/2].x * thetaTemp[tile_y/2 +3 + k*F/2].x;\
	temp7 += thetaTemp[tile_x/2 + k*F/2].x * thetaTemp[tile_y/2 +3 + k*F/2].y;\
	temp8 += thetaTemp[tile_x/2 + k*F/2].x * thetaTemp[tile_y/2 +4 + k*F/2].x;\
	temp9 += thetaTemp[tile_x/2 + k*F/2].x * thetaTemp[tile_y/2 +4 + k*F/2].y;\
	temp10 += thetaTemp[tile_x/2 + k*F/2].y * thetaTemp[tile_y/2 + k*F/2].x;\
	temp11 += thetaTemp[tile_x/2 + k*F/2].y * thetaTemp[tile_y/2 + k*F/2].y;\
	temp12 += thetaTemp[tile_x/2 + k*F/2].y * thetaTemp[tile_y/2 +1 + k*F/2].x;\
	temp13 += thetaTemp[tile_x/2 + k*F/2].y * thetaTemp[tile_y/2 +1 + k*F/2].y;\
	temp14 += thetaTemp[tile_x/2 + k*F/2].y * thetaTemp[tile_y/2 +2 + k*F/2].x;\
	temp15 += thetaTemp[tile_x/2 + k*F/2].y * thetaTemp[tile_y/2 +2 + k*F/2].y;\
	temp16 += thetaTemp[tile_x/2 + k*F/2].y * thetaTemp[tile_y/2 +3 + k*F/2].x;\
	temp17 += thetaTemp[tile_x/2 + k*F/2].y * thetaTemp[tile_y/2 +3 + k*F/2].y;\
	temp18 += thetaTemp[tile_x/2 + k*F/2].y * thetaTemp[tile_y/2 +4 + k*F/2].x;\
	temp19 += thetaTemp[tile_x/2 + k*F/2].y * thetaTemp[tile_y/2 +4 + k*F/2].y;\
	temp20 += thetaTemp[tile_x/2 +1 + k*F/2].x * thetaTemp[tile_y/2 + k*F/2].x;\
	temp21 += thetaTemp[tile_x/2 +1 + k*F/2].x * thetaTemp[tile_y/2 + k*F/2].y;\
	temp22 += thetaTemp[tile_x/2 +1 + k*F/2].x * thetaTemp[tile_y/2 +1 + k*F/2].x;\
	temp23 += thetaTemp[tile_x/2 +1 + k*F/2].x * thetaTemp[tile_y/2 +1 + k*F/2].y;\
	temp24 += thetaTemp[tile_x/2 +1 + k*F/2].x * thetaTemp[tile_y/2 +2 + k*F/2].x;\
	temp25 += thetaTemp[tile_x/2 +1 + k*F/2].x * thetaTemp[tile_y/2 +2 + k*F/2].y;\
	temp26 += thetaTemp[tile_x/2 +1 + k*F/2].x * thetaTemp[tile_y/2 +3 + k*F/2].x;\
	temp27 += thetaTemp[tile_x/2 +1 + k*F/2].x * thetaTemp[tile_y/2 +3 + k*F/2].y;\
	temp28 += thetaTemp[tile_x/2 +1 + k*F/2].x * thetaTemp[tile_y/2 +4 + k*F/2].x;\
	temp29 += thetaTemp[tile_x/2 +1 + k*F/2].x * thetaTemp[tile_y/2 +4 + k*F/2].y;\
	temp30 += thetaTemp[tile_x/2 +1 + k*F/2].y * thetaTemp[tile_y/2 + k*F/2].x;\
	temp31 += thetaTemp[tile_x/2 +1 + k*F/2].y * thetaTemp[tile_y/2 + k*F/2].y;\
	temp32 += thetaTemp[tile_x/2 +1 + k*F/2].y * thetaTemp[tile_y/2 +1 + k*F/2].x;\
	temp33 += thetaTemp[tile_x/2 +1 + k*F/2].y * thetaTemp[tile_y/2 +1 + k*F/2].y;\
	temp34 += thetaTemp[tile_x/2 +1 + k*F/2].y * thetaTemp[tile_y/2 +2 + k*F/2].x;\
	temp35 += thetaTemp[tile_x/2 +1 + k*F/2].y * thetaTemp[tile_y/2 +2 + k*F/2].y;\
	temp36 += thetaTemp[tile_x/2 +1 + k*F/2].y * thetaTemp[tile_y/2 +3 + k*F/2].x;\
	temp37 += thetaTemp[tile_x/2 +1 + k*F/2].y * thetaTemp[tile_y/2 +3 + k*F/2].y;\
	temp38 += thetaTemp[tile_x/2 +1 + k*F/2].y * thetaTemp[tile_y/2 +4 + k*F/2].x;\
	temp39 += thetaTemp[tile_x/2 +1 + k*F/2].y * thetaTemp[tile_y/2 +4 + k*F/2].y;\
	temp40 += thetaTemp[tile_x/2 +2 + k*F/2].x * thetaTemp[tile_y/2 + k*F/2].x;\
	temp41 += thetaTemp[tile_x/2 +2 + k*F/2].x * thetaTemp[tile_y/2 + k*F/2].y;\
	temp42 += thetaTemp[tile_x/2 +2 + k*F/2].x * thetaTemp[tile_y/2 +1 + k*F/2].x;\
	temp43 += thetaTemp[tile_x/2 +2 + k*F/2].x * thetaTemp[tile_y/2 +1 + k*F/2].y;\
	temp44 += thetaTemp[tile_x/2 +2 + k*F/2].x * thetaTemp[tile_y/2 +2 + k*F/2].x;\
	temp45 += thetaTemp[tile_x/2 +2 + k*F/2].x * thetaTemp[tile_y/2 +2 + k*F/2].y;\
	temp46 += thetaTemp[tile_x/2 +2 + k*F/2].x * thetaTemp[tile_y/2 +3 + k*F/2].x;\
	temp47 += thetaTemp[tile_x/2 +2 + k*F/2].x * thetaTemp[tile_y/2 +3 + k*F/2].y;\
	temp48 += thetaTemp[tile_x/2 +2 + k*F/2].x * thetaTemp[tile_y/2 +4 + k*F/2].x;\
	temp49 += thetaTemp[tile_x/2 +2 + k*F/2].x * thetaTemp[tile_y/2 +4 + k*F/2].y;\
	temp50 += thetaTemp[tile_x/2 +2 + k*F/2].y * thetaTemp[tile_y/2 + k*F/2].x;\
	temp51 += thetaTemp[tile_x/2 +2 + k*F/2].y * thetaTemp[tile_y/2 + k*F/2].y;\
	temp52 += thetaTemp[tile_x/2 +2 + k*F/2].y * thetaTemp[tile_y/2 +1 + k*F/2].x;\
	temp53 += thetaTemp[tile_x/2 +2 + k*F/2].y * thetaTemp[tile_y/2 +1 + k*F/2].y;\
	temp54 += thetaTemp[tile_x/2 +2 + k*F/2].y * thetaTemp[tile_y/2 +2 + k*F/2].x;\
	temp55 += thetaTemp[tile_x/2 +2 + k*F/2].y * thetaTemp[tile_y/2 +2 + k*F/2].y;\
	temp56 += thetaTemp[tile_x/2 +2 + k*F/2].y * thetaTemp[tile_y/2 +3 + k*F/2].x;\
	temp57 += thetaTemp[tile_x/2 +2 + k*F/2].y * thetaTemp[tile_y/2 +3 + k*F/2].y;\
	temp58 += thetaTemp[tile_x/2 +2 + k*F/2].y * thetaTemp[tile_y/2 +4 + k*F/2].x;\
	temp59 += thetaTemp[tile_x/2 +2 + k*F/2].y * thetaTemp[tile_y/2 +4 + k*F/2].y;\
	temp60 += thetaTemp[tile_x/2 +3 + k*F/2].x * thetaTemp[tile_y/2 + k*F/2].x;\
	temp61 += thetaTemp[tile_x/2 +3 + k*F/2].x * thetaTemp[tile_y/2 + k*F/2].y;\
	temp62 += thetaTemp[tile_x/2 +3 + k*F/2].x * thetaTemp[tile_y/2 +1 + k*F/2].x;\
	temp63 += thetaTemp[tile_x/2 +3 + k*F/2].x * thetaTemp[tile_y/2 +1 + k*F/2].y;\
	temp64 += thetaTemp[tile_x/2 +3 + k*F/2].x * thetaTemp[tile_y/2 +2 + k*F/2].x;\
	temp65 += thetaTemp[tile_x/2 +3 + k*F/2].x * thetaTemp[tile_y/2 +2 + k*F/2].y;\
	temp66 += thetaTemp[tile_x/2 +3 + k*F/2].x * thetaTemp[tile_y/2 +3 + k*F/2].x;\
	temp67 += thetaTemp[tile_x/2 +3 + k*F/2].x * thetaTemp[tile_y/2 +3 + k*F/2].y;\
	temp68 += thetaTemp[tile_x/2 +3 + k*F/2].x * thetaTemp[tile_y/2 +4 + k*F/2].x;\
	temp69 += thetaTemp[tile_x/2 +3 + k*F/2].x * thetaTemp[tile_y/2 +4 + k*F/2].y;\
	temp70 += thetaTemp[tile_x/2 +3 + k*F/2].y * thetaTemp[tile_y/2 + k*F/2].x;\
	temp71 += thetaTemp[tile_x/2 +3 + k*F/2].y * thetaTemp[tile_y/2 + k*F/2].y;\
	temp72 += thetaTemp[tile_x/2 +3 + k*F/2].y * thetaTemp[tile_y/2 +1 + k*F/2].x;\
	temp73 += thetaTemp[tile_x/2 +3 + k*F/2].y * thetaTemp[tile_y/2 +1 + k*F/2].y;\
	temp74 += thetaTemp[tile_x/2 +3 + k*F/2].y * thetaTemp[tile_y/2 +2 + k*F/2].x;\
	temp75 += thetaTemp[tile_x/2 +3 + k*F/2].y * thetaTemp[tile_y/2 +2 + k*F/2].y;\
	temp76 += thetaTemp[tile_x/2 +3 + k*F/2].y * thetaTemp[tile_y/2 +3 + k*F/2].x;\
	temp77 += thetaTemp[tile_x/2 +3 + k*F/2].y * thetaTemp[tile_y/2 +3 + k*F/2].y;\
	temp78 += thetaTemp[tile_x/2 +3 + k*F/2].y * thetaTemp[tile_y/2 +4 + k*F/2].x;\
	temp79 += thetaTemp[tile_x/2 +3 + k*F/2].y * thetaTemp[tile_y/2 +4 + k*F/2].y;\
	temp80 += thetaTemp[tile_x/2 +4 + k*F/2].x * thetaTemp[tile_y/2 + k*F/2].x;\
	temp81 += thetaTemp[tile_x/2 +4 + k*F/2].x * thetaTemp[tile_y/2 + k*F/2].y;\
	temp82 += thetaTemp[tile_x/2 +4 + k*F/2].x * thetaTemp[tile_y/2 +1 + k*F/2].x;\
	temp83 += thetaTemp[tile_x/2 +4 + k*F/2].x * thetaTemp[tile_y/2 +1 + k*F/2].y;\
	temp84 += thetaTemp[tile_x/2 +4 + k*F/2].x * thetaTemp[tile_y/2 +2 + k*F/2].x;\
	temp85 += thetaTemp[tile_x/2 +4 + k*F/2].x * thetaTemp[tile_y/2 +2 + k*F/2].y;\
	temp86 += thetaTemp[tile_x/2 +4 + k*F/2].x * thetaTemp[tile_y/2 +3 + k*F/2].x;\
	temp87 += thetaTemp[tile_x/2 +4 + k*F/2].x * thetaTemp[tile_y/2 +3 + k*F/2].y;\
	temp88 += thetaTemp[tile_x/2 +4 + k*F/2].x * thetaTemp[tile_y/2 +4 + k*F/2].x;\
	temp89 += thetaTemp[tile_x/2 +4 + k*F/2].x * thetaTemp[tile_y/2 +4 + k*F/2].y;\
	temp90 += thetaTemp[tile_x/2 +4 + k*F/2].y * thetaTemp[tile_y/2 + k*F/2].x;\
	temp91 += thetaTemp[tile_x/2 +4 + k*F/2].y * thetaTemp[tile_y/2 + k*F/2].y;\
	temp92 += thetaTemp[tile_x/2 +4 + k*F/2].y * thetaTemp[tile_y/2 +1 + k*F/2].x;\
	temp93 += thetaTemp[tile_x/2 +4 + k*F/2].y * thetaTemp[tile_y/2 +1 + k*F/2].y;\
	temp94 += thetaTemp[tile_x/2 +4 + k*F/2].y * thetaTemp[tile_y/2 +2 + k*F/2].x;\
	temp95 += thetaTemp[tile_x/2 +4 + k*F/2].y * thetaTemp[tile_y/2 +2 + k*F/2].y;\
	temp96 += thetaTemp[tile_x/2 +4 + k*F/2].y * thetaTemp[tile_y/2 +3 + k*F/2].x;\
	temp97 += thetaTemp[tile_x/2 +4 + k*F/2].y * thetaTemp[tile_y/2 +3 + k*F/2].y;\
	temp98 += thetaTemp[tile_x/2 +4 + k*F/2].y * thetaTemp[tile_y/2 +4 + k*F/2].x;\
	temp99 += thetaTemp[tile_x/2 +4 + k*F/2].y * thetaTemp[tile_y/2 +4 + k*F/2].y;\
	}\
	while (0)\

#define fill_upper_half_from_registers()\
do{\
	tt[index + tile_x + tile_y*F] = temp0;\
	tt[index + tile_x + (tile_y + 1)*F] = temp1;\
	tt[index + tile_x + (tile_y + 2)*F] = temp2;\
	tt[index + tile_x + (tile_y + 3)*F] = temp3;\
	tt[index + tile_x + (tile_y + 4)*F] = temp4;\
	tt[index + tile_x + (tile_y + 5)*F] = temp5;\
	tt[index + tile_x + (tile_y + 6)*F] = temp6;\
	tt[index + tile_x + (tile_y + 7)*F] = temp7;\
	tt[index + tile_x + (tile_y + 8)*F] = temp8;\
	tt[index + tile_x + (tile_y + 9)*F] = temp9;\
\
	tt[index + tile_x + 1 + tile_y*F] = temp10;\
	tt[index + tile_x + 1 + (tile_y + 1)*F] = temp11;\
	tt[index + tile_x + 1 + (tile_y + 2)*F] = temp12;\
	tt[index + tile_x + 1 + (tile_y + 3)*F] = temp13;\
	tt[index + tile_x + 1 + (tile_y + 4)*F] = temp14;\
	tt[index + tile_x + 1 + (tile_y + 5)*F] = temp15;\
	tt[index + tile_x + 1 + (tile_y + 6)*F] = temp16;\
	tt[index + tile_x + 1 + (tile_y + 7)*F] = temp17;\
	tt[index + tile_x + 1 + (tile_y + 8)*F] = temp18;\
	tt[index + tile_x + 1 + (tile_y + 9)*F] = temp19;\
\
	tt[index + tile_x + 2 + tile_y*F] = temp20;\
	tt[index + tile_x + 2 + (tile_y + 1)*F] = temp21;\
	tt[index + tile_x + 2 + (tile_y + 2)*F] = temp22;\
	tt[index + tile_x + 2 + (tile_y + 3)*F] = temp23;\
	tt[index + tile_x + 2 + (tile_y + 4)*F] = temp24;\
	tt[index + tile_x + 2 + (tile_y + 5)*F] = temp25;\
	tt[index + tile_x + 2 + (tile_y + 6)*F] = temp26;\
	tt[index + tile_x + 2 + (tile_y + 7)*F] = temp27;\
	tt[index + tile_x + 2 + (tile_y + 8)*F] = temp28;\
	tt[index + tile_x + 2 + (tile_y + 9)*F] = temp29;\
\
	tt[index + tile_x + 3 + tile_y*F] = temp30;\
	tt[index + tile_x + 3 + (tile_y + 1)*F] = temp31;\
	tt[index + tile_x + 3 + (tile_y + 2)*F] = temp32;\
	tt[index + tile_x + 3 + (tile_y + 3)*F] = temp33;\
	tt[index + tile_x + 3 + (tile_y + 4)*F] = temp34;\
	tt[index + tile_x + 3 + (tile_y + 5)*F] = temp35;\
	tt[index + tile_x + 3 + (tile_y + 6)*F] = temp36;\
	tt[index + tile_x + 3 + (tile_y + 7)*F] = temp37;\
	tt[index + tile_x + 3 + (tile_y + 8)*F] = temp38;\
	tt[index + tile_x + 3 + (tile_y + 9)*F] = temp39;\
\
	tt[index + tile_x + 4 + tile_y*F] = temp40;\
	tt[index + tile_x + 4 + (tile_y + 1)*F] = temp41;\
	tt[index + tile_x + 4 + (tile_y + 2)*F] = temp42;\
	tt[index + tile_x + 4 + (tile_y + 3)*F] = temp43;\
	tt[index + tile_x + 4 + (tile_y + 4)*F] = temp44;\
	tt[index + tile_x + 4 + (tile_y + 5)*F] = temp45;\
	tt[index + tile_x + 4 + (tile_y + 6)*F] = temp46;\
	tt[index + tile_x + 4 + (tile_y + 7)*F] = temp47;\
	tt[index + tile_x + 4 + (tile_y + 8)*F] = temp48;\
	tt[index + tile_x + 4 + (tile_y + 9)*F] = temp49;\
\
	tt[index + tile_x + 5 + tile_y*F] = temp50;\
	tt[index + tile_x + 5 + (tile_y + 1)*F] = temp51;\
	tt[index + tile_x + 5 + (tile_y + 2)*F] = temp52;\
	tt[index + tile_x + 5 + (tile_y + 3)*F] = temp53;\
	tt[index + tile_x + 5 + (tile_y + 4)*F] = temp54;\
	tt[index + tile_x + 5 + (tile_y + 5)*F] = temp55;\
	tt[index + tile_x + 5 + (tile_y + 6)*F] = temp56;\
	tt[index + tile_x + 5 + (tile_y + 7)*F] = temp57;\
	tt[index + tile_x + 5 + (tile_y + 8)*F] = temp58;\
	tt[index + tile_x + 5 + (tile_y + 9)*F] = temp59;\
\
	tt[index + tile_x + 6 + tile_y*F] = temp60;\
	tt[index + tile_x + 6 + (tile_y + 1)*F] = temp61;\
	tt[index + tile_x + 6 + (tile_y + 2)*F] = temp62;\
	tt[index + tile_x + 6 + (tile_y + 3)*F] = temp63;\
	tt[index + tile_x + 6 + (tile_y + 4)*F] = temp64;\
	tt[index + tile_x + 6 + (tile_y + 5)*F] = temp65;\
	tt[index + tile_x + 6 + (tile_y + 6)*F] = temp66;\
	tt[index + tile_x + 6 + (tile_y + 7)*F] = temp67;\
	tt[index + tile_x + 6 + (tile_y + 8)*F] = temp68;\
	tt[index + tile_x + 6 + (tile_y + 9)*F] = temp69;\
\
	tt[index + tile_x + 7 + tile_y*F] = temp70;\
	tt[index + tile_x + 7 + (tile_y + 1)*F] = temp71;\
	tt[index + tile_x + 7 + (tile_y + 2)*F] = temp72;\
	tt[index + tile_x + 7 + (tile_y + 3)*F] = temp73;\
	tt[index + tile_x + 7 + (tile_y + 4)*F] = temp74;\
	tt[index + tile_x + 7 + (tile_y + 5)*F] = temp75;\
	tt[index + tile_x + 7 + (tile_y + 6)*F] = temp76;\
	tt[index + tile_x + 7 + (tile_y + 7)*F] = temp77;\
	tt[index + tile_x + 7 + (tile_y + 8)*F] = temp78;\
	tt[index + tile_x + 7 + (tile_y + 9)*F] = temp79;\
\
	tt[index + tile_x + 8 + tile_y*F] = temp80;\
	tt[index + tile_x + 8 + (tile_y + 1)*F] = temp81;\
	tt[index + tile_x + 8 + (tile_y + 2)*F] = temp82;\
	tt[index + tile_x + 8 + (tile_y + 3)*F] = temp83;\
	tt[index + tile_x + 8 + (tile_y + 4)*F] = temp84;\
	tt[index + tile_x + 8 + (tile_y + 5)*F] = temp85;\
	tt[index + tile_x + 8 + (tile_y + 6)*F] = temp86;\
	tt[index + tile_x + 8 + (tile_y + 7)*F] = temp87;\
	tt[index + tile_x + 8 + (tile_y + 8)*F] = temp88;\
	tt[index + tile_x + 8 + (tile_y + 9)*F] = temp89;\
\
	tt[index + tile_x + 9 + tile_y*F] = temp90;\
	tt[index + tile_x + 9 + (tile_y + 1)*F] = temp91;\
	tt[index + tile_x + 9 + (tile_y + 2)*F] = temp92;\
	tt[index + tile_x + 9 + (tile_y + 3)*F] = temp93;\
	tt[index + tile_x + 9 + (tile_y + 4)*F] = temp94;\
	tt[index + tile_x + 9 + (tile_y + 5)*F] = temp95;\
	tt[index + tile_x + 9 + (tile_y + 6)*F] = temp96;\
	tt[index + tile_x + 9 + (tile_y + 7)*F] = temp97;\
	tt[index + tile_x + 9 + (tile_y + 8)*F] = temp98;\
	tt[index + tile_x + 9 + (tile_y + 9)*F] = temp99;\
}\
while(0)
	
#define fill_lower_half_from_registers()\
do{\
	tt[index + tile_y + 0+ (tile_x + 0)*F]= temp0;\
	tt[index + tile_y + 1+ (tile_x + 0)*F]= temp1;\
	tt[index + tile_y + 2+ (tile_x + 0)*F]= temp2;\
	tt[index + tile_y + 3+ (tile_x + 0)*F]= temp3;\
	tt[index + tile_y + 4+ (tile_x + 0)*F]= temp4;\
	tt[index + tile_y + 5+ (tile_x + 0)*F]= temp5;\
	tt[index + tile_y + 6+ (tile_x + 0)*F]= temp6;\
	tt[index + tile_y + 7+ (tile_x + 0)*F]= temp7;\
	tt[index + tile_y + 8+ (tile_x + 0)*F]= temp8;\
	tt[index + tile_y + 9+ (tile_x + 0)*F]= temp9;\
\
	tt[index + tile_y + 0+ (tile_x + 1)*F]= temp10;\
	tt[index + tile_y + 1+ (tile_x + 1)*F]= temp11;\
	tt[index + tile_y + 2+ (tile_x + 1)*F]= temp12;\
	tt[index + tile_y + 3+ (tile_x + 1)*F]= temp13;\
	tt[index + tile_y + 4+ (tile_x + 1)*F]= temp14;\
	tt[index + tile_y + 5+ (tile_x + 1)*F]= temp15;\
	tt[index + tile_y + 6+ (tile_x + 1)*F]= temp16;\
	tt[index + tile_y + 7+ (tile_x + 1)*F]= temp17;\
	tt[index + tile_y + 8+ (tile_x + 1)*F]= temp18;\
	tt[index + tile_y + 9+ (tile_x + 1)*F]= temp19;\
\
	tt[index + tile_y + 0+ (tile_x + 2)*F]= temp20;\
	tt[index + tile_y + 1+ (tile_x + 2)*F]= temp21;\
	tt[index + tile_y + 2+ (tile_x + 2)*F]= temp22;\
	tt[index + tile_y + 3+ (tile_x + 2)*F]= temp23;\
	tt[index + tile_y + 4+ (tile_x + 2)*F]= temp24;\
	tt[index + tile_y + 5+ (tile_x + 2)*F]= temp25;\
	tt[index + tile_y + 6+ (tile_x + 2)*F]= temp26;\
	tt[index + tile_y + 7+ (tile_x + 2)*F]= temp27;\
	tt[index + tile_y + 8+ (tile_x + 2)*F]= temp28;\
	tt[index + tile_y + 9+ (tile_x + 2)*F]= temp29;\
\
	tt[index + tile_y + 0+ (tile_x + 3)*F]= temp30;\
	tt[index + tile_y + 1+ (tile_x + 3)*F]= temp31;\
	tt[index + tile_y + 2+ (tile_x + 3)*F]= temp32;\
	tt[index + tile_y + 3+ (tile_x + 3)*F]= temp33;\
	tt[index + tile_y + 4+ (tile_x + 3)*F]= temp34;\
	tt[index + tile_y + 5+ (tile_x + 3)*F]= temp35;\
	tt[index + tile_y + 6+ (tile_x + 3)*F]= temp36;\
	tt[index + tile_y + 7+ (tile_x + 3)*F]= temp37;\
	tt[index + tile_y + 8+ (tile_x + 3)*F]= temp38;\
	tt[index + tile_y + 9+ (tile_x + 3)*F]= temp39;\
\
	tt[index + tile_y + 0+ (tile_x + 4)*F]= temp40;\
	tt[index + tile_y + 1+ (tile_x + 4)*F]= temp41;\
	tt[index + tile_y + 2+ (tile_x + 4)*F]= temp42;\
	tt[index + tile_y + 3+ (tile_x + 4)*F]= temp43;\
	tt[index + tile_y + 4+ (tile_x + 4)*F]= temp44;\
	tt[index + tile_y + 5+ (tile_x + 4)*F]= temp45;\
	tt[index + tile_y + 6+ (tile_x + 4)*F]= temp46;\
	tt[index + tile_y + 7+ (tile_x + 4)*F]= temp47;\
	tt[index + tile_y + 8+ (tile_x + 4)*F]= temp48;\
	tt[index + tile_y + 9+ (tile_x + 4)*F]= temp49;\
\
	tt[index + tile_y + 0+ (tile_x + 5)*F]= temp50;\
	tt[index + tile_y + 1+ (tile_x + 5)*F]= temp51;\
	tt[index + tile_y + 2+ (tile_x + 5)*F]= temp52;\
	tt[index + tile_y + 3+ (tile_x + 5)*F]= temp53;\
	tt[index + tile_y + 4+ (tile_x + 5)*F]= temp54;\
	tt[index + tile_y + 5+ (tile_x + 5)*F]= temp55;\
	tt[index + tile_y + 6+ (tile_x + 5)*F]= temp56;\
	tt[index + tile_y + 7+ (tile_x + 5)*F]= temp57;\
	tt[index + tile_y + 8+ (tile_x + 5)*F]= temp58;\
	tt[index + tile_y + 9+ (tile_x + 5)*F]= temp59;\
\
	tt[index + tile_y + 0+ (tile_x + 6)*F]= temp60;\
	tt[index + tile_y + 1+ (tile_x + 6)*F]= temp61;\
	tt[index + tile_y + 2+ (tile_x + 6)*F]= temp62;\
	tt[index + tile_y + 3+ (tile_x + 6)*F]= temp63;\
	tt[index + tile_y + 4+ (tile_x + 6)*F]= temp64;\
	tt[index + tile_y + 5+ (tile_x + 6)*F]= temp65;\
	tt[index + tile_y + 6+ (tile_x + 6)*F]= temp66;\
	tt[index + tile_y + 7+ (tile_x + 6)*F]= temp67;\
	tt[index + tile_y + 8+ (tile_x + 6)*F]= temp68;\
	tt[index + tile_y + 9+ (tile_x + 6)*F]= temp69;\
\
	tt[index + tile_y + 0+ (tile_x + 7)*F]= temp70;\
	tt[index + tile_y + 1+ (tile_x + 7)*F]= temp71;\
	tt[index + tile_y + 2+ (tile_x + 7)*F]= temp72;\
	tt[index + tile_y + 3+ (tile_x + 7)*F]= temp73;\
	tt[index + tile_y + 4+ (tile_x + 7)*F]= temp74;\
	tt[index + tile_y + 5+ (tile_x + 7)*F]= temp75;\
	tt[index + tile_y + 6+ (tile_x + 7)*F]= temp76;\
	tt[index + tile_y + 7+ (tile_x + 7)*F]= temp77;\
	tt[index + tile_y + 8+ (tile_x + 7)*F]= temp78;\
	tt[index + tile_y + 9+ (tile_x + 7)*F]= temp79;\
\
	tt[index + tile_y + 0+ (tile_x + 8)*F]= temp80;\
	tt[index + tile_y + 1+ (tile_x + 8)*F]= temp81;\
	tt[index + tile_y + 2+ (tile_x + 8)*F]= temp82;\
	tt[index + tile_y + 3+ (tile_x + 8)*F]= temp83;\
	tt[index + tile_y + 4+ (tile_x + 8)*F]= temp84;\
	tt[index + tile_y + 5+ (tile_x + 8)*F]= temp85;\
	tt[index + tile_y + 6+ (tile_x + 8)*F]= temp86;\
	tt[index + tile_y + 7+ (tile_x + 8)*F]= temp87;\
	tt[index + tile_y + 8+ (tile_x + 8)*F]= temp88;\
	tt[index + tile_y + 9+ (tile_x + 8)*F]= temp89;\
\
	tt[index + tile_y + 0+ (tile_x + 9)*F]= temp90;\
	tt[index + tile_y + 1+ (tile_x + 9)*F]= temp91;\
	tt[index + tile_y + 2+ (tile_x + 9)*F]= temp92;\
	tt[index + tile_y + 3+ (tile_x + 9)*F]= temp93;\
	tt[index + tile_y + 4+ (tile_x + 9)*F]= temp94;\
	tt[index + tile_y + 5+ (tile_x + 9)*F]= temp95;\
	tt[index + tile_y + 6+ (tile_x + 9)*F]= temp96;\
	tt[index + tile_y + 7+ (tile_x + 9)*F]= temp97;\
	tt[index + tile_y + 8+ (tile_x + 9)*F]= temp98;\
	tt[index + tile_y + 9+ (tile_x + 9)*F]= temp99;\
}\
while(0)

#define fill_lower_half_from_registers_fp16()\
do{\
	tt[index + tile_y/2 + 0 + (tile_x + 0)*F/2] = __floats2half2_rn(temp0, temp1); \
	tt[index + tile_y/2 + 1 + (tile_x + 0)*F/2] = __floats2half2_rn(temp2, temp3); \
	tt[index + tile_y/2 + 2 + (tile_x + 0)*F/2] = __floats2half2_rn(temp4, temp5); \
	tt[index + tile_y/2 + 3 + (tile_x + 0)*F/2] = __floats2half2_rn(temp6, temp7); \
	tt[index + tile_y/2 + 4 + (tile_x + 0)*F/2] = __floats2half2_rn(temp8, temp9); \
	\
	tt[index + tile_y/2 + 0 + (tile_x + 1)*F/2] = __floats2half2_rn(temp10, temp11); \
	tt[index + tile_y/2 + 1 + (tile_x + 1)*F/2] = __floats2half2_rn(temp12, temp13); \
	tt[index + tile_y/2 + 2 + (tile_x + 1)*F/2] = __floats2half2_rn(temp14, temp15); \
	tt[index + tile_y/2 + 3 + (tile_x + 1)*F/2] = __floats2half2_rn(temp16, temp17); \
	tt[index + tile_y/2 + 4 + (tile_x + 1)*F/2] = __floats2half2_rn(temp18, temp19); \
	\
	tt[index + tile_y/2 + 0 + (tile_x + 2)*F/2] = __floats2half2_rn(temp20, temp21); \
	tt[index + tile_y/2 + 1 + (tile_x + 2)*F/2] = __floats2half2_rn(temp22, temp23); \
	tt[index + tile_y/2 + 2 + (tile_x + 2)*F/2] = __floats2half2_rn(temp24, temp25); \
	tt[index + tile_y/2 + 3 + (tile_x + 2)*F/2] = __floats2half2_rn(temp26, temp27); \
	tt[index + tile_y/2 + 4 + (tile_x + 2)*F/2] = __floats2half2_rn(temp28, temp29); \
	\
	tt[index + tile_y/2 + 0 + (tile_x + 3)*F/2] = __floats2half2_rn(temp30, temp31); \
	tt[index + tile_y/2 + 1 + (tile_x + 3)*F/2] = __floats2half2_rn(temp32, temp33); \
	tt[index + tile_y/2 + 2 + (tile_x + 3)*F/2] = __floats2half2_rn(temp34, temp35); \
	tt[index + tile_y/2 + 3 + (tile_x + 3)*F/2] = __floats2half2_rn(temp36, temp37); \
	tt[index + tile_y/2 + 4 + (tile_x + 3)*F/2] = __floats2half2_rn(temp38, temp39); \
	\
	tt[index + tile_y/2 + 0 + (tile_x + 4)*F/2] = __floats2half2_rn(temp40, temp41); \
	tt[index + tile_y/2 + 1 + (tile_x + 4)*F/2] = __floats2half2_rn(temp42, temp43); \
	tt[index + tile_y/2 + 2 + (tile_x + 4)*F/2] = __floats2half2_rn(temp44, temp45); \
	tt[index + tile_y/2 + 3 + (tile_x + 4)*F/2] = __floats2half2_rn(temp46, temp47); \
	tt[index + tile_y/2 + 4 + (tile_x + 4)*F/2] = __floats2half2_rn(temp48, temp49); \
	\
	tt[index + tile_y/2 + 0 + (tile_x + 5)*F/2] = __floats2half2_rn(temp50, temp51); \
	tt[index + tile_y/2 + 1 + (tile_x + 5)*F/2] = __floats2half2_rn(temp52, temp53); \
	tt[index + tile_y/2 + 2 + (tile_x + 5)*F/2] = __floats2half2_rn(temp54, temp55); \
	tt[index + tile_y/2 + 3 + (tile_x + 5)*F/2] = __floats2half2_rn(temp56, temp57); \
	tt[index + tile_y/2 + 4 + (tile_x + 5)*F/2] = __floats2half2_rn(temp58, temp59); \
	\
	tt[index + tile_y/2 + 0 + (tile_x + 6)*F/2] = __floats2half2_rn(temp60, temp61); \
	tt[index + tile_y/2 + 1 + (tile_x + 6)*F/2] = __floats2half2_rn(temp62, temp63); \
	tt[index + tile_y/2 + 2 + (tile_x + 6)*F/2] = __floats2half2_rn(temp64, temp65); \
	tt[index + tile_y/2 + 3 + (tile_x + 6)*F/2] = __floats2half2_rn(temp66, temp67); \
	tt[index + tile_y/2 + 4 + (tile_x + 6)*F/2] = __floats2half2_rn(temp68, temp69); \
	\
	tt[index + tile_y/2 + 0 + (tile_x + 7)*F/2] = __floats2half2_rn(temp70, temp71); \
	tt[index + tile_y/2 + 1 + (tile_x + 7)*F/2] = __floats2half2_rn(temp72, temp73); \
	tt[index + tile_y/2 + 2 + (tile_x + 7)*F/2] = __floats2half2_rn(temp74, temp75); \
	tt[index + tile_y/2 + 3 + (tile_x + 7)*F/2] = __floats2half2_rn(temp76, temp77); \
	tt[index + tile_y/2 + 4 + (tile_x + 7)*F/2] = __floats2half2_rn(temp78, temp79); \
	\
	tt[index + tile_y/2 + 0 + (tile_x + 8)*F/2] = __floats2half2_rn(temp80, temp81); \
	tt[index + tile_y/2 + 1 + (tile_x + 8)*F/2] = __floats2half2_rn(temp82, temp83); \
	tt[index + tile_y/2 + 2 + (tile_x + 8)*F/2] = __floats2half2_rn(temp84, temp85); \
	tt[index + tile_y/2 + 3 + (tile_x + 8)*F/2] = __floats2half2_rn(temp86, temp87); \
	tt[index + tile_y/2 + 4 + (tile_x + 8)*F/2] = __floats2half2_rn(temp88, temp89); \
	\
	tt[index + tile_y/2 + 0 + (tile_x + 9)*F/2] = __floats2half2_rn(temp90, temp91); \
	tt[index + tile_y/2 + 1 + (tile_x + 9)*F/2] = __floats2half2_rn(temp92, temp93); \
	tt[index + tile_y/2 + 2 + (tile_x + 9)*F/2] = __floats2half2_rn(temp94, temp95); \
	tt[index + tile_y/2 + 3 + (tile_x + 9)*F/2] = __floats2half2_rn(temp96, temp97); \
	tt[index + tile_y/2 + 4 + (tile_x + 9)*F/2] = __floats2half2_rn(temp98, temp99); \
}\
while(0)
	
#define fill_upper_half_from_registers_fp16()\
do{\
	tt[index + tile_x/2 + 0 + (tile_y + 0)*F/2] = __floats2half2_rn(temp0, temp10); \
	tt[index + tile_x/2 + 1 + (tile_y + 0)*F/2] = __floats2half2_rn(temp20, temp30); \
	tt[index + tile_x/2 + 2 + (tile_y + 0)*F/2] = __floats2half2_rn(temp40, temp50); \
	tt[index + tile_x/2 + 3 + (tile_y + 0)*F/2] = __floats2half2_rn(temp60, temp70); \
	tt[index + tile_x/2 + 4 + (tile_y + 0)*F/2] = __floats2half2_rn(temp80, temp90); \
	\
	tt[index + tile_x/2 + 0 + (tile_y + 1)*F/2] = __floats2half2_rn(temp1, temp11); \
	tt[index + tile_x/2 + 1 + (tile_y + 1)*F/2] = __floats2half2_rn(temp21, temp31); \
	tt[index + tile_x/2 + 2 + (tile_y + 1)*F/2] = __floats2half2_rn(temp41, temp51); \
	tt[index + tile_x/2 + 3 + (tile_y + 1)*F/2] = __floats2half2_rn(temp61, temp71); \
	tt[index + tile_x/2 + 4 + (tile_y + 1)*F/2] = __floats2half2_rn(temp81, temp91); \
	\
	tt[index + tile_x/2 + 0 + (tile_y + 2)*F/2] = __floats2half2_rn(temp2, temp12); \
	tt[index + tile_x/2 + 1 + (tile_y + 2)*F/2] = __floats2half2_rn(temp22, temp32); \
	tt[index + tile_x/2 + 2 + (tile_y + 2)*F/2] = __floats2half2_rn(temp42, temp52); \
	tt[index + tile_x/2 + 3 + (tile_y + 2)*F/2] = __floats2half2_rn(temp62, temp72); \
	tt[index + tile_x/2 + 4 + (tile_y + 2)*F/2] = __floats2half2_rn(temp82, temp92); \
	\
	tt[index + tile_x/2 + 0 + (tile_y + 3)*F/2] = __floats2half2_rn(temp3, temp13); \
	tt[index + tile_x/2 + 1 + (tile_y + 3)*F/2] = __floats2half2_rn(temp23, temp33); \
	tt[index + tile_x/2 + 2 + (tile_y + 3)*F/2] = __floats2half2_rn(temp43, temp53); \
	tt[index + tile_x/2 + 3 + (tile_y + 3)*F/2] = __floats2half2_rn(temp63, temp73); \
	tt[index + tile_x/2 + 4 + (tile_y + 3)*F/2] = __floats2half2_rn(temp83, temp93); \
	\
	tt[index + tile_x/2 + 0 + (tile_y + 4)*F/2] = __floats2half2_rn(temp4, temp14); \
	tt[index + tile_x/2 + 1 + (tile_y + 4)*F/2] = __floats2half2_rn(temp24, temp34); \
	tt[index + tile_x/2 + 2 + (tile_y + 4)*F/2] = __floats2half2_rn(temp44, temp54); \
	tt[index + tile_x/2 + 3 + (tile_y + 4)*F/2] = __floats2half2_rn(temp64, temp74); \
	tt[index + tile_x/2 + 4 + (tile_y + 4)*F/2] = __floats2half2_rn(temp84, temp94); \
	\
	tt[index + tile_x/2 + 0 + (tile_y + 5)*F/2] = __floats2half2_rn(temp5, temp15); \
	tt[index + tile_x/2 + 1 + (tile_y + 5)*F/2] = __floats2half2_rn(temp25, temp35); \
	tt[index + tile_x/2 + 2 + (tile_y + 5)*F/2] = __floats2half2_rn(temp45, temp55); \
	tt[index + tile_x/2 + 3 + (tile_y + 5)*F/2] = __floats2half2_rn(temp65, temp75); \
	tt[index + tile_x/2 + 4 + (tile_y + 5)*F/2] = __floats2half2_rn(temp85, temp95); \
	\
	tt[index + tile_x/2 + 0 + (tile_y + 6)*F/2] = __floats2half2_rn(temp6, temp16); \
	tt[index + tile_x/2 + 1 + (tile_y + 6)*F/2] = __floats2half2_rn(temp26, temp36); \
	tt[index + tile_x/2 + 2 + (tile_y + 6)*F/2] = __floats2half2_rn(temp46, temp56); \
	tt[index + tile_x/2 + 3 + (tile_y + 6)*F/2] = __floats2half2_rn(temp66, temp76); \
	tt[index + tile_x/2 + 4 + (tile_y + 6)*F/2] = __floats2half2_rn(temp86, temp96); \
	\
	tt[index + tile_x/2 + 0 + (tile_y + 7)*F/2] = __floats2half2_rn(temp7, temp17); \
	tt[index + tile_x/2 + 1 + (tile_y + 7)*F/2] = __floats2half2_rn(temp27, temp37); \
	tt[index + tile_x/2 + 2 + (tile_y + 7)*F/2] = __floats2half2_rn(temp47, temp57); \
	tt[index + tile_x/2 + 3 + (tile_y + 7)*F/2] = __floats2half2_rn(temp67, temp77); \
	tt[index + tile_x/2 + 4 + (tile_y + 7)*F/2] = __floats2half2_rn(temp87, temp97); \
	\
	tt[index + tile_x/2 + 0 + (tile_y + 8)*F/2] = __floats2half2_rn(temp8, temp18); \
	tt[index + tile_x/2 + 1 + (tile_y + 8)*F/2] = __floats2half2_rn(temp28, temp38); \
	tt[index + tile_x/2 + 2 + (tile_y + 8)*F/2] = __floats2half2_rn(temp48, temp58); \
	tt[index + tile_x/2 + 3 + (tile_y + 8)*F/2] = __floats2half2_rn(temp68, temp78); \
	tt[index + tile_x/2 + 4 + (tile_y + 8)*F/2] = __floats2half2_rn(temp88, temp98); \
	\
	tt[index + tile_x/2 + 0 + (tile_y + 9)*F/2] = __floats2half2_rn(temp9, temp19); \
	tt[index + tile_x/2 + 1 + (tile_y + 9)*F/2] = __floats2half2_rn(temp29, temp39); \
	tt[index + tile_x/2 + 2 + (tile_y + 9)*F/2] = __floats2half2_rn(temp49, temp59); \
	tt[index + tile_x/2 + 3 + (tile_y + 9)*F/2] = __floats2half2_rn(temp69, temp79); \
	tt[index + tile_x/2 + 4 + (tile_y + 9)*F/2] = __floats2half2_rn(temp89, temp99); \
}\
while(0)
	
#define fill_upper_half_from_registers_float2()\
do{\
	tt[index + tile_x/2 + 0 + (tile_y + 0)*F/2] = make_float2(temp0, temp10); \
	tt[index + tile_x/2 + 1 + (tile_y + 0)*F/2] = make_float2(temp20, temp30); \
	tt[index + tile_x/2 + 2 + (tile_y + 0)*F/2] = make_float2(temp40, temp50); \
	tt[index + tile_x/2 + 3 + (tile_y + 0)*F/2] = make_float2(temp60, temp70); \
	tt[index + tile_x/2 + 4 + (tile_y + 0)*F/2] = make_float2(temp80, temp90); \
	\
	tt[index + tile_x/2 + 0 + (tile_y + 1)*F/2] = make_float2(temp1, temp11); \
	tt[index + tile_x/2 + 1 + (tile_y + 1)*F/2] = make_float2(temp21, temp31); \
	tt[index + tile_x/2 + 2 + (tile_y + 1)*F/2] = make_float2(temp41, temp51); \
	tt[index + tile_x/2 + 3 + (tile_y + 1)*F/2] = make_float2(temp61, temp71); \
	tt[index + tile_x/2 + 4 + (tile_y + 1)*F/2] = make_float2(temp81, temp91); \
	\
	tt[index + tile_x/2 + 0 + (tile_y + 2)*F/2] = make_float2(temp2, temp12); \
	tt[index + tile_x/2 + 1 + (tile_y + 2)*F/2] = make_float2(temp22, temp32); \
	tt[index + tile_x/2 + 2 + (tile_y + 2)*F/2] = make_float2(temp42, temp52); \
	tt[index + tile_x/2 + 3 + (tile_y + 2)*F/2] = make_float2(temp62, temp72); \
	tt[index + tile_x/2 + 4 + (tile_y + 2)*F/2] = make_float2(temp82, temp92); \
	\
	tt[index + tile_x/2 + 0 + (tile_y + 3)*F/2] = make_float2(temp3, temp13); \
	tt[index + tile_x/2 + 1 + (tile_y + 3)*F/2] = make_float2(temp23, temp33); \
	tt[index + tile_x/2 + 2 + (tile_y + 3)*F/2] = make_float2(temp43, temp53); \
	tt[index + tile_x/2 + 3 + (tile_y + 3)*F/2] = make_float2(temp63, temp73); \
	tt[index + tile_x/2 + 4 + (tile_y + 3)*F/2] = make_float2(temp83, temp93); \
	\
	tt[index + tile_x/2 + 0 + (tile_y + 4)*F/2] = make_float2(temp4, temp14); \
	tt[index + tile_x/2 + 1 + (tile_y + 4)*F/2] = make_float2(temp24, temp34); \
	tt[index + tile_x/2 + 2 + (tile_y + 4)*F/2] = make_float2(temp44, temp54); \
	tt[index + tile_x/2 + 3 + (tile_y + 4)*F/2] = make_float2(temp64, temp74); \
	tt[index + tile_x/2 + 4 + (tile_y + 4)*F/2] = make_float2(temp84, temp94); \
	\
	tt[index + tile_x/2 + 0 + (tile_y + 5)*F/2] = make_float2(temp5, temp15); \
	tt[index + tile_x/2 + 1 + (tile_y + 5)*F/2] = make_float2(temp25, temp35); \
	tt[index + tile_x/2 + 2 + (tile_y + 5)*F/2] = make_float2(temp45, temp55); \
	tt[index + tile_x/2 + 3 + (tile_y + 5)*F/2] = make_float2(temp65, temp75); \
	tt[index + tile_x/2 + 4 + (tile_y + 5)*F/2] = make_float2(temp85, temp95); \
	\
	tt[index + tile_x/2 + 0 + (tile_y + 6)*F/2] = make_float2(temp6, temp16); \
	tt[index + tile_x/2 + 1 + (tile_y + 6)*F/2] = make_float2(temp26, temp36); \
	tt[index + tile_x/2 + 2 + (tile_y + 6)*F/2] = make_float2(temp46, temp56); \
	tt[index + tile_x/2 + 3 + (tile_y + 6)*F/2] = make_float2(temp66, temp76); \
	tt[index + tile_x/2 + 4 + (tile_y + 6)*F/2] = make_float2(temp86, temp96); \
	\
	tt[index + tile_x/2 + 0 + (tile_y + 7)*F/2] = make_float2(temp7, temp17); \
	tt[index + tile_x/2 + 1 + (tile_y + 7)*F/2] = make_float2(temp27, temp37); \
	tt[index + tile_x/2 + 2 + (tile_y + 7)*F/2] = make_float2(temp47, temp57); \
	tt[index + tile_x/2 + 3 + (tile_y + 7)*F/2] = make_float2(temp67, temp77); \
	tt[index + tile_x/2 + 4 + (tile_y + 7)*F/2] = make_float2(temp87, temp97); \
	\
	tt[index + tile_x/2 + 0 + (tile_y + 8)*F/2] = make_float2(temp8, temp18); \
	tt[index + tile_x/2 + 1 + (tile_y + 8)*F/2] = make_float2(temp28, temp38); \
	tt[index + tile_x/2 + 2 + (tile_y + 8)*F/2] = make_float2(temp48, temp58); \
	tt[index + tile_x/2 + 3 + (tile_y + 8)*F/2] = make_float2(temp68, temp78); \
	tt[index + tile_x/2 + 4 + (tile_y + 8)*F/2] = make_float2(temp88, temp98); \
	\
	tt[index + tile_x/2 + 0 + (tile_y + 9)*F/2] = make_float2(temp9, temp19); \
	tt[index + tile_x/2 + 1 + (tile_y + 9)*F/2] = make_float2(temp29, temp39); \
	tt[index + tile_x/2 + 2 + (tile_y + 9)*F/2] = make_float2(temp49, temp59); \
	tt[index + tile_x/2 + 3 + (tile_y + 9)*F/2] = make_float2(temp69, temp79); \
	tt[index + tile_x/2 + 4 + (tile_y + 9)*F/2] = make_float2(temp89, temp99); \
}\
while(0)
	
#define fill_lower_half_from_registers_float2()\
do{\
	tt[index + tile_y/2 + 0 + (tile_x + 0)*F/2] = make_float2(temp0, temp1); \
	tt[index + tile_y/2 + 1 + (tile_x + 0)*F/2] = make_float2(temp2, temp3); \
	tt[index + tile_y/2 + 2 + (tile_x + 0)*F/2] = make_float2(temp4, temp5); \
	tt[index + tile_y/2 + 3 + (tile_x + 0)*F/2] = make_float2(temp6, temp7); \
	tt[index + tile_y/2 + 4 + (tile_x + 0)*F/2] = make_float2(temp8, temp9); \
	\
	tt[index + tile_y/2 + 0 + (tile_x + 1)*F/2] = make_float2(temp10, temp11); \
	tt[index + tile_y/2 + 1 + (tile_x + 1)*F/2] = make_float2(temp12, temp13); \
	tt[index + tile_y/2 + 2 + (tile_x + 1)*F/2] = make_float2(temp14, temp15); \
	tt[index + tile_y/2 + 3 + (tile_x + 1)*F/2] = make_float2(temp16, temp17); \
	tt[index + tile_y/2 + 4 + (tile_x + 1)*F/2] = make_float2(temp18, temp19); \
	\
	tt[index + tile_y/2 + 0 + (tile_x + 2)*F/2] = make_float2(temp20, temp21); \
	tt[index + tile_y/2 + 1 + (tile_x + 2)*F/2] = make_float2(temp22, temp23); \
	tt[index + tile_y/2 + 2 + (tile_x + 2)*F/2] = make_float2(temp24, temp25); \
	tt[index + tile_y/2 + 3 + (tile_x + 2)*F/2] = make_float2(temp26, temp27); \
	tt[index + tile_y/2 + 4 + (tile_x + 2)*F/2] = make_float2(temp28, temp29); \
	\
	tt[index + tile_y/2 + 0 + (tile_x + 3)*F/2] = make_float2(temp30, temp31); \
	tt[index + tile_y/2 + 1 + (tile_x + 3)*F/2] = make_float2(temp32, temp33); \
	tt[index + tile_y/2 + 2 + (tile_x + 3)*F/2] = make_float2(temp34, temp35); \
	tt[index + tile_y/2 + 3 + (tile_x + 3)*F/2] = make_float2(temp36, temp37); \
	tt[index + tile_y/2 + 4 + (tile_x + 3)*F/2] = make_float2(temp38, temp39); \
	\
	tt[index + tile_y/2 + 0 + (tile_x + 4)*F/2] = make_float2(temp40, temp41); \
	tt[index + tile_y/2 + 1 + (tile_x + 4)*F/2] = make_float2(temp42, temp43); \
	tt[index + tile_y/2 + 2 + (tile_x + 4)*F/2] = make_float2(temp44, temp45); \
	tt[index + tile_y/2 + 3 + (tile_x + 4)*F/2] = make_float2(temp46, temp47); \
	tt[index + tile_y/2 + 4 + (tile_x + 4)*F/2] = make_float2(temp48, temp49); \
	\
	tt[index + tile_y/2 + 0 + (tile_x + 5)*F/2] = make_float2(temp50, temp51); \
	tt[index + tile_y/2 + 1 + (tile_x + 5)*F/2] = make_float2(temp52, temp53); \
	tt[index + tile_y/2 + 2 + (tile_x + 5)*F/2] = make_float2(temp54, temp55); \
	tt[index + tile_y/2 + 3 + (tile_x + 5)*F/2] = make_float2(temp56, temp57); \
	tt[index + tile_y/2 + 4 + (tile_x + 5)*F/2] = make_float2(temp58, temp59); \
	\
	tt[index + tile_y/2 + 0 + (tile_x + 6)*F/2] = make_float2(temp60, temp61); \
	tt[index + tile_y/2 + 1 + (tile_x + 6)*F/2] = make_float2(temp62, temp63); \
	tt[index + tile_y/2 + 2 + (tile_x + 6)*F/2] = make_float2(temp64, temp65); \
	tt[index + tile_y/2 + 3 + (tile_x + 6)*F/2] = make_float2(temp66, temp67); \
	tt[index + tile_y/2 + 4 + (tile_x + 6)*F/2] = make_float2(temp68, temp69); \
	\
	tt[index + tile_y/2 + 0 + (tile_x + 7)*F/2] = make_float2(temp70, temp71); \
	tt[index + tile_y/2 + 1 + (tile_x + 7)*F/2] = make_float2(temp72, temp73); \
	tt[index + tile_y/2 + 2 + (tile_x + 7)*F/2] = make_float2(temp74, temp75); \
	tt[index + tile_y/2 + 3 + (tile_x + 7)*F/2] = make_float2(temp76, temp77); \
	tt[index + tile_y/2 + 4 + (tile_x + 7)*F/2] = make_float2(temp78, temp79); \
	\
	tt[index + tile_y/2 + 0 + (tile_x + 8)*F/2] = make_float2(temp80, temp81); \
	tt[index + tile_y/2 + 1 + (tile_x + 8)*F/2] = make_float2(temp82, temp83); \
	tt[index + tile_y/2 + 2 + (tile_x + 8)*F/2] = make_float2(temp84, temp85); \
	tt[index + tile_y/2 + 3 + (tile_x + 8)*F/2] = make_float2(temp86, temp87); \
	tt[index + tile_y/2 + 4 + (tile_x + 8)*F/2] = make_float2(temp88, temp89); \
	\
	tt[index + tile_y/2 + 0 + (tile_x + 9)*F/2] = make_float2(temp90, temp91); \
	tt[index + tile_y/2 + 1 + (tile_x + 9)*F/2] = make_float2(temp92, temp93); \
	tt[index + tile_y/2 + 2 + (tile_x + 9)*F/2] = make_float2(temp94, temp95); \
	tt[index + tile_y/2 + 3 + (tile_x + 9)*F/2] = make_float2(temp96, temp97); \
	tt[index + tile_y/2 + 4 + (tile_x + 9)*F/2] = make_float2(temp98, temp99); \
}\
while(0)
#define cudacall(call) \
    do\
    {\
	cudaError_t err = (call);\
	if(cudaSuccess != err)\
	    {\
		fprintf(stderr,"CUDA Error:\nFile = %s\nLine = %d\nReason = %s\n", __FILE__, __LINE__, cudaGetErrorString(err));\
		cudaDeviceReset();\
		exit(EXIT_FAILURE);\
	    }\
    }\
    while (0)\

#define cublascall(call) \
do\
{\
	cublasStatus_t status = (call);\
	if(CUBLAS_STATUS_SUCCESS != status)\
	{\
		fprintf(stderr,"CUBLAS Error:\nFile = %s\nLine = %d\nCode = %d\n", __FILE__, __LINE__, status);\
		cudaDeviceReset();\
		exit(EXIT_FAILURE);\
	}\
}\
while(0)\

#define cusparsecall(call) \
do\
{\
	cusparseStatus_t status = (call);\
	if(CUSPARSE_STATUS_SUCCESS != status)\
	{\
		fprintf(stderr,"CUSPARSE Error:\nFile = %s\nLine = %d\nCode = %d\n", __FILE__, __LINE__, status);\
		cudaDeviceReset();\
		exit(EXIT_FAILURE);\
	}\
}\
while(0)\

#define cudaCheckError() {                                          \
        cudaError_t e=cudaGetLastError();                                 \
        if(e!=cudaSuccess) {                                              \
            printf("CUDA failure %s:%d: '%s'\n",__FILE__,__LINE__,cudaGetErrorString(e));           \
            exit(EXIT_FAILURE);                                           \
        }                                                                 \
    }\
while(0)\

float doALS(const int* csrRowIndexHostPtr, const int* csrColIndexHostPtr, const float* csrValHostPtr,
		const int* cscRowIndexHostPtr, const int* cscColIndexHostPtr, const float* cscValHostPtr,
		const int* cooRowIndexHostPtr, float* thetaTHost, float * XTHost,
		const int * cooRowIndexTestHostPtr, const int * cooColIndexTestHostPtr, const float * cooValHostTestPtr,
		const int m, const int n, const int f, const long nnz, const long nnz_test, const float lambda,
		const int ITERS, const int X_BATCH, const int THETA_BATCH, const int DEVICEID);

#endif /* ALS_H_ */
