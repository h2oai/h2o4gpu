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
 * cg.h
 *
 *  Created on: July 23, 2016
 *      Author: weitan
 */

#ifndef CG_H_
#define CG_H_

int updateXWithCG(const int batchSize, const int batchOffset, float * ythetaT, float * tt, float * XT,
		cublasHandle_t handle, const int m, const int n, const int f, const int nnz);
		
void updateXWithCGHost(float * A, float * x, float * b, const int batchSize, const int f, const float cgIter);

void updateXWithCGHost_tt_fp16(float * A, float * x, float * b, const int batchSize, const int f, const float cgIter);

void alsUpdateFeature100Host(const int batch_offset,
		const int* csrRowIndex, const int* csrColIndex, const float lambda, const int m, const int F,
		const float* thetaT, float* XT, float* ythetaT, int cgIter);
#endif /* CG_H_ */
