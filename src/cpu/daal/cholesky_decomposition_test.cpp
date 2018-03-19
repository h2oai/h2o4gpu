/*******************************************************************************
!  Copyright(C) 2014-2017 Intel Corporation. All Rights Reserved.
!
!  The source code, information  and  material ("Material") contained herein is
!  owned  by Intel Corporation or its suppliers or licensors, and title to such
!  Material remains  with Intel Corporation  or its suppliers or licensors. The
!  Material  contains proprietary information  of  Intel or  its  suppliers and
!  licensors. The  Material is protected by worldwide copyright laws and treaty
!  provisions. No  part  of  the  Material  may  be  used,  copied, reproduced,
!  modified, published, uploaded, posted, transmitted, distributed or disclosed
!  in any way  without Intel's  prior  express written  permission. No  license
!  under  any patent, copyright  or  other intellectual property rights  in the
!  Material  is  granted  to  or  conferred  upon  you,  either  expressly,  by
!  implication, inducement,  estoppel or  otherwise.  Any  license  under  such
!  intellectual  property  rights must  be express  and  approved  by  Intel in
!  writing.
!
!  *Third Party trademarks are the property of their respective owners.
!
!  Unless otherwise  agreed  by Intel  in writing, you may not remove  or alter
!  this  notice or  any other notice embedded  in Materials by Intel or Intel's
!  suppliers or licensors in any way.
!
!*******************************************************************************
!  Content:
!    Cholesky decomposition sample program.
!******************************************************************************/

#include "daal.h"
#include <iostream>

using namespace daal;
using namespace daal::algorithms;
using namespace daal::data_management;
using namespace daal::services;

const size_t dimension = 3;
double inputArray[dimension *dimension] =
{
    1.0,  2.0,  4.0,
    2.0, 13.0, 23.0,
    4.0, 23.0, 77.0
};

int main(int argc, char *argv[])
{
    /* Create input numeric table from array */
    SharedPtr<NumericTable> inputData = SharedPtr<NumericTable>(new Matrix<double>(dimension, dimension, inputArray));

    /*  Create the algorithm object for computation of the Cholesky decomposition using the default method */
    cholesky::Batch<> algorithm;

    /* Set input for the algorithm */
    algorithm.input.set(cholesky::data, inputData);

    /* Compute Cholesky decomposition */
    algorithm.compute();

    /* Get pointer to Cholesky factor */
    SharedPtr<Matrix<double> > factor =
        staticPointerCast<Matrix<double>, NumericTable>(algorithm.getResult()->get(cholesky::choleskyFactor));

    /* Print the first element of the Cholesky factor */
    std::cout << "The first element of the Cholesky factor: " << (*factor)[0][0];

    return 0;
}


