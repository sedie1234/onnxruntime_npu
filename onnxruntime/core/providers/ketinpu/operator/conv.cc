// Copyright (c) Microsoft Corporation. All rights reserved.
// Copyright (c) 2019-2020, NXP Semiconductor, Inc. All rights reserved.
// Licensed under the MIT License.

#ifdef _WIN32
#pragma warning(disable : 4244)
#endif
#include <thread>
#include <mutex>

#include "core/common/common.h"
#include "core/framework/op_kernel.h"
#include "core/util/math.h"
//#include "core/util/math_cpuonly.h"

#include "core/providers/ketinpu/operator/conv.h"
#include "core/providers/ketinpu/ketinpu_fwd.h"

namespace onnxruntime {
namespace ketinpu {

template <typename T>
Status Conv<T>::Compute(OpKernelContext* context) const {
  size_t num_inputs = OpKernel::Node().InputDefs().size();

  const Tensor* X = context->Input<Tensor>(0);
  const Tensor* W = context->Input<Tensor>(1);
  const Tensor* B = num_inputs == 3 ? context->Input<Tensor>(2) : nullptr;

  std::cout << "Conv(Keti npu)" << std::endl;

  LOGS_DEFAULT(VERBOSE) << "X " << X->Shape().ToString().c_str();
  LOGS_DEFAULT(VERBOSE) << "W " << W->Shape().ToString().c_str();
#if 1
  if (B != nullptr)
    LOGS_DEFAULT(VERBOSE) << "B " << B->Shape().ToString().c_str();
  else
    LOGS_DEFAULT(VERBOSE) << "B is nullptr"; 
#endif

  LOGS_DEFAULT(VERBOSE) << std::endl;

  Status s = onnxruntime::Conv<T>::Compute(context);
  return s;
}

ONNX_OPERATOR_KERNEL_EX(
    Conv,
    kOnnxDomain,
    1,
    kKetinpuExecutionProvider,
    KernelDefBuilder().TypeConstraint("T", DataTypeImpl::GetTensorType<float>()),
    Conv<float>);

}  // namespace ketinpu
}  // namespace onnxruntime
