// Copyright (c) Microsoft Corporation. All rights reserved.
// Copyright (c) 2019-2020, NXP Semiconductor, Inc. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include "core/framework/op_kernel.h"
#include "core/providers/cpu/nn/conv.h"
#include "core/providers/ketinpu/ketinpu_execution_provider.h"


namespace onnxruntime {
namespace ketinpu {

template <typename T>
class Conv : public onnxruntime::Conv<T> {
 public:
  explicit Conv(const OpKernelInfo& info):  onnxruntime::Conv<T>(info), conv_attrs_(info) {
  }

 public:
  Status Compute(OpKernelContext* context) const override;
 protected:
  ConvAttributes conv_attrs_;


};

}  // namespace ketinpu
}  // namespace onnxruntime
