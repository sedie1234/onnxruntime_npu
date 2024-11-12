// Copyright (c) Microsoft Corporation. All rights reserved.
// Copyright (c) 2019-2020, NXP Semiconductor, Inc. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include "core/framework/op_kernel.h"
#include "core/providers/cpu/nn/conv_attributes.h"
#include "core/providers/ketinpu/ketinpu_execution_provider.h"


namespace onnxruntime {

class ConvInteger : public OpKernel {
 public:
  explicit ConvInteger(const OpKernelInfo& info) : OpKernel(info), conv_attrs_(info) {}

  Status Compute(OpKernelContext* context) const override;

  ConvAttributes conv_attrs_;
};

namespace ketinpu {

class ConvInteger : public onnxruntime::ConvInteger {
 public:
  explicit ConvInteger(const OpKernelInfo& info):  onnxruntime::ConvInteger(info), conv_attrs_(info) {
  }

 public:
  Status Compute(OpKernelContext* context) const override;
 protected:
  ConvAttributes conv_attrs_;


};

}  // namespace ketinpu
}  // namespace onnxruntime
