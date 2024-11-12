// Copyright (c) Microsoft Corporation. All rights reserved.
// Copyright (c) 2019-2020, NXP Semiconductor, Inc. All rights reserved.
// Licensed under the MIT License.

#include "ketinpu_execution_provider.h"
#include "core/framework/allocator.h"
#include "core/framework/op_kernel.h"
#include "core/framework/kernel_registry.h"
#include "core/framework/compute_capability.h"
#include "contrib_ops/cpu/cpu_contrib_kernels.h"
#include "ketinpu_fwd.h"

#include "core/framework/allocatormgr.h"
#include "core/framework/memcpy.h"


namespace onnxruntime {

constexpr const char* KETINPU = "Ketinpu";
constexpr const char* KETINPU_CPU = "KetinpuCpu";

namespace ketinpu {

// Forward declarations of op kernels
class ONNX_OPERATOR_KERNEL_CLASS_NAME(kKetinpuExecutionProvider, kOnnxDomain, 1, Conv);
class ONNX_OPERATOR_KERNEL_CLASS_NAME(kKetinpuExecutionProvider, kOnnxDomain, 10, ConvInteger);


ONNX_OPERATOR_KERNEL_EX(
    MemcpyFromHost,
    kOnnxDomain,
    1,
    kKetinpuExecutionProvider,
    KernelDefBuilder()
        .InputMemoryType(OrtMemTypeCPUInput, 0)
        .TypeConstraint("T", DataTypeImpl::AllFixedSizeTensorTypes()),
    Memcpy);

ONNX_OPERATOR_KERNEL_EX(
    MemcpyToHost,
    kOnnxDomain,
    1,
    kKetinpuExecutionProvider,
    KernelDefBuilder()
        .OutputMemoryType(OrtMemTypeCPUOutput, 0)
        .TypeConstraint("T", DataTypeImpl::AllFixedSizeTensorTypes()),
    Memcpy);

Status RegisterKetinpuKernels(KernelRegistry& kernel_registry) {
  static const BuildKernelCreateInfoFn function_table[] = {
      //BuildKernelCreateInfo<void>,  //default entry to avoid the list become empty after ops-reducing
      BuildKernelCreateInfo<ONNX_OPERATOR_KERNEL_CLASS_NAME(kKetinpuExecutionProvider, kOnnxDomain, 1, Conv)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_KERNEL_CLASS_NAME(kKetinpuExecutionProvider, kOnnxDomain, 1, MemcpyFromHost)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_KERNEL_CLASS_NAME(kKetinpuExecutionProvider, kOnnxDomain, 1, MemcpyToHost)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_KERNEL_CLASS_NAME(kKetinpuExecutionProvider, kOnnxDomain, 10, ConvInteger)>,


  };

  for (auto& function_table_entry : function_table) {
    KernelCreateInfo info = function_table_entry();
    if (info.kernel_def != nullptr) {  // filter disabled entries where type is void
      ORT_RETURN_IF_ERROR(kernel_registry.Register(std::move(info)));
    }
  }

  return Status::OK();
}

std::shared_ptr<KernelRegistry> GetKetinpuKernelRegistry() {
  std::shared_ptr<KernelRegistry> kernel_registry = std::make_shared<KernelRegistry>();
  ORT_THROW_IF_ERROR(RegisterKetinpuKernels(*kernel_registry));
  return kernel_registry;
}

}  // namespace ketinpu

KetinpuExecutionProvider::KetinpuExecutionProvider(const KetinpuExecutionProviderInfo& info)
    : IExecutionProvider{onnxruntime::kKetinpuExecutionProvider} {
  ORT_UNUSED_PARAMETER(info);

  AllocatorCreationInfo default_memory_info{
      [](int) {
        return std::make_unique<CPUAllocator>(OrtMemoryInfo(KETINPU, OrtAllocatorType::OrtDeviceAllocator));
      },
      0,
      info.create_arena};

  InsertAllocator(CreateAllocator(default_memory_info));

  AllocatorCreationInfo cpu_memory_info{
      [](int) {
        return std::make_unique<CPUAllocator>(
            OrtMemoryInfo(KETINPU_CPU, OrtAllocatorType::OrtDeviceAllocator, OrtDevice(), 0, OrtMemTypeCPUOutput));
      },
      0,
      info.create_arena};

  InsertAllocator(CreateAllocator(cpu_memory_info));
}

KetinpuExecutionProvider::~KetinpuExecutionProvider() {
}

std::shared_ptr<KernelRegistry> KetinpuExecutionProvider::GetKernelRegistry() const {
  static std::shared_ptr<KernelRegistry> kernel_registry = onnxruntime::ketinpu::GetKetinpuKernelRegistry();
  return kernel_registry;
}

std::vector<std::unique_ptr<ComputeCapability>>
KetinpuExecutionProvider::GetCapability(const onnxruntime::GraphViewer& graph,
                                    const std::vector<const KernelRegistry*>& kernel_registries) const {
  std::vector<std::unique_ptr<ComputeCapability>>
      result = IExecutionProvider::GetCapability(graph, kernel_registries);

  return result;
}

}  // namespace onnxruntime
