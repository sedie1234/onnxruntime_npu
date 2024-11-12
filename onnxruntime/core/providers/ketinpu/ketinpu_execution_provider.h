#pragma once

#include "core/framework/allocatormgr.h"
#include "core/framework/execution_provider.h"
#include "core/graph/constants.h"

namespace onnxruntime {

// Information needed to construct Ketinpu execution providers.
struct KetinpuExecutionProviderInfo {
  bool create_arena{true};

  explicit KetinpuExecutionProviderInfo(bool use_arena)
      : create_arena(use_arena) {}

  KetinpuExecutionProviderInfo() = default;
};

// Logical device representation.
class KetinpuExecutionProvider : public IExecutionProvider {
 public:
  explicit KetinpuExecutionProvider(const KetinpuExecutionProviderInfo& info);
  virtual ~KetinpuExecutionProvider();

  std::vector<std::unique_ptr<ComputeCapability>> GetCapability(
      const onnxruntime::GraphViewer& graph,
      const std::vector<const KernelRegistry*>& kernel_registries) const override;

  const void* GetExecutionHandle() const noexcept override {
    // The Ketinpu interface does not return anything interesting.
    return nullptr;
  }

  std::shared_ptr<KernelRegistry> GetKernelRegistry() const override;
};

}  // namespace onnxruntime
