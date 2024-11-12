// Copyright (c) 2020, keti smart lab.

#include "core/providers/ketinpu/ketinpu_provider_factory.h"
#include "ketinpu_execution_provider.h"
#include "core/session/abi_session_options_impl.h"

namespace onnxruntime {

struct KetinpuProviderFactory : IExecutionProviderFactory {
  KetinpuProviderFactory(bool create_arena) : create_arena_(create_arena) {}
  ~KetinpuProviderFactory() override {}
  std::unique_ptr<IExecutionProvider> CreateProvider() override;

 private:
  bool create_arena_;
};

std::unique_ptr<IExecutionProvider> KetinpuProviderFactory::CreateProvider() {
  KetinpuExecutionProviderInfo info;
  info.create_arena = create_arena_;
  return std::make_unique<KetinpuExecutionProvider>(info);
}

std::shared_ptr<IExecutionProviderFactory> CreateExecutionProviderFactory_Ketinpu(int use_arena) {
  return std::make_shared<onnxruntime::KetinpuProviderFactory>(use_arena != 0);
}

}  // namespace onnxruntime

ORT_API_STATUS_IMPL(OrtSessionOptionsAppendExecutionProvider_Ketinpu, _In_ OrtSessionOptions* options, int use_arena) {
  options->provider_factories.push_back(onnxruntime::CreateExecutionProviderFactory_Ketinpu(use_arena));
  return nullptr;
}
