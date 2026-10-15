#include <torch/extension.h>
#include "clm_kernels.h"
#include "ssim.h"
#include "adam.h"
#include "compute_sh_bwd.h"
#include "tile_contribution.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  // These audited entry points use only C++/ATen/CUDA, with per-call device
  // guards and the caller's current PyTorch stream; none calls back to Python.
  m.attr("PC1_NATIVE_ABI_VERSION") = 1;
  m.attr("PC1_NATIVE_BINDING") = "clm_kernels/ext.cpp";
  m.attr("PC1_SSIM_CURRENT_STREAM") = true;
  m.attr("PC1_GIL_RELEASED_OPS") = py::make_tuple(
      "fusedssim", "fusedssim_backward", "tile_contribution_mask");
  m.def("set_signal", &SetSignal);
  m.def("compute_cnt_h", &ComputeCntH);
  m.def("extract_ffs", &ExtractFFS);
  m.def("scatter_to_bit", &ScatterToBit);
  m.def("send_shs2gpu_stream", &SendSHS2GpuStreamCUDA);
  m.def("send_shs2cpu_grad_buffer_stream", &SendSHS2CpuGradBufferStreamCUDA);
  m.def("send_shs2gpu_stream_retention", &SendSHS2GpuStreamRetentionCUDA);
  m.def("send_shs2cpu_grad_buffer_stream_retention", &SendSHS2CpuGradBufferStreamRetentionCUDA);

  // Add SSIM functions
  m.def("fusedssim", &fusedssim, py::call_guard<py::gil_scoped_release>());
  m.def("fusedssim_backward", &fusedssim_backward,
        py::call_guard<py::gil_scoped_release>());

  m.def("selective_adam_update", &selective_adam_update);

  m.def("compute_sh_bwd_inplace", &compute_sh_bwd_inplace_tensor);
  m.def("tile_contribution_mask", &TileContributionMaskCUDA,
        py::call_guard<py::gil_scoped_release>());
}
