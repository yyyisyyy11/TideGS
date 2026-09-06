#pragma once

#include <torch/extension.h>

#include <vector>

std::vector<torch::Tensor> TileContributionMaskCUDA(
    const torch::Tensor &means2d,
    const torch::Tensor &conics,
    const torch::Tensor &opacities,
    const torch::Tensor &radii,
    int64_t image_width,
    int64_t image_height,
    int64_t tile_size,
    double alpha_threshold);
