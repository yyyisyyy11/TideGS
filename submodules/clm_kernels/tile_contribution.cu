#include "tile_contribution.h"

#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <limits>

namespace {

__device__ __forceinline__ float clamp_value(float value, float lower, float upper) {
    return fminf(fmaxf(value, lower), upper);
}

__device__ __forceinline__ int clamp_int(int value, int lower, int upper) {
    return value < lower ? lower : (value > upper ? upper : value);
}

__device__ __forceinline__ float quadratic(
    float x, float y, float a, float b, float c) {
    return a * x * x + 2.0f * b * x * y + c * y * y;
}

__device__ float rectangle_min_quadratic(
    float x0, float x1, float y0, float y1, float a, float b, float c) {
    float best = CUDART_INF_F;
    if (x0 <= 0.0f && 0.0f <= x1 && y0 <= 0.0f && 0.0f <= y1) {
        return 0.0f;
    }

    best = fminf(best, quadratic(x0, y0, a, b, c));
    best = fminf(best, quadratic(x0, y1, a, b, c));
    best = fminf(best, quadratic(x1, y0, a, b, c));
    best = fminf(best, quadratic(x1, y1, a, b, c));

    if (c > 0.0f) {
        float y = clamp_value(-b * x0 / c, y0, y1);
        best = fminf(best, quadratic(x0, y, a, b, c));
        y = clamp_value(-b * x1 / c, y0, y1);
        best = fminf(best, quadratic(x1, y, a, b, c));
    }
    if (a > 0.0f) {
        float x = clamp_value(-b * y0 / a, x0, x1);
        best = fminf(best, quadratic(x, y0, a, b, c));
        x = clamp_value(-b * y1 / a, x0, x1);
        best = fminf(best, quadratic(x, y1, a, b, c));
    }
    return fmaxf(best, 0.0f);
}

__global__ void tile_contribution_mask_kernel(
    const float *__restrict__ means2d,
    const float *__restrict__ conics,
    const float *__restrict__ opacities,
    const int32_t *__restrict__ radii,
    int64_t count,
    int radius_dims,
    int image_width,
    int image_height,
    int tile_size,
    float alpha_threshold,
    bool *__restrict__ keep,
    int32_t *__restrict__ candidate_counts,
    int32_t *__restrict__ contributing_counts) {
    int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= count) {
        return;
    }

    const int32_t radius_x = radii[index * radius_dims];
    const int32_t radius_y = radii[index * radius_dims + (radius_dims == 2)];
    if (radius_x <= 0 || radius_y <= 0) {
        keep[index] = false;
        candidate_counts[index] = 0;
        contributing_counts[index] = 0;
        return;
    }

    const float mean_x = means2d[index * 2];
    const float mean_y = means2d[index * 2 + 1];
    const int tile_width = (image_width + tile_size - 1) / tile_size;
    const int tile_height = (image_height + tile_size - 1) / tile_size;
    if (!isfinite(mean_x) || !isfinite(mean_y)) {
        const int32_t all_tiles = tile_width * tile_height;
        keep[index] = true;
        candidate_counts[index] = all_tiles;
        contributing_counts[index] = all_tiles;
        return;
    }
    const int min_x = clamp_int(static_cast<int>(floorf(
        (mean_x - static_cast<float>(radius_x)) / tile_size)), 0, tile_width);
    const int min_y = clamp_int(static_cast<int>(floorf(
        (mean_y - static_cast<float>(radius_y)) / tile_size)), 0, tile_height);
    const int max_x = clamp_int(static_cast<int>(ceilf(
        (mean_x + static_cast<float>(radius_x)) / tile_size)), 0, tile_width);
    const int max_y = clamp_int(static_cast<int>(ceilf(
        (mean_y + static_cast<float>(radius_y)) / tile_size)), 0, tile_height);

    const int candidate_width = max_x > min_x ? max_x - min_x : 0;
    const int candidate_height = max_y > min_y ? max_y - min_y : 0;
    const int32_t candidates = candidate_width * candidate_height;
    candidate_counts[index] = candidates;
    if (candidates == 0) {
        keep[index] = false;
        contributing_counts[index] = 0;
        return;
    }

    const float opacity = opacities[index];
    if (!isfinite(opacity)) {
        keep[index] = true;
        contributing_counts[index] = candidates;
        return;
    }
    if (!(opacity >= alpha_threshold)) {
        keep[index] = false;
        contributing_counts[index] = 0;
        return;
    }

    const float a = conics[index * 3];
    const float b = conics[index * 3 + 1];
    const float c = conics[index * 3 + 2];
    if (!isfinite(a) || !isfinite(b) || !isfinite(c)) {
        keep[index] = true;
        contributing_counts[index] = candidates;
        return;
    }
    int32_t contributing = 0;
    for (int tile_y = min_y; tile_y < max_y; ++tile_y) {
        const int pixel_y0 = tile_y * tile_size;
        const int pixel_y1 =
            ((tile_y + 1) * tile_size < image_height
                 ? (tile_y + 1) * tile_size
                 : image_height) - 1;
        const float y0 = static_cast<float>(pixel_y0) + 0.5f - mean_y;
        const float y1 = static_cast<float>(pixel_y1) + 0.5f - mean_y;
        for (int tile_x = min_x; tile_x < max_x; ++tile_x) {
            const int pixel_x0 = tile_x * tile_size;
            const int pixel_x1 =
                ((tile_x + 1) * tile_size < image_width
                     ? (tile_x + 1) * tile_size
                     : image_width) - 1;
            const float x0 = static_cast<float>(pixel_x0) + 0.5f - mean_x;
            const float x1 = static_cast<float>(pixel_x1) + 0.5f - mean_x;
            const float power = 0.5f * rectangle_min_quadratic(
                x0, x1, y0, y1, a, b, c);
            if (opacity * __expf(-power) >= alpha_threshold) {
                ++contributing;
            }
        }
    }
    contributing_counts[index] = contributing;
    keep[index] = contributing > 0;
}

void check_cuda_contiguous(const torch::Tensor &value, const char *name) {
    TORCH_CHECK(value.is_cuda(), name, " must be a CUDA tensor");
    TORCH_CHECK(value.is_contiguous(), name, " must be contiguous");
}

}  // namespace

std::vector<torch::Tensor> TileContributionMaskCUDA(
    const torch::Tensor &means2d,
    const torch::Tensor &conics,
    const torch::Tensor &opacities,
    const torch::Tensor &radii,
    int64_t image_width,
    int64_t image_height,
    int64_t tile_size,
    double alpha_threshold) {
    check_cuda_contiguous(means2d, "means2d");
    check_cuda_contiguous(conics, "conics");
    check_cuda_contiguous(opacities, "opacities");
    check_cuda_contiguous(radii, "radii");
    TORCH_CHECK(means2d.scalar_type() == torch::kFloat32, "means2d must be float32");
    TORCH_CHECK(conics.scalar_type() == torch::kFloat32, "conics must be float32");
    TORCH_CHECK(opacities.scalar_type() == torch::kFloat32, "opacities must be float32");
    TORCH_CHECK(radii.scalar_type() == torch::kInt32, "radii must be int32");
    TORCH_CHECK(means2d.dim() >= 1 && means2d.size(-1) == 2,
                "means2d must end in dimension 2");
    TORCH_CHECK(conics.dim() >= 1 && conics.size(-1) == 3,
                "conics must end in dimension 3");
    const int64_t count = means2d.numel() / 2;
    TORCH_CHECK(conics.numel() == count * 3, "conics shape does not match means2d");
    TORCH_CHECK(opacities.numel() == count, "opacities shape does not match means2d");
    TORCH_CHECK(count == 0 || radii.numel() % count == 0, "invalid radii shape");
    const int radius_dims = count == 0 ? 1 : static_cast<int>(radii.numel() / count);
    TORCH_CHECK(radius_dims == 1 || radius_dims == 2, "radii must have one or two values per projection");
    TORCH_CHECK(image_width > 0 && image_height > 0, "image dimensions must be positive");
    TORCH_CHECK(tile_size > 0, "tile_size must be positive");
    TORCH_CHECK(image_width <= std::numeric_limits<int>::max() &&
                    image_height <= std::numeric_limits<int>::max() &&
                    tile_size <= std::numeric_limits<int>::max(),
                "image dimensions and tile_size must fit int32");
    TORCH_CHECK(std::isfinite(alpha_threshold) && alpha_threshold > 0.0 &&
                    alpha_threshold <= 1.0 / 255.0,
                "alpha_threshold must be in (0, 1/255]");
    TORCH_CHECK(means2d.device() == conics.device() && means2d.device() == opacities.device() &&
                    means2d.device() == radii.device(),
                "all inputs must be on the same CUDA device");

    c10::cuda::CUDAGuard guard(means2d.device());
    auto keep = torch::empty({count}, means2d.options().dtype(torch::kBool));
    auto candidates = torch::empty({count}, means2d.options().dtype(torch::kInt32));
    auto contributing = torch::empty({count}, means2d.options().dtype(torch::kInt32));
    if (count > 0) {
        constexpr int threads = 256;
        const int blocks = static_cast<int>((count + threads - 1) / threads);
        tile_contribution_mask_kernel<<<
            blocks, threads, 0, c10::cuda::getCurrentCUDAStream()>>>(
            means2d.data_ptr<float>(),
            conics.data_ptr<float>(),
            opacities.data_ptr<float>(),
            radii.data_ptr<int32_t>(),
            count,
            radius_dims,
            static_cast<int>(image_width),
            static_cast<int>(image_height),
            static_cast<int>(tile_size),
            static_cast<float>(alpha_threshold),
            keep.data_ptr<bool>(),
            candidates.data_ptr<int32_t>(),
            contributing.data_ptr<int32_t>());
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
    return {keep, candidates, contributing};
}
