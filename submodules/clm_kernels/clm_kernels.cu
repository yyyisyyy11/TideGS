#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_runtime_api.h>


// CUDA kernel: gather selected spherical-harmonic coefficients from CPU memory in parallel, then transfer and reorder them in GPU memory.
// Thread: treat the 48 float coefficients of each spherical-harmonic feature as one unit.
__global__ void transfer_shs_cpu2gpu_kernel_stream(
    float *d_shs,
    const float *h_shs,
    const int64_t *rank2id, // rank2id[i]=1024 means selected point i has original-array ID 1024.
    int64_t num_select
) {
    int64_t stride = gridDim.x * blockDim.x;
    int64_t total_elements = num_select * 48;

    for (int64_t i = blockIdx.x * blockDim.x + threadIdx.x; i < total_elements; i += stride) {
        int64_t row = i / 48; // Determine which Gaussian owns this float; e.g. i=48 or i=95 -> row=1.
        int col = i % 48; // Parameter offset within the Gaussian, col=0~47.

        int64_t offset_srce = rank2id[row] * 48 + col; // Read parameter col of the source Gaussian and place it at GPU offset i.
        int64_t offset_dest = i;

        d_shs[offset_dest] = h_shs[offset_srce];
    }
}


// cuda kernel launcher function
void SendSHS2GpuStreamCUDA(
    torch::Tensor& d_parameters, // GPU tensor.
    torch::Tensor& h_parameters, // CPU tensor.
    torch::Tensor& mask_indices, // CPU mapping tensor, equivalent to rank2id.
    int grid_size,
    int block_size)
{
    int64_t N = h_parameters.size(0);
    int64_t num_select = mask_indices.size(0); // Number of Gaussians transferred in this batch.

    // Use PyTorch's current CUDA stream because PyTorch launches work asynchronously.
    // Using the same stream avoids data races and unnecessary CPU/GPU synchronization.
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream();

    // const int grid_size = 32;
    // const int grid_size = 16;
    // const int grid_size = 8;
    // const int block_size = 256;
    transfer_shs_cpu2gpu_kernel_stream<<<grid_size, block_size, 0, stream>>>(
        d_parameters.contiguous().data<float>(), // .data<float>() returns the pointer to the contiguous buffer.
        h_parameters.contiguous().data<float>(),
        mask_indices.contiguous().data<int64_t>(),
        num_select
    );
}

__global__ void transfer_H_shs_cpu2gpu_kernel_stream(
    float *d_shs,
    const float *h_shs,
    const int *host_indices, // Source Gaussian IDs in CPU memory.
    const int *param_indices_from_host, // Destination Gaussian slots in GPU memory.
    int num_select_from_host // Number of Gaussians transferred in this operation.
) { // GPU requests Gaussians updated by CPU Adam during microbatch pipelining.
    int stride = gridDim.x * blockDim.x;
    int total_elements = num_select_from_host * 48;

    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < total_elements; i += stride) {
        int row = i / 48;
        int col = i % 48;

        // size_t is an unsigned size type; it is 4 bytes on 32-bit systems and 8 bytes on 64-bit systems.
        size_t offset_srce = static_cast<size_t>(host_indices[row]) * 48 + col;
        size_t offset_dest = static_cast<size_t>(param_indices_from_host[row]) * 48 + col;

        d_shs[offset_dest] = h_shs[offset_srce];
    }
}

__global__ void transfer_D_shs_cpu2gpu_kernel_stream(
    float *d_shs,
    const float *r_shs,
    const int *host_indices,
    const int *param_indices_from_rtnt,
    int num_select_from_host
) {
    int stride = gridDim.x * blockDim.x;
    int total_elements = num_select_from_host * 48;

    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < total_elements; i += stride) {
        int row = i / 48;
        int col = i % 48;

        size_t offset_srce = static_cast<size_t>(host_indices[row]) * 48 + col;
        size_t offset_dest = static_cast<size_t>(param_indices_from_rtnt[row]) * 48 + col;

        d_shs[offset_dest] = r_shs[offset_srce];
    }
}

void SendSHS2GpuStreamRetentionCUDA(
    torch::Tensor& d_parameters,
    torch::Tensor& h_parameters,
    torch::Tensor& r_parameters, // on gpu, retent from last iteration
    torch::Tensor& host_indices,
    torch::Tensor& rtnt_indices,
    torch::Tensor& param_indices_from_host,
    torch::Tensor& param_indices_from_rtnt,
    int grid_size_H,
    int block_size_H,
    int grid_size_D,
    int block_size_D)
{ 
    int N = h_parameters.size(0); // number of all gaussians
    int num_select_from_host = host_indices.size(0);
    int num_select_from_retent = rtnt_indices.size(0);

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream();

    // Gaussians that are new in this inter-microbatch must be transferred from the CPU.
    transfer_H_shs_cpu2gpu_kernel_stream<<<grid_size_H, block_size_H, 0, stream>>>(
        d_parameters.contiguous().data<float>(),
        h_parameters.contiguous().data<float>(),
        host_indices.contiguous().data<int>(),
        param_indices_from_host.contiguous().data<int>(),
        num_select_from_host
    );

    // Gaussians retained across inter-microbatches are transferred from the retention buffer.
    transfer_D_shs_cpu2gpu_kernel_stream<<<grid_size_D, block_size_D, 0, stream>>>(
        d_parameters.contiguous().data<float>(),
        r_parameters.contiguous().data<float>(),
        rtnt_indices.contiguous().data<int>(),
        param_indices_from_rtnt.contiguous().data<int>(),
        num_select_from_retent
    );
}

__global__ void transfer_shsgrad_gpu2cpu_kernel_stream(
    const float *d_parameters,
    float *h_parameters,
    const int64_t *rank2id,
    const int64_t num_select,
    bool accum
) {
    int64_t stride = gridDim.x * blockDim.x;
    int64_t total_elements = num_select * 48;

    for (int64_t i = blockIdx.x * blockDim.x + threadIdx.x; i < total_elements; i += stride) {
        int64_t row = i / 48;
        int col = i % 48;

        int64_t offset_srce = i;
        int64_t offset_dest = rank2id[row] * 48 + col;

        if (accum) h_parameters[offset_dest] += d_parameters[offset_srce];
        else h_parameters[offset_dest] = d_parameters[offset_srce];
    }
}

void SendSHS2CpuGradBufferStreamCUDA(
    torch::Tensor& d_parameters,
    torch::Tensor& h_parameters,
    torch::Tensor& mask_indices,
    bool accum,
    int grid_size,
    int block_size)
{
    int64_t N = h_parameters.size(0);
    int64_t num_select = mask_indices.size(0);

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream();

    // const int grid_size = 32;
    // const int grid_size = 16;
    // const int grid_size = 8;
    // const int block_size = 256;

    transfer_shsgrad_gpu2cpu_kernel_stream<<<grid_size, block_size, 0, stream>>>(
        d_parameters.contiguous().data<float>(),
        h_parameters.contiguous().data<float>(),
        mask_indices.contiguous().data<int64_t>(),
        num_select,
        accum
    );
}

__global__ void transfer_H_shsgrad_gpu2cpu_kernel_stream(
    const float *d_shs,
    float *h_shs,
    const int *host_indices,
    const int *grad_indices,
    int num_select_from_host,
    bool accum
)
{
    int stride = gridDim.x * blockDim.x;
    int total_elements = num_select_from_host * 48;

    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < total_elements; i += stride) {
        int row = i / 48;
        int col = i % 48;

        size_t offset_host = static_cast<size_t>(host_indices[row]) * 48 + col;
        size_t offset_grad = static_cast<size_t>(grad_indices[row]) * 48 + col;

        float grad = d_shs[offset_grad];
        // if (std::fabs(grad) < 1e-25f){
        //     if (accum) h_shs[offset_host] += d_shs[offset_grad];
        //     else h_shs[offset_host] = d_shs[offset_grad];
        // }
        if (accum) h_shs[offset_host] += d_shs[offset_grad];
        else h_shs[offset_host] = d_shs[offset_grad];
    }
}

__global__ void transfer_D_shsgrad_gpu2cpu_kernel_stream(
    const float *d_shs,
    float *r_shs,
    const int *rtnt_indices,
    const int *grad_indices,
    int num_select_from_retent,
    bool accum
)
{
    int stride = gridDim.x * blockDim.x;
    int total_elements = num_select_from_retent * 48;

    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < total_elements; i += stride) {
        int row = i / 48;
        int col = i % 48;

        size_t offset_rtnt = static_cast<size_t>(rtnt_indices[row]) * 48 + col;
        size_t offset_grad = static_cast<size_t>(grad_indices[row]) * 48 + col;

        if (accum) r_shs[offset_rtnt] += d_shs[offset_grad];
        else r_shs[offset_rtnt] = d_shs[offset_grad];
    }
}

void SendSHS2CpuGradBufferStreamRetentionCUDA(
    torch::Tensor& d_parameters, // input
    torch::Tensor& h_parameters, // output 1
    torch::Tensor& r_parameters, // output 2, on gpu, gradients retent for next iteration
    torch::Tensor& host_indices,
    torch::Tensor& rtnt_indices,
    torch::Tensor& grad_indices_to_host,
    torch::Tensor& grad_indices_to_rtnt,
    bool accum,
    int grid_size_H,
    int block_size_H,
    int grid_size_D,
    int block_size_D
)
{
    int N = h_parameters.size(0); // number of all gaussians
    int num_select_from_host = host_indices.size(0);
    int num_select_from_retent = rtnt_indices.size(0);

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream();

    transfer_H_shsgrad_gpu2cpu_kernel_stream<<<grid_size_H, block_size_H, 0, stream>>>(
        d_parameters.contiguous().data<float>(),
        h_parameters.contiguous().data<float>(),
        host_indices.contiguous().data<int>(),
        grad_indices_to_host.contiguous().data<int>(),
        num_select_from_host,
        accum
    );

    transfer_D_shsgrad_gpu2cpu_kernel_stream<<<grid_size_D, block_size_D, 0, stream>>>(
        d_parameters.contiguous().data<float>(),
        r_parameters.contiguous().data<float>(),
        rtnt_indices.contiguous().data<int>(),
        grad_indices_to_rtnt.contiguous().data<int>(),
        num_select_from_retent,
        accum
    );
}

__global__ void set_signal_kernel(
    int* signal_tensor, // Pointer to the signal array.
    int microbatch_idx,
    int signal)
{
    __threadfence_system(); // Write data before the flag to ensure visibility.
    signal_tensor[microbatch_idx] = signal;
    __threadfence_system();
}

void SetSignal(torch::Tensor& signal_tensor, int microbatch_idx, int signal)
{
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    set_signal_kernel<<<1, 1, 0, stream>>>(
        reinterpret_cast<int*>(signal_tensor.contiguous().data_ptr()),
        microbatch_idx,
        signal
    );

}

template <typename T>
__global__ void extract_ffs_kernel(
    T* input,
    uint8_t *output,
    int N
)
{
    int stride = gridDim.x * blockDim.x;

    for (int i = threadIdx.x + blockIdx.x * blockDim.x; i < N; i += stride)
    {
        int casted = static_cast<int>(input[i]);
        output[i] = static_cast<uint8_t>(__ffs(casted));
    }
}

__global__ void extract_ffsll_kernel(
    uint64_t* input,
    uint8_t *output,
    int N
)
{
    int stride = gridDim.x * blockDim.x;

    for (int i = threadIdx.x + blockIdx.x * blockDim.x; i < N; i += stride)
    {
        int64_t casted = static_cast<int64_t>(input[i]);
        output[i] = static_cast<uint8_t>(__ffsll(casted));
    }
}

void ExtractFFS(
    torch::Tensor &input,
    torch::Tensor &output
)
{
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    int N = input.size(0);

    if (input.dtype() == torch::kInt64) {
        extract_ffsll_kernel<<<64, 256, 0, stream>>>(
            reinterpret_cast<uint64_t*>(input.contiguous().data_ptr()),
            reinterpret_cast<uint8_t*>(output.contiguous().data_ptr()),
            N
        );
    }
    else if (input.dtype() == torch::kInt32) {
        extract_ffs_kernel<uint32_t><<<64, 256, 0, stream>>>(
            reinterpret_cast<uint32_t*>(input.contiguous().data_ptr()),
            reinterpret_cast<uint8_t*>(output.contiguous().data_ptr()),
            N
        );
    }
    else if (input.dtype() == torch::kInt16) {
        extract_ffs_kernel<uint16_t><<<64, 256, 0, stream>>>(
            reinterpret_cast<uint16_t*>(input.contiguous().data_ptr()),
            reinterpret_cast<uint8_t*>(output.contiguous().data_ptr()),
            N
        );
    }
    else if (input.dtype() == torch::kInt8) {
        extract_ffs_kernel<uint8_t><<<64, 256, 0, stream>>>(
            reinterpret_cast<uint8_t*>(input.contiguous().data_ptr()),
            reinterpret_cast<uint8_t*>(output.contiguous().data_ptr()),
            N
        );
    }
    else AT_ERROR("`input` must have dtype (int8, int16, int32, int64).");

}

template <typename T>
__global__ void scatter_to_bit_kernel(
    T* bitmap,
    int64_t* filter,
    int bit,
    int nnz
)
{
    int stride = gridDim.x * blockDim.x;

    for (int i = threadIdx.x + blockIdx.x * blockDim.x; i < nnz; i += stride)
    {
        int64_t offset = filter[i];
        bitmap[offset] |= (uint64_t(1) << bit); //or 1LL
    }
}

void ScatterToBit(
    torch::Tensor &bitmap,
    torch::Tensor &filter,
    int bit
)
{
    cudaStream_t stream = at::cuda::getCurrentCUDAStream(); 
    int nnz = filter.size(0);

    if (bitmap.dtype() == torch::kInt64) {
        scatter_to_bit_kernel<uint64_t><<<64, 256, 0, stream>>>(
            reinterpret_cast<uint64_t*>(bitmap.contiguous().data_ptr()),
            reinterpret_cast<int64_t*>(filter.contiguous().data_ptr()),
            bit,
            nnz
        );
    }
    else if (bitmap.dtype() == torch::kInt32) {
        scatter_to_bit_kernel<uint32_t><<<64, 256, 0, stream>>>(
            reinterpret_cast<uint32_t*>(bitmap.contiguous().data_ptr()),
            reinterpret_cast<int64_t*>(filter.contiguous().data_ptr()),
            bit,
            nnz
        );
    }
    else if (bitmap.dtype() == torch::kInt16) {
        scatter_to_bit_kernel<uint16_t><<<64, 256, 0, stream>>>(
            reinterpret_cast<uint16_t*>(bitmap.contiguous().data_ptr()),
            reinterpret_cast<int64_t*>(filter.contiguous().data_ptr()),
            bit,
            nnz
        );
    }
    else if (bitmap.dtype() == torch::kInt8) {
        scatter_to_bit_kernel<uint8_t><<<64, 256, 0, stream>>>(
            reinterpret_cast<uint8_t*>(bitmap.contiguous().data_ptr()),
            reinterpret_cast<int64_t*>(filter.contiguous().data_ptr()),
            bit,
            nnz
        );
    }
    else AT_ERROR("`bitmap` must have dtype (int8, int16, int32, int64).");
}

__global__ void compute_cnt_h_64_kernel(
    uint64_t* bitmap,
    int* out_buffer,
    int N
)
{
    int stride = gridDim.x * blockDim.x;
    int reducer[63];

    #pragma unroll
    for (int i = 0; i < 63; i++) reducer[i] = 0;

    for (int i = threadIdx.x + blockIdx.x * blockDim.x; i < N; i += stride)
    {
        uint64_t t = bitmap[i];
        t = __brevll(t); // LSB: first micro batch; MSB: last micro batch.
        uint64_t overlap = t & (t >> 1);
        int count = __popcll(overlap);

        for (int j = 0; j < count; j++) {
            int pos = __ffsll(overlap);
            reducer[pos - 1] += 1;
            overlap &= overlap - 1; // Reset lowest set bit
        }
    }

    out_buffer = out_buffer + threadIdx.x + blockIdx.x * blockDim.x;
    #pragma unroll
    for (int i = 0; i < 63; i++) out_buffer[i * stride] = reducer[i];
}

__global__ void compute_cnt_h_32_kernel(
    uint32_t* bitmap,
    int* out_buffer,
    int N
)
{
    int stride = gridDim.x * blockDim.x;
    int reducer[31];

    #pragma unroll
    for (int i = 0; i < 31; i++) reducer[i] = 0;

    for (int i = threadIdx.x + blockIdx.x * blockDim.x; i < N; i += stride)
    {
        uint32_t t = bitmap[i];
        t = __brev(t); // LSB: first micro batch; MSB: last micro batch.
        uint32_t overlap = t & (t >> 1);
        int count = __popc(overlap);

        for (int j = 0; j < count; j++) {
            int pos = __ffs(overlap);
            reducer[pos - 1] += 1;
            overlap &= overlap - 1; // Reset lowest set bit
        }
    }

    out_buffer = out_buffer + threadIdx.x + blockIdx.x * blockDim.x;
    #pragma unroll
    for (int i = 0; i < 31; i++) out_buffer[i * stride] = reducer[i];
}

__global__ void compute_cnt_h_16_kernel(
    uint16_t* bitmap,
    int* out_buffer,
    int N
)
{
    int stride = gridDim.x * blockDim.x;
    int reducer[15];

    #pragma unroll
    for (int i = 0; i < 15; i++) reducer[i] = 0;

    for (int i = threadIdx.x + blockIdx.x * blockDim.x; i < N; i += stride)
    {
        uint32_t t = static_cast<uint32_t>(bitmap[i]);
        t = __brev(t) >> 16; // LSB: first micro batch; MSB: last micro batch.
        uint32_t overlap = t & (t >> 1);
        int count = __popc(overlap);

        for (int j = 0; j < count; j++) {
            int pos = __ffs(overlap);
            reducer[pos - 1] += 1;
            overlap &= overlap - 1; // Reset lowest set bit
        }
    }

    out_buffer = out_buffer + threadIdx.x + blockIdx.x * blockDim.x;
    #pragma unroll
    for (int i = 0; i < 15; i++) out_buffer[i * stride] = reducer[i];
}

__global__ void compute_cnt_h_8_kernel(
    uint8_t* bitmap,
    int* out_buffer,
    int N
)
{
    int stride = gridDim.x * blockDim.x;
    int reducer[7];

    #pragma unroll
    for (int i = 0; i < 7; i++) reducer[i] = 0;

    for (int i = threadIdx.x + blockIdx.x * blockDim.x; i < N; i += stride)
    {
        uint32_t t = static_cast<uint32_t>(bitmap[i]);
        t = __brev(t) >> 24; // LSB: first micro batch; MSB: last micro batch.
        uint32_t overlap = t & (t >> 1);
        int count = __popc(overlap);

        for (int j = 0; j < count; j++) {
            int pos = __ffs(overlap);
            reducer[pos - 1] += 1;
            overlap &= overlap - 1; // Reset lowest set bit
        }
    }

    out_buffer = out_buffer + threadIdx.x + blockIdx.x * blockDim.x;
    #pragma unroll
    for (int i = 0; i < 7; i++) out_buffer[i * stride] = reducer[i];
}

__global__ void compute_cnt_h_4_kernel(
    uint8_t* bitmap,
    int* out_buffer,
    int N
)
{
    int stride = gridDim.x * blockDim.x;
    int reducer[3];

    #pragma unroll
    for (int i = 0; i < 3; i++) reducer[i] = 0;

    for (int i = threadIdx.x + blockIdx.x * blockDim.x; i < N; i += stride)
    {
        uint32_t t = static_cast<uint32_t>(bitmap[i]);
        t = __brev(t) >> 28; // LSB: first micro batch; MSB: last micro batch.
        uint32_t overlap = t & (t >> 1);
        int count = __popc(overlap);

        for (int j = 0; j < count; j++) {
            int pos = __ffs(overlap);
            reducer[pos - 1] += 1;
            overlap &= overlap - 1; // Reset lowest set bit
        }
    }

    out_buffer = out_buffer + threadIdx.x + blockIdx.x * blockDim.x;
    #pragma unroll
    for (int i = 0; i < 3; i++) out_buffer[i * stride] = reducer[i];
}

void ComputeCntH(
    torch::Tensor &bitmap,
    torch::Tensor &tmp_buffer,
    int grid_size,
    int blk_size
)
{
    cudaStream_t stream = at::cuda::getCurrentCUDAStream(); 
    int N = bitmap.size(0);
    int bsz = tmp_buffer.size(0) + 1;

    if (bitmap.dtype() == torch::kInt64) {
        compute_cnt_h_64_kernel<<<grid_size, blk_size, 0, stream>>>(
            reinterpret_cast<uint64_t*>(bitmap.contiguous().data_ptr()),
            reinterpret_cast<int*>(tmp_buffer.contiguous().data_ptr()),
            N
        );
    }
    else if (bitmap.dtype() == torch::kInt32) {
        compute_cnt_h_32_kernel<<<grid_size, blk_size, 0, stream>>>(
            reinterpret_cast<uint32_t*>(bitmap.contiguous().data_ptr()),
            reinterpret_cast<int*>(tmp_buffer.contiguous().data_ptr()),
            N
        );
    }
    else if (bitmap.dtype() == torch::kInt16) {
        compute_cnt_h_16_kernel<<<grid_size, blk_size, 0, stream>>>(
            reinterpret_cast<uint16_t*>(bitmap.contiguous().data_ptr()),
            reinterpret_cast<int*>(tmp_buffer.contiguous().data_ptr()),
            N
        );
    }
    else if (bitmap.dtype() == torch::kInt8) {
        if (bsz == 4) {
            compute_cnt_h_4_kernel<<<grid_size, blk_size, 0, stream>>>(
                reinterpret_cast<uint8_t*>(bitmap.contiguous().data_ptr()),
                reinterpret_cast<int*>(tmp_buffer.contiguous().data_ptr()),
                N
            );
        }
        else {
            compute_cnt_h_8_kernel<<<grid_size, blk_size, 0, stream>>>(
                reinterpret_cast<uint8_t*>(bitmap.contiguous().data_ptr()),
                reinterpret_cast<int*>(tmp_buffer.contiguous().data_ptr()),
                N
            );
        }
    }
    else AT_ERROR("`reset_col_gathered` must have dtype (int8, int16, int32, int64).");
}
