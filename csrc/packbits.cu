#include "utils.h"
#include <thrust/execution_policy.h>
#include <thrust/scan.h>
#include <cmath>


__global__ void packbits_u32_kernel(
    torch::PackedTensorAccessor32<int32_t,1,torch::RestrictPtrTraits> idx_array,
    torch::PackedTensorAccessor64<int64_t,1,torch::RestrictPtrTraits> bits_array
){
    // const int32_t n = blockIdx.x * blockDim.x + threadIdx.x;//一维时
    const int32_t n = blockIdx.x*blockDim.x*blockDim.y + threadIdx.x*blockDim.y + threadIdx.y;//二维时
    if(n > bits_array.size(0))
        return;
    int mask_size = 32;
    if (n == bits_array.size(0))
        mask_size = (idx_array.size(0) % 32) - 1;
    const int64_t flag = 1;
    for(int i = 0 ; i < mask_size ; i++){
        int32_t hit_pix = idx_array[n*32 + i];
        if (hit_pix > 0){
            bits_array[n] |= flag << i;
        }
    }
}


torch::Tensor packbits_u32_cu(
    torch::Tensor idx_array,
    torch::Tensor bits_array
){
    // 每个线程处理32位长数据即32个像素
    const int num_pixs = std::ceil(idx_array.size(0)/32);
    // const int threads = 256, blocks = (num_pixs+threads-1)/threads;
    const int BLOCK_W = 64;
    const int BLOCK_H = 16;
    const dim3 blockSize(BLOCK_W,BLOCK_H,1);
    const dim3 gridSize((num_pixs + BLOCK_W*BLOCK_H - 1)/(BLOCK_W * BLOCK_H),1,1);
    // const dim3 gridSize(8,1,1);

    // torch::Tensor bit_array = torch::zeros({bits_array.size(0)},bits_array.options());
    AT_DISPATCH_ALL_TYPES(idx_array.type(),"packbits_u32_cu",
    // AT_DISPATCH_ALL_TYPES(idx_array.type(),"packbits_u64_cu",
    ([&] {
        packbits_u32_kernel<<<gridSize, blockSize>>>(
        // packbits_u64_kernel<<<4, 64>>>(
            idx_array.packed_accessor32<int32_t,1,torch::RestrictPtrTraits>(),
            bits_array.packed_accessor64<int64_t,1,torch::RestrictPtrTraits>()
        );
    }));
    return bits_array;
}

__global__ void un_packbits_u32_kernel(
    torch::PackedTensorAccessor32<int32_t,1,torch::RestrictPtrTraits> idx_array,
    torch::PackedTensorAccessor64<int64_t,1,torch::RestrictPtrTraits> bits_array
){
    // const int32_t n = blockIdx.x * blockDim.x + threadIdx.x;//一维时
    const int32_t n = blockIdx.x*blockDim.x*blockDim.y + threadIdx.x*blockDim.y + threadIdx.y;//二维时

    if(n > bits_array.size(0))
        return;
    int mask_size = 32;
    if (n == bits_array.size(0))
        mask_size = (idx_array.size(0) % 32) - 1;
    const int64_t flag = 1;
    for(int i = 0 ; i < mask_size ; i++){
        if (bits_array[n] & (flag << i)){
            idx_array[n*32 + i]++;
        }
    }
}
torch::Tensor un_packbits_u32_cu(
    torch::Tensor idx_array,
    torch::Tensor bits_array
){
    // 每个线程处理64位长数据即64个像素
    const int num_pixs = std::ceil(idx_array.size(0)/64);
    // const int threads = 256, blocks = (num_pixs+threads-1)/threads;
    const int BLOCK_W = 64;
    const int BLOCK_H = 16;
    const dim3 blockSize(BLOCK_W,BLOCK_H,1);
    const dim3 gridSize((num_pixs + BLOCK_W*BLOCK_H - 1)/(BLOCK_W * BLOCK_H),1,1);

    AT_DISPATCH_ALL_TYPES(idx_array.type(),"un_packbits_u32_cu",
    ([&] {
        un_packbits_u32_kernel<<<gridSize, blockSize>>>(
            idx_array.packed_accessor32<int32_t,1,torch::RestrictPtrTraits>(),
            bits_array.packed_accessor64<int64_t,1,torch::RestrictPtrTraits>()
        );
    }));
    return idx_array;
}

