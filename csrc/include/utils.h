#pragma once
#include <torch/extension.h>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)


//packbits_32

torch::Tensor packbits_u32(torch::Tensor idx_array, torch::Tensor bits_array);

torch::Tensor un_packbits_u32(torch::Tensor idx_array, torch::Tensor bits_array);

torch::Tensor packbits_u32_cu(
    torch::Tensor idx_array,
    torch::Tensor bits_array
);

torch::Tensor un_packbits_u32_cu(
    torch::Tensor idx_array,
    torch::Tensor bits_array
);

torch::Tensor distance_mask_cu(
    torch::Tensor dirsMap,//[WxH , 3]
    torch::Tensor locMap,//[WxH , 3]
    torch::Tensor centroids,//[C , 3]
    torch::Tensor mask,//[WxH , C]
    const float threshould//重叠阈值
);
torch::Tensor mega_nerf_mask_cu(
    torch::Tensor dirsMap,//[WxH , 3]
    torch::Tensor locMap,//[WxH , 3]
    torch::Tensor centroids,//[C , 3]
    torch::Tensor t_range,
    const int samples,//每条射线上采样点数
    const float threshould//重叠阈值
);