
#include <torch/extension.h>
#include <iostream>
#include <ATen/ATen.h>
#include <math.h>
#include <ATen/TensorAccessor.h>
#include <ATen/cuda/CUDAContext.h>
#include <torch/extension.h>
#include "helper_math.h"
using namespace std;

__global__ void distance_mask_kernel(
    torch::PackedTensorAccessor32<float,2,torch::RestrictPtrTraits> dirsMap,//视线方向 [N 3]
    torch::PackedTensorAccessor32<float,1,torch::RestrictPtrTraits> locMap,//相机光心 [3]
    torch::PackedTensorAccessor32<float,2,torch::RestrictPtrTraits> centroids,//分块质心 [C 3]
    torch::PackedTensorAccessor32<int32_t,2,torch::RestrictPtrTraits> mask,//一张图像对于C个质心的mask值 [N , C]

    const float threshould//重叠阈值
){
    // const int threadId_2D = threadIdx.x + threadIdx.y*blockDim.x;
    // const int blockId_2D = blockIdx.x + blockIdx.y*gridDim.x;
    // const int n = threadId_2D + (blockDim.x*blockDim.y)*blockId_2D;
    const int32_t n = blockIdx.x*blockDim.x*blockDim.y + threadIdx.x*blockDim.y + threadIdx.y;//二维时
    if(n >= dirsMap.size(0))
        return;
    const float dx = dirsMap[n][0], dy = dirsMap[n][1], dz = dirsMap[n][2];
    const float ox = locMap[0], oy = locMap[1], oz = locMap[2];
    float3 dir = make_float3(dx,dy,dz);
    float3 loc = make_float3(ox,oy,oz);
    float* d = new float[centroids.size(0)];
    float d_min = 99999.9;
    
    dir /= length(dir);

    for(int i  = 0; i < centroids.size(0); i++){
        float3 centroid = make_float3(centroids[i][0],centroids[i][1],centroids[i][2]); 
        float3 l_vec = centroid - loc;
        float3 d_vec = cross(l_vec,dir);
        d[i] = length(d_vec);
        if (d_min >= d[i])
            d_min = d[i];

        // \  d   |
        //  *-----|
        //   \    |
        // l  \   | dir   d = |(l X dir)|/|dir|
        //     \  |
        //      \ |
        //       \|

    }
    
    for(int i  = 0; i < centroids.size(0); i++){
        if (d[i] <= (d_min * threshould))
            mask[n][i] = 1;
    }
    delete d;
    d = nullptr;
}


torch::Tensor distance_mask_cu(
    torch::Tensor dirsMap,//[WxH , 3]
    torch::Tensor locMap,//[WxH , 3]
    torch::Tensor centroids,//[C , 3]
    torch::Tensor mask,//[WxH , C]
    const float threshould//重叠阈值
){
    const int num_pisxels = dirsMap.size(0);
    const int BLOCK_W = 64;
    const int BLOCK_H = 16;
    const dim3 blockSize(BLOCK_W,BLOCK_H,1);
    const dim3 gridSize((num_pisxels + BLOCK_W*BLOCK_H - 1)/(BLOCK_W*BLOCK_H),1,1);
    // const dim3 gridSize(1,1,1);
    AT_DISPATCH_ALL_TYPES(dirsMap.type(),"distance_mask_cu",
    ([&] {
        distance_mask_kernel<<<gridSize, blockSize>>>(
        // packbits_u64_kernel<<<4, 64>>>(
            dirsMap.packed_accessor32<float,2,torch::RestrictPtrTraits>(),
            locMap.packed_accessor32<float,1,torch::RestrictPtrTraits>(),
            centroids.packed_accessor32<float,2,torch::RestrictPtrTraits>(),
            mask.packed_accessor32<int32_t,2,torch::RestrictPtrTraits>(),
            threshould

        );
    }));
    return mask;
}


