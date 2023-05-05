#include "utils.h"



torch::Tensor packbits_u32(torch::Tensor idx_array, torch::Tensor bits_array){
    CHECK_CUDA(idx_array);
    CHECK_CUDA(bits_array);
    
    return packbits_u32_cu(idx_array,bits_array);
}

torch::Tensor un_packbits_u32(torch::Tensor idx_array, torch::Tensor bits_array){
    CHECK_CUDA(idx_array);
    CHECK_CUDA(bits_array);
    
    return un_packbits_u32_cu(idx_array,bits_array);
}


torch::Tensor distance_mask(
    torch::Tensor dirsMap, 
    torch::Tensor locMap,
    torch::Tensor centroids,
    torch::Tensor mask,
    const float threshould
    ){
    CHECK_CUDA(dirsMap);
    CHECK_CUDA(locMap);
    CHECK_CUDA(centroids);
    CHECK_CUDA(mask);
    
    return distance_mask_cu(
        dirsMap,
        locMap,
        centroids,
        mask,
        threshould);
}

torch::Tensor mega_nerf_mask(   
    torch::Tensor dirsMap,//[WxH , 3]   
    torch::Tensor locMap,//[WxH , 3]
    torch::Tensor centroids,//[C , 3]
    torch::Tensor t_range,//[WxH , 2]
    const int samples,//每条射线上采样点数
    const float threshould//重叠阈值
    )
    {
    CHECK_CUDA(dirsMap);
    CHECK_CUDA(locMap);
    CHECK_CUDA(centroids);
    CHECK_CUDA(t_range);
    
    return mega_nerf_mask_cu(
        dirsMap,
        locMap,
        centroids,
        t_range,
        samples,
        threshould);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m){
    
    m.def("packbits_u32",&packbits_u32);
    m.def("un_packbits_u32",&un_packbits_u32);

    m.def("distance_mask",&distance_mask);
    m.def("mega_nerf_mask",&mega_nerf_mask);
    
}

