# mega-nerf-mask-cuda

[mega-nerf](https://github.com/cmusatyalab/mega-nerf)在对区域初始分块后，会根据射线到各区域质心点的距离比例进行分割，具体算法如下：
设空间$V$被均匀分割为$N_{grad}$个子区域$G=\{g_1,g_2,...,g_{N_{grad}}\}$
