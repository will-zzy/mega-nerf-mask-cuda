# mega-nerf-mask-cuda
[mega-nerf](https://github.com/cmusatyalab/mega-nerf) will use ```create_cluster_masks.py``` to mask all images after dividing the region into blocks.This process will take a considerable amount of time and memory.So I use cuda with pybinding to optimize it.
The algorithm in ```create_cluster_masks.py``` can be summarized as:
Let space $V$ be evenly divided into $N_{grad}$ sub regions $G=\{g_1,g_2,...,g_{N_{grad}}\}$. Given the camera's pose $c2w=\{[R_1|t_1],...,[R_M|t_M]\}$, several rays $r=\{r_{1,1},...,r_{W,H}\}$ can be generated that pass through the pixel coordinates $(u,v,1)^T$ and the camera's optical center $o$
