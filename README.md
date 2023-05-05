# mega-nerf-mask-cuda
[mega-nerf](https://github.com/cmusatyalab/mega-nerf) will use ```create_cluster_masks.py``` to mask all images after dividing the region into blocks.This process will take a considerable amount of time and memory.So I use cuda with pybinding to optimize it.<br>
The algorithm in ```create_cluster_masks.py``` can be summarized as:<br>
Let space $V$ be evenly divided into $N_{grad}$ sub regions $G=\{g_1,g_2,...,g_{N_{grad}}\}$, where $g_i=(c_i,s_i)\in R^{2\times3}$. Given the camera's pose $c2w=\{[R_1|t_1],...,[R_M|t_M]\}$, several rays $r=\{r_{1,1},...,r_{W,H}\}$ can be generated that pass through the pixel coordinates $(u,v,1)^T$ and the camera's optical center $o$:<br>
&emsp;1.Sample $N_{sample}$ points $p=\{p_1,...,p_{N_{sample}}\}$ for $r_{u,v}$ a and calculate $d(p,c)$,where $d(x,y)=||x-y||_2^2$<br>
&emsp;2.Let $d_i^*=\min_j\{d(p_i,c_j)\},ratio_j=\min_i\frac{d(p_i,c_j)}{d_i^\*}$,  If $ratio_j \leq T$, then mask is True for $(u,v)$ of $g_i$


# Install
git clone


