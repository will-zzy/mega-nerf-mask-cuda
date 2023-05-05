import glob
import os.path as osp
from setuptools import setup
from torch.utils.cpp_extension import CUDAExtension, BuildExtension


ROOT_DIR = osp.dirname(osp.abspath(__file__))
include_dirs = [osp.join(ROOT_DIR, "include")]
# "helper_math.h" is copied from https://github.com/NVIDIA/cuda-samples/blob/master/Common/helper_math.h

sources = glob.glob('*.cpp')+glob.glob('*.cu')


setup(
    name='maskStudio',
    version='3.0',
    description='Block Mask Algorithm in Large Scene Nerf',
    author='will.',
    ext_modules=[
        CUDAExtension(
            name='maskStudio',
            sources=sources,
            include_dirs=include_dirs,
            libraries =["c10","torch_python","torch"],
            extra_compile_args={'cxx': ['-O2'],
                                'nvcc': ['-O2']}
        )
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)