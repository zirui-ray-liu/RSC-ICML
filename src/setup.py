from setuptools import setup, find_packages
from torch.utils import cpp_extension

setup(
    name="fastgnn",
      ext_modules=[
          cpp_extension.CUDAExtension(
              'fastgnn.spmm_cuda',
              ['fastgnn/spmm_cuda/spmm.cc', 'fastgnn/spmm_cuda/spmm_cuda.cu']
          ),          
      ],
    cmdclass={'build_ext': cpp_extension.BuildExtension},
    packages=find_packages(),
)