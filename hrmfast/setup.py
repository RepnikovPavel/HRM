from setuptools import setup
from torch.utils.cpp_extension import CUDAExtension, BuildExtension

ARCHS = []
for cc in ("89", "90", "120"):
    ARCHS.append(f"-gencode=arch=compute_{cc},code=sm_{cc}")

setup(
    name="hrmfast",
    version="0.1",
    packages=["hrmfast"],
    ext_modules=[
        CUDAExtension(
            "hrmfast_backend",
            ["csrc/bindings.cpp", "csrc/mlp_swiglu.cu", "csrc/mlp_swiglu_backward.cu", "csrc/attention.cu"],
            extra_compile_args={
                "cxx": ["-O3", "-std=c++17"],
                "nvcc": ["-O3", "-std=c++17", "--expt-extended-lambda"] + ARCHS,
            },
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)
