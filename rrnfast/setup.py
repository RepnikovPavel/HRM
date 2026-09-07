from setuptools import setup
from torch.utils.cpp_extension import CUDAExtension, BuildExtension

ARCHS = []
for cc in ("89", "90", "120"):
    ARCHS.append(f"-gencode=arch=compute_{cc},code=sm_{cc}")

setup(
    name="rrnfast",
    version="0.1",
    packages=["rrnfast"],
    ext_modules=[
        CUDAExtension(
            "rrnfast_backend",
            ["csrc/bindings.cpp", "csrc/edge_fwd.cu", "csrc/edge_bwd.cu",
             "csrc/lstm.cu", "csrc/rrn_step.cu"],
            extra_compile_args={
                "cxx": ["-O3", "-std=c++17"],
                "nvcc": ["-O3", "-std=c++17", "--expt-extended-lambda"] + ARCHS,
            },
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)
