from setuptools import setup, Extension
import pybind11

ext_modules = [
    Extension(
        "ports_cpp",  # Module name
        ["ports_cpp.cpp"],  # Source file
        include_dirs=[pybind11.get_include()],  # Include pybind11 headers
        language="c++",
        extra_compile_args=["-std=c++17"],  # Use C++17 standard
    )
]

setup(
    name="ports_cpp",
    version="0.1",
    ext_modules=ext_modules,
    zip_safe=False,
)
