from setuptools import Extension, setup


native = Extension(
    "vinf._native",
    sources=["csrc/native.c"],
    extra_compile_args=["-O3"],
)


setup(ext_modules=[native])

