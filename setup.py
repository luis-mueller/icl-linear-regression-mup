from setuptools import setup

setup(
    name="icl-linear-regression-mup",
    py_modules=["icl_linear_regression_mup"],
    install_requires=[
        "torch==2.1.0",
        "hydra-core==1.3.2",
        "loguru==0.7.2",
        "mup==1.0.0",
        "pandas==2.1.2",
    ],
    version="0.0.1",
)
