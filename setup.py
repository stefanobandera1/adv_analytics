from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="adsat",
    version="0.1.0",
    author="Stefano Bandera",
    author_email="mr.stefanobandera@gmail.com",
    description="Advertising Saturation Analysis Toolkit – identify impression saturation points in campaign data",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/stefanobandera1/adsat",
    packages=find_packages(exclude=["tests*", "examples*"]),
    python_requires=">=3.9",
    install_requires=[
        "numpy>=1.23",
        "pandas>=1.5",
        "scipy>=1.9",
        "scikit-learn>=1.1",
        "matplotlib>=3.6",
    ],
    extras_require={
        "bayesian": [
            "pymc>=5.0",
            "arviz>=0.15",
        ],
        "dev": [
            "pytest>=7.0",
            "pytest-cov",
            "black",
            "ruff",
            "twine",
            "build",
        ],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Information Analysis",
        "Topic :: Office/Business :: Financial",
    ],
    keywords="advertising saturation hill-function media-mix-modeling mmm impressions",
)
