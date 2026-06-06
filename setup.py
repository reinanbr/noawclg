from setuptools import find_packages, setup

with open("README.md", "r") as fh:
    readme = fh.read()

setup(
    name="noawclg",
    version="2.3.0",
    url="https://github.com/reinanbr/noawclg",
    license="GPLv3",
    author="Reinan Br",
    author_email="slimchatuba@gmail.com",
    description="Python library for NOAA GFS forecasts and GODAS/ERSST ocean data",
    long_description=readme,
    long_description_content_type="text/markdown",
    keywords="climate weather noaa gfs godas ersst enso el-nino ocean",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=[
        "xarray",
        "netCDF4",
        "numpy",
        "pandas",
        "tqdm",
        "cfgrib",
        "requests",
        "geopy",
        "openpyxl",
    ],
    extras_require={
        "plots": [
            "matplotlib>=3.8",
            "cartopy>=0.22",
            "scipy>=1.11",
            "seaborn>=0.13",
            "windrose>=1.9",
            "metpy>=1.6",
            "cmocean>=3.0",
        ],
        "dev": [
            "pytest>=8",
            "pytest-cov",
            "ruff",
            "mypy",
            "types-requests",
            "build",
            "twine",
        ],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Atmospheric Science",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Programming Language :: Python :: 3.13",
    ],
)
