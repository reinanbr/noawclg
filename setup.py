from setuptools import find_packages, setup


with open("README.md", "r") as fh:
    readme = fh.read()

setup(
    name="noawclg",
    version="2.2.6",
    url="https://github.com/reinanbr/noawclg",
    license="GPLv3",
    author="Reinan Br",
    long_description=readme,
    long_description_content_type="text/markdown",
    author_email="slimchatuba@gmail.com",
    keywords="climate weather noaa",
    description="Library for getting dataset from noaa site",
    packages=find_packages(),
    install_requires=[
        "xarray",
        "netCDF4",
        "matplotlib",
        "tqdm",
        "cfgrib",
        "requests",
        "geopy",
        "openpyxl",
        "numpy",
    ],
)
