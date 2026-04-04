from setuptools import setup
from setuptools import find_packages, setup

import json


with open("README.md", "r") as fh:
    readme = fh.read()

setup(name='noawclg',
    version='0.0.5',
    url='https://github.com/reinanbr/noawclg',
    license='GPLv3',
    author='Reinan Br',
    long_description=readme,
    long_description_content_type="text/markdown",
    author_email='slimchatuba@gmail.com',
    keywords='climate weather noaa',
    description=u'Library for getting dataset from noaa site',
    packages=find_packages(),
    install_requires=['numpy','xarray<=0.20.1','netCDF4<=1.5.7','matplotlib','geopy','openpyxl'],)
