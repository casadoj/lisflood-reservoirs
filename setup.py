from setuptools import setup, find_packages

setup(
    name='lisflood-reservoirs',
    version='1.0.0',
    packages=find_packages(where='src'),
    package_dir={'': 'src'},
    entry_points={
        'console_scripts': [
            'simulate=lisfloodreservoirs.simulate:main',
            'calibrate=lisfloodreservoirs.calibrate:main',
            'catchstats=lisfloodreservoirs.catchstats:main'
        ],
    },
    install_requires=[
        'cartopy',
        'dask',
        'matplotlib',
        'netcdf4',
        'numpy',
        'pandas',
        'pyyaml',
        'seaborn',
        'spotpy',
        'statsmodels',
        'tqdm',
        'xarray',
    ],
    author='Jesús Casado Rodríguez',
    author_email='jesus.casado-rodriguez@ec.europa.eu',
    description='Package to simulate reservoir operations according to different modelling routines.',
    keywords='hydrology reservoir simulation calibration',
)
