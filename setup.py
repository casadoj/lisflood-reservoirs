from setuptools import setup, find_packages

setup(
    name='lisflood-reservoirs',
    version='0.1.0',
    packages=find_packages(where='src'),
    package_dir={'': 'src'},
    entry_points={
        'console_scripts': [
            'calibrate-mHM=lisfloodreservoirs.calibration.calibrate_mHM:main', 
        ],
    },
    install_requires=[
        'numpy',
        'pandas',
        'xarray',
        'matplotlib',
        'seaborn',
        'cartopy',
        'statsmodels',
        'tqdm',
    ],
    author='Jesús Casado Rodríguez',
    author_email='jesus.casado-rodriguez@ec.europa.eu',
    description='Package to simulate reservoir operations according to different modelling routines.',
    keywords='hydrology reservoir simulation calibration',
    # Consider adding classifiers to provide more metadata
)
