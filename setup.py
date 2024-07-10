from setuptools import setup, find_packages

setup(
    name='lisflood-reservoirs',
    version='0.1.0',
    packages=find_packages(where='src'),
    package_dir={'': 'src'},
    entry_points={
        'console_scripts': [
            'simulate=lisfloodreservoirs.simulate:main',
            'calibrate=lisfloodreservoirs.calibrate:main',
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
        'pyyaml',
        'spotpy'
    ],
    author='Jesús Casado Rodríguez',
    author_email='jesus.casado-rodriguez@ec.europa.eu',
    description='Package to simulate reservoir operations according to different modelling routines.',
    keywords='hydrology reservoir simulation calibration',
)
