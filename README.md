# LISFLOOD reservoirs

An analysis of possible improvements in the reservoir representation in the hydrological model [LISFLOOD Open Source](https://github.com/ec-jrc/lisflood-code). 


## Data

Two data sets are used to test the different modelling approaches.

### Reservoir Operations USA ([ResOpsUS](https://www.nature.com/articles/s41597-022-01134-7))

A dataset that includes records for 679 major reservoirs across the US. The time series include inflow, storage, outflow and evaporation, although not all variables are available for all reservoirs. The reservoir characteristics (storage capacity, surface area, catchment area, use...) is taken from the Global Reservoir and Dam dataBase ([GRanD](https://www.globaldamwatch.org/grand/)).

### Reservoir Operations Spain (ResOpsES)

ResOps-ES is a hydrometeorological dataset created in this repository that covers 291 reservoirs in Spain and the time period from 1991 to 2023. The dataset includes both reservoir static attributes and time series. The final purpose of ResOpsES is to train hydrological models that reproduce reservoir operations.

The time series were extracted from the [database of the Spanish Ministry of the Environment](https://ceh.cedex.es/anuarioaforos/default.asp), whereas the reservoir characteristics are a combination of the data in the [Inventory of Dams and Reservoirs of Spain](https://www.miteco.gob.es/es/agua/temas/seguridad-de-presas-y-embalses/inventario-presas-y-embalses.html), GRanD and the database of the [International Commission on Large Dams](https://www.icold-cigb.org/).

## Models

Four reservoir routines are implemented in this repository.

### Linear reservoir

The reservoir outflow is a linear function of the current storage. The parameter that relates these two variables is the residences time ($T$), whose default value is the quotient of the total storage ($V_{\text{tot}}$) and the annual inflow volume ($\bar{I}$):

$$T = \frac{V_{\text{tot}}}{\bar{I}}$$

Class [`Linear`](./src/lisfloodreservoirs/models/linear.py)

### [LISFLOOD](https://ec-jrc.github.io/lisflood-model/3_03_optLISFLOOD_reservoirs/)

Class [`Lisflood`](./src/lisfloodreservoirs/models/lisflood.py)

### [Hanazaki](https://agupubs.onlinelibrary.wiley.com/doi/full/10.1029/2021MS002944)

Class [`Hanazaki`](./src/lisfloodreservoirs/models/hanazaki.py)

### [mHM](https://agupubs.onlinelibrary.wiley.com/doi/10.1029/2023WR035433)

Class [`mHM`](./src/lisfloodreservoirs/models/mhm.py)

### Starfit

Class [`Starfit`](./src/lisfloodreservoirs/models/starfit/Starfit.py)

## Quick start

The repository contains 4 tools to calibrate and run reservoir models. The models included in the repository can be classified in two groups: those that can be calibrated with an iterative process (i.e., a genetic algorithm), and those that are simply fitted using standard `SciPy` tools. To the first group belong the **linear**, **LISFLOOD**, **Hanazaki** and **mHM** models; the tools `run_reservoir` and `cal_reservoir` apply to this group. To the second group belongs the **Starfit** model; the tools `run_starfit` and `fit_starfit` apply to it.

### Configuration

All the tools require a configuration file as the input. A template of this configuration file can be found [here](./src/lisfloodreservoirs/config.yml). The structure of this template is applicable to all the tools.

The configuration file has three sections dedicated to data, simulation, and calibration, repectively.

* The data section defines the location of the reservoir data set and the files that defines the reservoirs to be used (TXT format) and the study period for each of those reservoirs (Pickle format). All the tools are based in a fixed dataset structure:
    *  Attributes must be in a subfolder named *attributes* within the dataset folder.
    *  Time series must be in a subolder named *time_series/csv* within the dataset folder.
* The simulation section defines the reservoir model to be used (`linear`, `lisflood`, `hanazaki`, `mhm` or `starfit`) and the folder where the results of the simulation with default parameters will be saved.
* The calibration section defines the name of the input variable, the target or targets of the calibration (`storage`, `outflow` or both), the parameters of the SCE-UA algorithm, and the directory where results will be saved.

### Tools

#### `run_reservoir`

This tool simulates the reservoir module with default parameters. It is applicable to the **linear**, **LISFLOOD**, **Hanazaki** and **mHM** models.

```
usage: simulate.py [-h] -c CONFIG_FILE [-w]

Run the reservoir routine with default parameters

options:
  -h, --help
                          Show this help message and exit
  -c CONFIG_FILE, --config-file CONFIG_FILE
                          Path to the configuration file
  -w, --overwrite
                          Overwrite existing simulation files. Default: False
```

#### `calibrate`

This tool calibrates the reservoir model using the algorithm Shuffle Complex Evolution - University of Arizona (SCE-UA). It is applicable to the **linear**, **LISFLOOD**, **Hanazaki** and **mHM** models, and it can calibrate the observed storage, outflow, or both at the same time. Eventually, the model is run with the optimised parameters.

```
usage: calibrate.py [-h] -c CONFIG_FILE [-w]

Run the calibration script with a specified configuration file.
It calibrates the reservoir model parameters of the defined routine using the
SCE-UA (Shuffle Complex Evolution-University of Arizona) algorithm for each of
the selected reservoirs.
The optimal parameters are simulated and plotted, if possible comparing against
a simulation with default parameters

options:
  -h, --help
                          Show this help message and exit
  -c CONFIG_FILE, --config-file CONFIG_FILE
                          Path to the configuration file
  -w, --overwrite
                          Overwrite existing simulation files. Default: False
```

#### `fit_starfit`

This tool fits the Starfit reservoir model to the observed data.

```
usage: fit_starfit.py [-h] -c CONFIG_FILE [-w]

Fit the storage and release rules for the Starfit reservoir routine.
The fitted models are saved as Pickle files and plotted against the
observed data used for fitting.

options:
  -h, --help
                          Show this help message and exit
  -c CONFIG_FILE, --config-file CONFIG_FILE
                          Path to the configuration file
  -w, --overwrite
                          Overwrite existing model. Default: False
```

#### `run_starfit`

This tool runs the Starfit reservoir model that was previously fitted with the tool [`fit_starfit`](#fit_starfit).

```
usage: run_starfit.py [-h] -c CONFIG_FILE [-w]

Run Starfit simulation with the paremeter fitted using `fit_starfit`.
The simulated time series are saved as CSV files. To analyse the results,
the code creates a CSV file of performance metrics, and a scatter and a 
line plot comparing the observed and simulated time series.

options:
  -h, --help
                          Show this help message and exit
  -c CONFIG_FILE, --config-file CONFIG_FILE
                          Path to the configuration file
  -w, --overwrite
                          Overwrite existing simulation files. Default: False
```
