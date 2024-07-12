# LISFLOOD reservoirs

An analysis of possible improvements in the reservoir representation in the hydrological model [LISFLOOD Open Source](https://github.com/ec-jrc/lisflood-code). 


## Data

Two data sets are used to test the different modelling approaches.

### Reservoir Operations USA ([ResOpsUS](https://www.nature.com/articles/s41597-022-01134-7))

A dataset that includes records for 679 major reservoirs across the US. The time series include inflow, storage, outflow and evaporation, although not all variables are available for all reservoirs. The reservoir characteristics (storage capacity, surface area, catchment area, use...) is taken from the Global Reservoir and Dam dataBase ([GRanD](https://www.globaldamwatch.org/grand/)).

### Reservoir Operations Spain (ResOpsES)

ResOps-ES is a hydrometeorological dataset created in this repository that covers 291 reservoirs in Spain and the time period from 1991 to 2023. The dataset includes both reservoir static attributes and time series. The final purpose of ResOpsES is to train hydrological models that reproduce reservoir operations.

The time series were extracted from the [database of the Spanish Ministry of the Environment](https://ceh.cedex.es/anuarioaforos/default.asp), whereas the reservoir characteristics are a combination of the data in the [Inventory of Dams and Reservoirs of Spain](https://www.miteco.gob.es/es/agua/temas/seguridad-de-presas-y-embalses/inventario-presas-y-embalses.html), GRanD and the database of the [International Commission on Large Dams](https://www.icold-cigb.org/).

## Reservoir modelling

Four reservoir routines are implemented in this repository.

### Linear reservoir

### [LISFLOOD](https://ec-jrc.github.io/lisflood-model/3_03_optLISFLOOD_reservoirs/)

### [Hanazaki](https://agupubs.onlinelibrary.wiley.com/doi/full/10.1029/2021MS002944)

### [mHM](https://agupubs.onlinelibrary.wiley.com/doi/10.1029/2023WR035433)

