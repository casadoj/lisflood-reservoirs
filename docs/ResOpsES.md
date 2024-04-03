# Reservoir Operations Spain (ResOpsES)
***

This document explains the process of creation of the data set _Reservoir Operations Spain_ (ResOpsES) that will be later on to train the LISFLOOD reservoir routine.

## Input data

### Static attributes

#### Reservoir attributes

I compared 4 sources of information:

* The Spanish Inventory of Reservoirs and Dams:
    * Inventory of reservoirs:
    * Inventory of dams:
* The reservoir attributes in EFASv5 (European Flood Awareness System), which were mostly extracted from GWLD or GRanD.
* GRanD data set (Global Reservoirs and Dams)
* ICOLD data set (Internation Commission on Large Dams)

#### Catchment attributes

* EFAS static maps.
* EMO-1 meteorology.

### Time series

##### Observed time series

Three sources:

* _Anuario de Aforos de España_: daily time series of reservoir level, storage and outflow.
* _Agència Catalana de l'Aigua_: daily time series of reservoir level and storage.
* _Hidrosur_ (Andalucian Water Management Agency): hourly time series of reservoir level and storage.

#### Simulations

Inflow from EFAS simulations.