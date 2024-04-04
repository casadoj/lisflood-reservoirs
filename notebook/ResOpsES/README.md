# Reservoir Operations Spain (ResOpsES)

This document explains the process of creation of the data set _Reservoir Operations Spain_ (ResOpsES) that will be later on to train the LISFLOOD reservoir routine.

## Data

The dataset is made up of static attributes that define the characteristics of both the reservoir and its catchment, and time series.

### Static attributes

#### Reservoir attributes

I compared 4 sources of information:

* The [Spanish Inventory of Reservoirs and Dams](https://www.miteco.gob.es/es/cartografia-y-sig/ide/descargas/agua/inventario-presas-embalses.html).
* The reservoir attributes in EFASv5 (European Flood Awareness System), which were extracted from GLWD or GRanD.
* [GRanD](https://esajournals.onlinelibrary.wiley.com/doi/abs/10.1890/100125) dataset (Global Reservoirs and Dams)
* The ICOLD (Internation Commission on Large Dams) [World Register of Dams](https://www.icold-cigb.org/GB/world_register/world_register_of_dams.asp).

#### Catchment attributes

To replicate the information in EFASv5, I have created catchment attributes based on:

* LISFLOOD static maps. From these maps I computed several statistics (mean, sum, standard deviation, etc) depending on the specific static map. 
* EMO-1 meteorology. From this dataset I computed monthly and annual averages of precitpitation, air temperature and open water evaporation.

#### Time series

#### Reservoir time series

I have found three public sources of **observed** reservoir time series.

* _Anuario de Aforos de España_. This is the official national database of reservoir (also gauging stations and canals) observations created by CEDEX (_Centro de Experimentación_). It contains daily time series of level, storage and outflow for **394 reservoirs**.
* _Agència Catalana de l'Aigua_ (Catalan Water Agency). The catchments exclusively inside Catalunya are managed by ACA. The ACA database (accesible in this [link](https://analisi.transparenciacatalunya.cat/es/Medi-Ambient/Xarxes-de-control-del-medi-consulta-de-l-aigua-i-e/wc95-u57z/about_data) contains daily time series of percentage filling, level and volume for **15 reservoirs**.
* _Hidrosur_ (Andalucian Water Agency). The catchments exclusively inside Andalucía are managed by the regional government and the data infrastratucture maintained by Hidrosur. Their [website](http://www.redhidrosurmedioambiente.es/saih/datos/a/la/carta) allows access to the historical records, but it's limited to retrievals of 31 days; therefore, I have directly contacted the agency to ask for the reservoir time series. The dataset contains hourly time series of level and storage for **23 reservoirs**.

All the reservoir time series above were forwarded to the Hydrological Data Colection Centre (HDCC) to be quality checked and converted to UTC+00. At the moment of creation of the ResOpsES dataset, only the CEDEX time series had been added to the HDCC database.

None of the previous sources include reservoir inflow, which is the only dynamical input of the LISFLOOD reservoir routine. For this reason, and to better represent the conditions under which the LISFLOOD reservoir routine works, I have added to the ResOpsES dataset the inflow time series simulated in the EFASv5 long run.

#### Meteorology

Meterological forcing is, a priori, not needed for the simulation of LISFLOOD reservoir routine. However, I have added daily time series of average precipitation, air temperature and open water evaporation computed from EMO-1.

## Methods

### Selection of reservoirs

Filters by catchment area and reservoir volume.
Availability of time series.

### Estimation of catchment attributes

* Manual correction of reservoir coordinates to match the LDD.
* Creation of catchment masks using `cutmaps`.
* `catchstats`/`mask_statistics`

### Generation of input time series (inflow and meteorology)

Starting from the corrected reservoir point layer.
* `ncextract` using the EFAS5 long run to create the inflow time series.
* `catchstats`  to create areal meteorological time series (precipitation, air temperature and open water evaporation). From these time series I have created some catchment attributes with montly and annual means.





