# Compare reservoir models
***

**Author:** *Chus Casado Rodríguez*<br>
**Date:** *16-07-2024*<br>

## Introduction

This documents summarizes the comparison of four reservoir models on an identical dataset of 97 reservoirs in the USA with records of both inflow, storage and outflot. The objective is to identify which reservoir model is more suitable to be included in LISFLOOD-OS, both in terms of performance and feasibility to be implemented on a global/continental scale, where data is scarce.
    
## Data

This time I focused on reservoirs in the USA because they are the only ones for which I have observed inflows. 

In the Spanish dataset I developed before, there are only observations of storage, level and outflow. I estimated inflows from those records based on the reservoir mass balance, and compared that estimation against the LISFLOOD simulated inflows. I discovered that there were, in some cases, big differences between the simulation and the estimation. Therefore, I have tried to remove the uncertainty in the inflow time series by moving to a dataset where that includes inflow records.

I have used two data sources. The observed time series were taken from [ResOpsUS](https://www.nature.com/articles/s41597-022-01134-7). This dataset includes daily time series of inflow, storage, level, outflow and evaporation for 678 major reservoirs across the US (not all variables are available for all the reservoirs). The reservoir attributes were taken from [GRanD](https://esajournals.onlinelibrary.wiley.com/doi/full/10.1890/100125) (Global Reservoir and Dam Database). GRanD includes reservoir and dam attributes (storage capacity, water surface, dam height, catchment area, reservoir use...) for 7320 reservoirs over the world.

### Data treatment

#### Time series

A quick look at the time series in ResOpsUS shows that the records are not consistent in many cases. There are negative values (not possible in any of the variables involved), zero values, outliers, sudden drops in reservoir storage. I have developed simple functions to clean the storage timeseries ([`clean_storage`]('../src/lisfloodreservoirs/utils/timeseries.py')), and clean and fill in gaps in the inflow time series ([`clean_inflow`]('../src/lisfloodreservoirs/utils/timeseries.py')). I have filled in gaps only in the inflow time series because this is the input of the reservoir models, so missing values break the simulation. On the contrary, gaps in the storage or outflow time series have no effects, since these time series are only used for assessing the performance of the model. I used a linear filling up to 7 days (longer gaps are kept).

#### Selection of reservoirs and study period

The selection of reservoirs to be included in the analysis was based in three conditions:

* The observed time series of the three variables of interest (inflow, storage, outflow) must be available.
* The bias between observed outflow and inflow must be smaller than 30%. I discovered that the bias between these two time series is very high for many reservoirs. Some bias can be expected due to reservoir losses such as evaporation or leakages, but I assumed that those losses should not be larger than 30%. Biases below that threshold may be caused by reservoirs whose outflow is not released to the river, but to other water system (irrigation chanels, water supply...).
* The length of the time series must be at least 8 years. I have identified the longest period for which data for all thre variables is available; if that period is shorter than 8 years, I discard the reservoir.

In the end, I have selected 97 reservoirs.

## Methods

### Reservoir models

I have tested four reservoir models. In the following subsections I explain each of these models from the simpler to the more complex.

#### [Linear reservoir](../src/lisfloodreservoirs/models/linear.py)

The linear reservoirs models the outflow ($Q_t$) as a linear function of the current storage ($V_t$). 

$$Q_t = \frac{V_t}{T}$$

The only parameter is the residence time ($T$). It represents the number of days that a drop of water would on average stay in the reservoir. The default value can be estimated knowing the storage capacity ($V_tot$, in m3) and the mean inflow ($\bar{I}$, in m3/day):

$$T = \frac{V_{tot}}{\bar{I}} [\text{days}]$$

Table 1 shows the search range for this parameter defined in the calibration of the linear reservoir. This calibration is implemented in the class [`Linear_calibrator`](../src/lisfloodreservoirs/calibration/linear.py)

    
***Table 1**. Calibration parameters in the linear reservoir.*

| parameter | description    | units | minimum | maximum | default |
| --------- | -------------- | ----- | ------- | ------- | ------- |
| $T$       | residence time | days  | 7       | 2190    | $\frac{V_{tot}}{\bar{I}}$ |

The advantage of this routine is its simplicity, as it has a single model parameter that can be estimated kwnown the storage capacity (provided by GRanD, for instance) and the average inflow (from GloFASv4, for instance). The drawback is also its simplicity. A given storage will always produce the same outflow, which is not realistic. Neither seasonality nor other operator management can be reproduced with this approach.

As an example, Figure 1 compares the daily values of storage, outflow and inflow for the observed data (blue dots) and the default simulation of the linear model for reservoir 1053. The simplicity of this approach is particularly clear in the storage-outflow scatter plot, where the observation shows some scatter, but the simulation is a straight line.

<img src="../../results/ResOpsUS/linear/default/1053_scatter_obs_sim.jpg" alt="reservoir 1053" title="Linear model" width="800">

***Figure 1**. Comparison of the observed (blue) and default simulation (orange) of reservoir 1053 with the linear model.*

#### [LISFLOOD](../../src/lisfloodreservoirs/models/lisflood.py)

In simple terms, the current LISFLOOD model is a concatenation of linear reservoirs, where the constant connecting storage and outflow changes according to the storage zone at which the reservoir is at that moment: conservative, normal or flood.

\begin{equation}
Q_t = \begin{cases}
Q_{min} & \text{if } V_t < 2 \cdot V_{min} \\
Q_{min} + \left( Q_n - Q_{min} \right) \frac{V_t - 2 \cdot V_{min}}{V_n - 2 \cdot V_{min}} & \text{if } 2 \cdot V_{min} \leq V_t < V_n \\
Q_n & \text{if } V_n \leq V_t < V_{n,adj} \\
Q_n + \left( Q_f - Q_n \right) \frac{V_t - V_{n,adj}}{V_f - V_{n,adj}} & \text{if } V_{n,adj} \leq V < V_f \\
\max\left(\frac{V_t - V_f}{\Delta t}, \min\left(Q_f, \max \left( k \cdot I, Q_n \right) \right) \right) & \text{if } V_t > V_f
\end{cases}
\end{equation}

where $V_{min}$, $V_n$, $V_{n,adj}$ and $V_f$ are the minimum storage, lower and upper bound of the normal storage zone, and the flood storage, respectively. $Q_{min}$, $Q_n$ and $Q_f$ are the minimum, normal and flood outflow.

The LISFLOOD-OS calibration tunes two reservoir parameters, adjusting the normal outflow ($Q_n$) and the upper limit of the normal storage ($V_{n,adj}$). The other values defining the three break points in the routine (see Figure 2) are default.

In the calibration I have performed here, I have allowed maximum flexibility to the routine. I have calibrated the six parameters in Table 2, i.e., I only fixed the minimum storage and minimum outflow. This calibration is implemented in the class [`Lisflood_calibrator`](../src/lisfloodreservoirs/calibration/lisflood.py).

***Table 2**. Calibration parameters in the LISFLOOD reservoir.*

| parameter | description    | units | minimum | maximum | default |
| --------- | -------------- | ----- | ------- | ------- | ------- |
| $\alpha$  | Fraction of the total storage corresponding to the flood limit ($Q_f$) | -  | 0.2       | 0.99    | 0.97 |
| $\beta$   | Proportion between flood limit and minimum storage corresponding to the normal limit | -  | 0.001     | 0.999   | 0.655 |
| $\gamma$  | Proportion between flood and normal limits corresponding to the adjusted normal limit | -  | 0.001     | 0.999   | 1.533 |
| $\delta$  | Factor multiplying the 100-year inflow that defines the flood outflow | -  | 0.1     | 0.5   | 0.3 |
| $\epsilon$ | Ratio between normal and flood outflows | -  | 0.001     | 0.999   | $\frac{Q_f}{\bar{I}}$ |
| $k$   | Release coefficient | -  | 1     | 5   | 1.2 |

As the linear reservoir, the LISFLOOD routine is often a univocal relation between storage and outflow, which is not realistic. Above the normal zone, there are some limitations to the outflow based on the inflow which allows for some deviations from this univocal behaviour. It should be a more flexible routine, as the number of parameters is larger, but that is also a drawback, since those parameter need to be fitted.

Figure 2 shows a comparison of the observation and default simulation of the LISFLOOD reservoir model for reservoir 1053.

<img src="../../results/ResOpsUS/lisflood/default/1053_scatter_obs_sim.jpg" alt="reservoir 1053" title="Linear model" width="800">

***Figure 2**. Comparison of the observed (blue) and default simulation (orange) of reservoir 1053 with the LISFLOOD model.*

#### [Hanazaki](../../src/lisfloodreservoirs/models/hanazaki.py)

The model in [Hanazaki et al. (2022)](https://agupubs.onlinelibrary.wiley.com/doi/full/10.1029/2021MS002944) is an evolution of the LISFLOOD model that creates two different reservoir operations depending on the inflow ($I_t$). If the inflow is smaller than the flood outflow ($Q_f$), outflow is a quadratic function of storage; this quadratic behaviour limits the outflow when the reservoir empties, hence storing water for future needs. If the inflow is larger than the flood outflow, it's a linear reservoir.

\begin{equation}
Q =
\begin{cases}
Q_n \frac{V_t}{V_f} & \text{if } V_t < V_{\text{min}} \\
Q_n \frac{V_{\text{min}}}{V_f} + \left( \frac{V_t - V_{\text{min}}}{V_e - V_{\text{min}}} \right)^2 \left( Q_f - Q_n \frac{V_{\text{min}}}{V_f} \right) & \text{if } I_t < Q_f \text{ and } V_{\text{min}} \leq V_t < V_e \\
Q_f & \text{if }  I_t < Q_f \text{ and } V_t \geq V_e  \\
Q_n \frac{V_{\text{min}}}{V_f} + \frac{V_t - V_{\text{min}}}{V_f - V_{\text{min}}} \left( Q_f - Q_n \frac{V_{\text{min}}}{V_f} \right) & \text{if } I_t \geq Q_f \text{ and }  V_{\text{min}} \leq V_t < V_f \\
Q_f + k \cdot \frac{V_t - V_f}{V_e - V_f} \cdot (I_t - Q_f) & \text{if } I_t \geq Q_f \text{ and } V_f \leq V_t < V_e  \\
I_t & \text{if } I_t \geq Q_f \text{ and } V_t \geq V_e
\end{cases}
\end{equation}

***Table 2**. Calibration parameters in the LISFLOOD reservoir.*

| parameter | description    | units | minimum | maximum | default |
| --------- | -------------- | ----- | ------- | ------- | ------- |
| $\alpha$  | Fraction of the total storage corresponding to the flood limit ($Q_f$) | -  | 0.2       | 0.99    | 0.97 |
| $\beta$   | Proportion between flood limit and minimum storage corresponding to the normal limit | -  | 0.001     | 0.999   | 0.655 |
| $\gamma$  | Proportion between flood and normal limits corresponding to the adjusted normal limit | -  | 0.001     | 0.999   | 1.533 |
| $\delta$  | Factor multiplying the 100-year inflow that defines the flood outflow | -  | 0.1     | 0.5   | 0.3 |
| $\epsilon$ | Ratio between normal and flood outflows | -  | 0.001     | 0.999   | $\frac{Q_f}{\bar{I}}$ |
| $k$   | Release coefficient | -  | 1     | 5   | 1.2 |

<img src="../../results/ResOpsUS/hanazaki/default/1053_scatter_obs_sim.jpg" alt="reservoir 1053" title="Linear model" width="800">

***Figure 3**. Comparison of the observed (blue) and default simulation (orange) of reservoir 1053 with the Hanazaki model.*

#### mHM(../../src/lisfloodreservoirs/models/mhm.py)

This model differs from the others as outflow is not a univocal function of storage. Instead, it relies heavily on a demand time series that need to be estimated somehow. In the [paper](https://agupubs.onlinelibrary.wiley.com/doi/10.1029/2023WR035433), they train a random forest specific to each reservoir to predict the demand. The demand time series is used to limit releases (store water) when the current demand is lower compared with the annual mean, and to increase releases (empty the reservoir) with higher demands. The final release is further constrained by the current reservoir filling.

<img src="../../results/ResOpsUS/mhm/default/1053_scatter_obs_sim.jpg" alt="reservoir 1053" title="Linear model" width="800">

***Figure 4**. Comparison of the observed (blue) and default simulation (orange) of reservoir 1053 with the mHM model.*

### Runs

For each model, 4 simulations were run. In all cases, the input is the observed reservoir inflow. In previous tests I had used GloFAS simulations to be able to test reservoir for which there is no observed inflow. This time I tried to avoid errors induced by the quality of the GloFAS estimated inflow.

1. A simulation with **default attributes**. To run this simulations I've created the command [`simulate`](../../src/lisfloodreservoirs/simulate.py).
2. Three calibrations:
    * Univariate calibration of **storage**.
    * Univariate calibration of **outflow**.
    * Bivariate calibration of both **storage and outflow**.
    
    Calibrations were done using the implementation of the SCEUA (Shuffle Complex Evolution - University of Arizona) algorithm in the Python library [`spotpy`](https://spotpy.readthedocs.io/en/latest/). In all cases I used the complete observed time series, and I set up the algorithm to run a maximum of 1000 iterations with 4 complexes. The attributes calibrated for each model are different, both in number and in meaning. I've created specific classes to each of the models: [`Linear_calibrator`](../../src/lisfloodreservoirs/calibration/linear.py), [`Lisflood_calibrator`](../../src/lisfloodreservoirs/calibration/lisflood.py), [`Hanazaki_calibrator`](../../src/lisfloodreservoirs/calibration/hanazaki.py) and [`mHM_calibrator`](../../src/lisfloodreservoirs/calibration/mhm.py). To run the calibrations I've created the commad [`calibrate`](../../src/lisfloodreservoirs/calibrate.py).

#### Default parameters

#### Calibration

## Results


