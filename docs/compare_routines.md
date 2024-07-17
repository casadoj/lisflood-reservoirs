# Compare reservoir models
***

**Author:** Chus Casado Rodríguez<br>
**Date:** 17-07-2024<br>

## 1 Introduction

This documents summarizes the comparison of four reservoir models on an identical dataset of 97 reservoirs in the USA with records of both inflow, storage and outflot. The objective is to identify which reservoir model is more suitable to be included in LISFLOOD-OS, both in terms of performance and feasibility to be implemented on a global/continental scale, where data is scarce.
    
## 2 Data

This time I focused on reservoirs in the USA because they are the only ones for which I have observed inflows. 

In the Spanish dataset I developed before, there are only observations of storage, level and outflow. I estimated inflows from those records based on the reservoir mass balance, and compared that estimation against the LISFLOOD simulated inflows. I discovered that there were, in some cases, big differences between the simulation and the estimation. Therefore, I have tried to remove the uncertainty in the inflow time series by moving to a dataset where that includes inflow records.

I have used two data sources. The observed time series were taken from [ResOpsUS](https://www.nature.com/articles/s41597-022-01134-7). This dataset includes daily time series of inflow, storage, level, outflow and evaporation for 678 major reservoirs across the US (not all variables are available for all the reservoirs). The reservoir attributes were taken from [GRanD](https://esajournals.onlinelibrary.wiley.com/doi/full/10.1890/100125) (Global Reservoir and Dam Database). GRanD includes reservoir and dam attributes (storage capacity, water surface, dam height, catchment area, reservoir use...) for 7320 reservoirs over the world.

### 2.1 Data treatment

#### 2.1.1 Time series

A quick look at the time series in ResOpsUS shows that the records are not consistent in many cases. There are negative values (not possible in any of the variables involved), zero values, outliers, sudden drops in reservoir storage. I have developed simple functions to clean the storage timeseries ([`clean_storage`]('../src/lisfloodreservoirs/utils/timeseries.py')), and clean and fill in gaps in the inflow time series ([`clean_inflow`]('../src/lisfloodreservoirs/utils/timeseries.py')). I have filled in gaps only in the inflow time series because this is the input of the reservoir models, so missing values break the simulation. On the contrary, gaps in the storage or outflow time series have no effects, since these time series are only used for assessing the performance of the model. I used a linear filling up to 7 days (longer gaps are kept).

#### 2.1.2 Selection of reservoirs and study period

The selection of reservoirs to be included in the analysis was based in three conditions:

* The observed time series of the three variables of interest (inflow, storage, outflow) must be available.
* The reservoir catchment area must be at least 250 km². Actually, this condition did not remove any reservoir that fulfilled the other conditions.
* The reservoir storage capacity must be at least 10 hm3. Actually, this condition dis not remove any reservoir that fulfilled the other conditions.
* The degree of regulation must be larger or equal than 0.08.
* The bias between observed outflow and inflow must be smaller than 30%. I discovered that the bias between these two time series is very high for many reservoirs. Some bias can be expected due to reservoir losses such as evaporation or leakages, but I assumed that those losses should not be larger than 30%. Biases below that threshold may be caused by reservoirs whose outflow is not released to the river, but to other water system (irrigation chanels, water supply...).
* The length of the time series must be at least 8 years. I have identified the longest period for which data for all thre variables is available; if that period is shorter than 8 years, I discard the reservoir.

In the end, I have selected 90 reservoirs.

> **Note**. The dataset is available in _Z:/nahaUsers/casadje/datasets/reservoirs/ResOpsUS/v1.1/_ or in the HPC in _/BGFS/DISASTER/casadje2/ResOpsUS/v1.1/_

## 3 Methods

### 3.1 Reservoir models

I have tested four reservoir models. In the following subsections I explain each of these models from the simpler to the more complex.

#### 3.1.1 [Linear reservoir](../src/lisfloodreservoirs/models/linear.py)

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

As an example, Figure 1 compares the daily values of storage, outflow and inflow for the observed data (blue dots) and the default simulation of the linear model for reservoir 355. The simplicity of this approach is particularly clear in the storage-outflow scatter plot, where the observation shows some scatter, but the simulation is a straight line.

<img src="../results/ResOpsUS/linear/default/355_scatter_obs_sim.jpg" alt="reservoir 355" title="Linear model" width="800">

***Figure 1**. Comparison of the observed (blue) and default simulation (orange) of reservoir 355 with the linear model.*

#### 3.1.2 [LISFLOOD](../../src/lisfloodreservoirs/models/lisflood.py)

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
| $\alpha$  | Fraction of the total storage corresponding to the flood limit ($V_f$) | -  | 0.2       | 0.99    | 0.97 |
| $\beta$   | Proportion between flood limit and minimum storage corresponding to the normal limit | -  | 0.001     | 0.999   | 0.655 |
| $\gamma$  | Proportion between flood and normal limits corresponding to the adjusted normal limit | -  | 0.001     | 0.999   | 1.533 |
| $\delta$  | Factor multiplying the 100-year inflow that defines the flood outflow ($Q_f$) | -  | 0.1     | 0.5   | 0.3 |
| $\epsilon$ | Ratio between normal and flood outflows | -  | 0.001     | 0.999   | $\frac{Q_f}{\bar{I}}$ |
| $k$   | Release coefficient | -  | 1     | 5   | 1.2 |

As the linear reservoir, the LISFLOOD routine is often a univocal relation between storage and outflow, which is not realistic. Above the normal zone, there are some limitations to the outflow based on the inflow which allows for some deviations from this univocal behaviour. It should be a more flexible routine, as the number of parameters is larger, but that is also a drawback, since those parameter need to be fitted.

Figure 2 shows a comparison of the observation and default simulation of the LISFLOOD reservoir model for reservoir 355.

<img src="../results/ResOpsUS/lisflood/default/355_scatter_obs_sim.jpg" alt="reservoir 355" title="Linear model" width="800">

***Figure 2**. Comparison of the observed (blue) and default simulation (orange) of reservoir 355 with the LISFLOOD model.*

#### 3.1.3 [Hanazaki](../../src/lisfloodreservoirs/models/hanazaki.py)

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

Similarly to LISFLOOD, this routine needs to specify some storage and outflow limits. In their paper they do not calibrate the model, as they use default values to define the storage and outflow limits. On the contrary, I have developed a calibration (class [`Hanazaki_calibrator`](../src/lisfloodreservoirs/calibration/hanazaki.py)) with 5 model parameters. To be able to compare the LISFLOOD and Hanazaki routine, the definition of three of those parameters is identical between these two routines: $\alpha$, $\delta$, $\epsilon$.

***Table 3**. Calibration parameters in the Hanazaki reservoir.*

| parameter | description    | units | minimum | maximum | default |
| --------- | -------------- | ----- | ------- | ------- | ------- |
| $\alpha$  | Fraction of the total storage corresponding to the flood limit ($V_f$) | -  | 0.2       | 0.99    | 0.75 |
| $\beta$   | Proportion between flood limit and total storage corresponding to the extreme limit | -  | 0.001     | 0.999   | 0.2 |
| $\gamma$  | Proportion of the flood limit corresponding to the normal limit | -  | 0.001     | 0.999   | 0.5 |
| $\delta$  | Factor multiplying the 100-year inflow that defines the flood outflow ($Q_f$) | -  | 0.1     | 0.5   | 0.3 |
| $\epsilon$ | Ratio between normal and flood outflows | -  | 0.001     | 0.999   | $\frac{Q_f}{\bar{I}}$ |

The benefit of the Hanazaki model in comparison with LISFLOOD is that it enhances storing water when the inflows are small. The relation between storage and outflow is now bivocal instead of univocal. This routine is able to reproduce a larger dispersion in the inflow-outflow scatter plot, whereas LISFLOOD or the linear model where mostly lines.

<img src="../results/ResOpsUS/hanazaki/default/355_scatter_obs_sim.jpg" alt="reservoir 355" title="Linear model" width="800">

***Figure 3**. Comparison of the observed (blue) and default simulation (orange) of reservoir 355 with the Hanazaki model.*

#### 3.1.4 [mHM](../../src/lisfloodreservoirs/models/mhm.py)

This model differs from the others as outflow is not a simple function (linear or quadratic) of storage. Instead, it relies heavily on a demand time series that needs to be estimated somehow. In the [paper](https://agupubs.onlinelibrary.wiley.com/doi/10.1029/2023WR035433), they train a random forest specific to each reservoir to predict the demand. The demand time series is used to limit releases (store water) when the current demand is lower compared with the annual mean, and to increase releases (empty the reservoir) with higher demands. The final release is further constrained by the current reservoir filling.

In my implementation, I use the outflow records to estimate an annual demand time series that I repeat over the years. The function [`create_demand`](../src/lisfloodreservoirs/utils/timeseries.py) takes the outflow records and computes the mean ouflow for every day of the year over all the available years, then applies a bias to assume that not all the outflow is meant to fulfil demands, and finally applies a moving mean of window 28 days to smooth the time series. This is just a first approximation aiming at inducing some seasonality in the reservoir modelling.

**Reservoir release**

The reservoir release at day $t$ is partitioned in a component based on the hedged demand ($\hat{D}_t$) and a component based on the inflow ($I_t$):

$$Q_t = \rho \cdot \kappa_t \cdot \hat{D}_t + (1 - \rho) \cdot I_t$$

The partition coefficient ($\rho$) depends on the degree of regulation of the reservoir ($DOR$), i.e., the ratio between the reservoir storage capacity and the annual inflow. The release of highly regulated reservoirs ($DOR \geq \alpha$) depends only on the demand component ($\rho = 1$), whereas in the rest of the cases the release depends on both the demand and the current inflow components:

$$\rho = min\left(1, \left( \frac{DOR}{\alpha} \right)^\beta \right)$$

$\alpha$ and $\beta$ are model parameters in mHM. Shin et al. (2019) and Hanasaki et al. (2006) used fixed values as reported in Table 1.

In the release formulation, the current filling ($V_t$) of the reservoir is used indirectly to limit the supplied demand by the time varying coefficient $\kappa_t$:

$$\kappa_t = \left( \frac{V_t}{\gamma \cdot V} \right) ^\lambda = \left( \frac{V_t}{V_n} \right) ^\lambda$$

where $V_n$ is the normal filling of the reservoir (mHM estimates this value with the parameter $\gamma$), and $\lambda$ is a parameter that further controls demand hedging. Again, Shin et al. (2019) and Hanasaki et al. (2006) used fixed values for $\gamma$ and $\lambda$ (see Table 1).

**Demand hedging**

The hedged demand ($\hat{D}_t$) is a transformation of the actual daily demand ($D_t$) based on the average inflow ($\overline{I}$) and average demand ($\overline{D}$). The $\omega$ parameter represents the annual excess of water with repect to demands and it is used to classify reservoirs in water-stressed or non-water-stressed.

In reservoirs with high water stress ($\frac{\overline{D}}{\overline{I}} \gt 1 - \omega$), the hedged demand is a combination of a fixed percentage of the mean inflow, and a varying term depending on the ratio between the current and the average demand.

$$\hat{D}_t = \omega \cdot \overline{I} + \left( 1 - \omega \right) \frac{D_t}{\overline{D}} \cdot \overline{I}$$

In reservoirs with lower water stress ($\frac{\overline{D}}{\overline{I}} \lt 1 - \omega$), the hedged demand is the average surplus of inflow plus the current demand. 

$$\hat{D}_t = \overline{I} - \overline{D} + D_t$$

mHM considers $\omega$ a model parameter, but previous studies used fixed values (see Table 1).

> **Note**. The demand is further hedged in the computation of the reservoir release depending on the current filling of the reservoirs ($\kappa_t$)

**Calibration**

Following the procedure in [Shrestha et al. (2024)](https://agupubs.onlinelibrary.wiley.com/doi/10.1029/2023WR035433), I have calibrated all 5 parameters in the routine. Other papers used default values of some of the parameters, which I have used for the simulation with default parameters.

***Table 4**. Calibration parameters in the mHM reservoir.*

| parameter | description    | units | minimum | maximum | default |
| --------- | -------------- | ----- | ------- | ------- | ------- |
| $\alpha$  | Threshold in the degree of regulation that defines demand-controlled reservoirs (DOR > $\alpha$) | - | 0.0 | 5.0 | 0.5 |
| $\beta$   | It controls indirectly the proportion of inflow and demand in the releases | - | 0.5 | 3.0 | 1 |
| $\gamma$  | Ration between normal and total storage | - | 0 | 1 | $\frac{V_{0.9}}{V_{tot}}$ |
| $\lambda$  | It further controls hedging based on the current reservoir storage | - | 0.25 | 3.0 | 1 |
| $\omega$ | It controls hedging based on the ratio between the current and average demand | -  | 0| 1 | 0.1 |

The storage-outflow scatter plot in Figure 4 shows the different approach in this routine. The relationship between these two variables is not any more a clear function, but it is based on demands, which explains the dispersion in the scatter plot. The limitation of this routine is the amount of parameters to be fitted, and the need to estimate a demand time series.

<img src="../results/ResOpsUS/mhm/default/355_scatter_obs_sim.jpg" alt="reservoir 355" title="Linear model" width="800">

***Figure 4**. Comparison of the observed (blue) and default simulation (orange) of reservoir 355 with the mHM model.*

### 3.2 Runs

For each reservoir model, I have run 4 simulations. In all cases, the input is the timeseries of observed reservoir inflow and the observed storage at the beginning of the simulations. In previous tests I had used GloFAS/EFAS simulations to be able to be able to include in the sample of reservoirs those for which there is no observed inflow. This time I tried to avoid errors induced by the quality of the GloFAS/EFAS estimated inflow by using observations.

#### 3.2.1 Default parameters

As a benchmark, I have run a simulation with the default parameter values in the tables above. To run this simulations I've created the command [`simulate`](../../src/lisfloodreservoirs/simulate.py). Example use:

```Bash
# from the root folder of the lisflood-reservoirs repository
pip install -e .
simulate --config-file ./src/lisfloodreservoirs/config.yml
```

#### 3.2.2 Calibration

I have done three calibrations targeting different variables.

1. Univariate calibration of **outflow**. This would be the calibration closest to the LISFLOOD procedure that calibrates streamflow. I'll refer to it as _SCEUA-Q_.
2. Univariate calibration of **storage**. I'll refer to it as _SCEUA-S_.
3. Bivariate calibration of both **storage and outflow**. I'll refer to it as _SCEUA-QS_.

In all cases the objective function is the modified Kling-Gupta efficiency ($\text{KGE}$). For the bivariate calibration I compute a combined KGE from the individual KGE of each of the variables.

$$
\begin{align*}
\text{KGE} &= \sqrt{(1 - r)^2 + \left(1 - \frac{\mu_{\text{sim}}}{\mu_{\text{obs}}}\right)^2 + \left(1 - \frac{CV_{\text{sim}}}{CV_{\text{obs}}}\right)^2} \\
\text{KGE}_{\text{bivariate}} &= \sqrt{(1 - \text{KGE}_{\text{Q}})^2 + (1 - \text{KGE}_{\text{S}})^2}
\end{align*}
$$
    
Calibrations were done using the implementation of the SCEUA (Shuffle Complex Evolution - University of Arizona) algorithm in the Python library [`spotpy`](https://spotpy.readthedocs.io/en/latest/). In all cases I used the complete observed time series, and I set up the algorithm to run a maximum of 1000 iterations with 4 complexes. As explained above, the calibration parameters differ over reservoir models, both in number and meaning. 

To run the calibrations I've created the commad [`calibrate`](../../src/lisfloodreservoirs/calibrate.py). Example use:

```Bash
# from the root folder of the lisflood-reservoirs repository
pip install -e .
calibrate --config-file ./src/lisfloodreservoirs/config.yml
```

## 4 Results

> **Note**. Results are available in _Z:/nahaUsers/casadje/datasets/reservoirs/ResOpsUS/v1.1/results/_ or in the HPC in _/BGFS/DISASTER/casadje2/ResOpsUS/v1.1/results/_

### 4.1 Linear reservoir

#### 4.1.1 Performance

Figure 5 shows the performance in terms of KGE for the Linear reservoir in the four runs (different colour lines) for the three variables (each of the panels).

<img src="../results/ResOpsUS/linear/ecdf_KGE.jpg" width="800">

***Figure 5**. Empirical cumulative density curves of KGE for the linear reservoir in the four runs.*

The default parameterization of the linear reservoirs performs better in terms of outflow (median KGE 0.50) than storage (median KGE 0.27). The median performance of these two variables improves in all the calibrations but one case: the performance in terms of storage when calibrating outflow (median KGE 0.20). It indicates that the observed storage is a variables that informs better about the behaviour of the reservoir than the observed outflow. Actually, the bivariate calibration barely improves the performance of the univariate calibration of storage.

#### 4.1.2 Model parameters

The box plots in Figure 6 show the variability over the 90 reservoirs of the single model parameter in the linear reservoir (the residence time) for the four runs.

<img src="../results/ResOpsUS/linear/parameters_boxplot.jpg" width="300">

***Figure 6**. Comparison of the model parameters of the linear reservoir in the four runs.*

The default parameter tends to be larger than those obtained in the calibration of storage, and storage-outflow, which are rather similar between them. The calibration of outflow obtained the largest dispersion over reservoirs, but the lowest median value.

### 4.2 LISFLOOD

#### 4.2.1 Performance

Figure 7 shows the performance in terms of KGE for the LISFLOOD reservoir in the four runs (different colour lines) for the three variables (each of the panels).

<img src="../results/ResOpsUS/lisflood/ecdf_KGE.jpg" width="800">

***Figure 7**. Empirical cumulative density curves of KGE for the Lisflood reservoir in the four runs.*

The conclusions are similar to those presented for the linear reservoir. The default parameters perform better in terms of outflow (0.51 KGE) than storage (0.23 KGE). All the calibrations improve these values but in one case: the performance in terms of storage in the calibration of outflow (0.19 KGE). The univariate calibration of storage yields performance similar to the bivariate calibration, but the difference in outflow performance is a bit larger than in the linear reservoir

#### 4.2.2 Model parameters

Figure 8 compares the model parameters in the LISFLOOD reservoir routine for the four runs.

<img src="../results/ResOpsUS/lisflood/parameters_boxplot.jpg" width="1800">

***Figure 8**. Comparison of the model parameters of the Lisflood reservoir in the four runs.*

* The default value of $\alpha$, the fraction of the total storage that represents the flood storage limit, seems to be excessively large. This result is in line with the default parameterization of the Hanazaki routine, which uses a value of 0.75. It's important to remark that this parameter is not calibrated in the usual LISFLOOD calibration, so the definition of its default value is relevant.
* The default values of both $\beta$ and $\gamma$ (parameters that define the normal and normal-adjusted storage limits) seems to bemore accurate, but still are larger than the calibrated values. These two parameters show a large dispersion over reservoirs. From these two parameters, only $\gamma$ is fitted in the usual LISFLOOD calibration.
* The definition of the outflow limits (normal and flood/non-damaging) in LISFLOOD is uncertain. As default values I used the parameterization in Hanazaki, i.e., the non-damaging outflow is 30% ($\delta$) of the 100-year return period of inflow, and the normal outflow is the mean inflow. It seems like the 30% value in $\delta$ is a bit excessive, particularly when the focus is on the simulation of storage; this parameter is not fitted in the usual LISFLOOD calibration. On the contrary, the calibration of $epsilon$ shows that the normal outflow needs to be larger than the mean inflow in the majority of cases; this parameter is indeed fitted in the LISFLOOD calibration.
* The parameter $k$ is included in the LISFLOOD reservoir routine to limit the outflows depending on the inflow. It's default value of 1.2 limits outflows in the flood storage zone to 120% of the inflow at that timestep. The calibration proves that this limitation is not needed, yielding values around 3.

### 4.3 Hanazaki

#### 4.3.1 Performance

<img src="../results/ResOpsUS/hanazaki/ecdf_KGE.jpg" width="800">

***Figure 9**. Empirical cumulative density curves of KGE for the Hanazaki reservoir in the four runs.*

#### 4.3.2 Model parameters

<img src="../results/ResOpsUS/hanazaki/parameters_boxplot.jpg" width="1500">

***Figure 10**. Comparison of the model parameters of the Hanazaki reservoir in the four runs.*

### 4.4 mHM

#### 4.4.1 Performance

Figure 11 shows the performance in terms of KGE for the mHM reservoir in the four runs (different colour lines) for the three variables (each of the panels).

<img src="../results/ResOpsUS/mhm/ecdf_KGE.jpg" width="800">

***Figure 11**. Empirical cumulative density curves of KGE for the mHM reservoir in the four runs.*

As in the other models, the default parameterization of mHM performs better in terms of outflow (median KGE 0.65) than storage (median KGE 0.31). All the calibrations (even only calibrating storage) improve the outflow performance, but the gains are marginal compared with those obtained in the calibration of storage (either univariate or bivariate). Calibrating only storage is a good compromise as it improves significantly the simulation of storage (median KGE 0.78), and it also improves slightly the simulation of outflow (median KGE 0.70).

#### 4.4.2 Model parameters

Figure 12 compares the model parameters in the mHM reservoir routine for the four runs.

<img src="../results/ResOpsUS/mhm/parameters_boxplot.jpg" width="1500">

***Figure 12**. Comparison of the model parameters of the mHM reservoir in the four runs.*

* The calibrated values of $\omega$ are in general larger that the default value of 0.1, particularly in the univariate calibration of storage. It means an increased influence of the mean inflow over the current demand in the hedged demand.

* The default value of $\alpha$ is relatively accurate, perhaps a bit too small. The calibration indicates that the outflow from reservoirs with a degree of regulation larger or equal than 0.7 approximately are controlled only by the demand.

* The $\beta$ parameter affects the partition of releases between demand and inflow for those reservoirs with a degree of regulation below $\alpha$. Values over 1 (default) increase the influence of inflow over demand in the reservoir release. All the calibrations show that $\beta$ should take values larger than 1.

* The calibration of $\gamma$ shows that the normal storage is in general around 60% of the total storage, which is significantly smaller than the default value of 0.85.

* The $\lambda$ parameter modifies the effect of the current reservoir storage in the outflow. The default value of 1 means that $\lambda$ has no effect. All the calibrations take median values around 1.5, which means that releases are increased when the reservoir is over the normal storage, and reduced when the reservoir is below the normal storage.

### 4.5 Model comparison

Figure 13 is a reorganization of the line plots in figures 5, 7, 9 and 11 to compare the performance of the four reservoir models.

<img src="../results/ResOpsUS/ecdf_KGE_default.jpg" width="800">
<img src="../results/ResOpsUS/ecdf_KGE_SCEUA-S.jpg" width="800">
<img src="../results/ResOpsUS/ecdf_KGE_SCEUA-Q.jpg" width="800">

***Figure 13**. Comparison of the performance of the four reservoir models (colour lines) for each of the runs (rows) and variables (columns).*

> **Note**. A similar plot for the bivariate calibration can be found [here](../results/Results/ResOpsUS/ecdf_KGE_SCEUA-QS.jpg).

In the simulation with **default parameters** (top row) Hanazaki and mHM slightly stand out from the other two models, particularly in the simulation of outflow. It's not surprising that mHM is the best model in terms of outflow, as this model has more information than the rest because it uses a demand time series derived from the observed outflow. I remark here that the Hanazaki routine is meant to be used with default parameters, which are proved here to perform reasonably well. The performance of the linear and LISFLOOD models is very similar, despite the former being so simple. However, the overall differences in performance among models are relatively small. 

It is when parameters are calibrated that the differences between models emerge. The **calibration of storage** (middle row) shows large differences in performance. The mHM model stands out as the best approach both in terms of storage and outflow. The LISFLOOD and linear reservoirs perform similarly in both variables. Hanazaki performs clearly worse in terms of storage, but similarly in terms of outflow. In the previous sections we've seen that this calibration outperforms the default parameterization in both storage and outflow for all the models.

The **calibration of outflow** (bottom row) yields a similar ranking of model performance, but the differences are narrower. mHM is the best model, followed by LISFLOOD and linear reservoir (that perform similarly), and Hanazaki is slightly worse. Previous sections showed that the calibration of outflow produces a loss in storage performance in some of the models compared with the default parameterization.

## Conclusions


