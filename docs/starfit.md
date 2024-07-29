# [Starfit ](https://github.com/IMMM-SFA/starfit)

The full implementation of the Starfir reservoir routine requires:

* Reservoir attributes:
    * $S_{cap}$: total reservoir storage [hm3]
    * $\bar{I}$: average reservoir inflow [hm3/week]
* Daily time series:
    * $S_t$: reservoir storage [hm3]
    * $I_t$: reservoir inflow [m3/s]
    * $R_t$: reservoir release [m3/s]
    
The daily time series are aggregated into weekly time series, units are converted into volume [hm3], and gaps are tried to be filled in by mass balance.

## Storage normal operating range (NOR)

The first step is to normalise the storage time series according to reservoir capacity, so the variable is identical for all reservoirs and parameters can be regionalised.

$$\hat{S}_t = \frac{S_t}{S_{cap}}$$

Two normal operating rules are fitted using an harmonic function. These two rules define the upper and lower bounds of the normal reservoir filling for every week of the year.

$$\text{NOR}_{\text{ub}} = min \left( max \left( A + B \cdot sin\,2 \pi \omega t + C \cdot cos\,2 \pi \omega t, \hat{S}_{min} \right), \hat{S}_{max} \right)$$
$$\text{NOR}_{\text{lb}} = min \left( max \left( a + b \cdot sin\,2 \pi \omega t + c \cdot cos\,2 \pi \omega t, \hat{s}_{min} \right), \hat{s}_{max} \right)$$

Each of the NOR harmonics has 5 parameters: 3 defining the harmonic, and 2 capping the maximum and minimum values of that NOR. Therefore, the storage routine has 10 parameters.

> $\omega$ is the frequency. As the time series are weekly: $\omega=\frac{1}{52}$. <br>
> $t$ is the epistemologic week of the year.

## Release function

The discharge time series (both inflow and release) are standardised using the mean inflow, so that parameters can be regionalised between reservoirs.

$$\hat{R}_t = \frac{R_t - \bar{I}}{\bar{I}}$$
$$\hat{I}_t = \frac{I_t - \bar{I}}{\bar{I}}$$

When the reservoir filling is within the NOR, the routine models the standardised release ($\hat{R}_t$) as the sum of an harmonic model ($\tilde{R}_t$) that defines the seasonality and a linear model ($\epsilon_t$) that applies a correction based on the current reservoir filling and inflow.

$$\hat{R}_t = \tilde{R}_t + \epsilon_t$$
$$\tilde{R}_t = d \cdot sin\,2 \pi \omega t + e \cdot sin\,4 \pi \omega t + f \cdot cos\,2 \pi \omega t + g \cdot cos\,4 \pi \omega t$$
$$\epsilon_t = h + i \frac{\hat{S}_t - \text{NOR}_{\text{lb}}}{\text{NOR}_{\text{ub}} - \text{NOR}_{\text{lb}}} + j \cdot \bar{I}$$

Those two models refer to the standardised release, which needs to be reconverted to actual release. The release function including the three possible reservoir states (below, within or above NOR) is:

$$\ddot{R}_t = R_{\text{min}} \quad \text{if} \quad \hat{S}_t < \text{NOR}_{\text{lb}}$$
$$\ddot{R}_t = \min \left( \bar{I} \left( \tilde{R}_t + \epsilon_t \right) + \bar{I} , R_{\text{max}} \right) \quad \text{if} \quad \text{NOR}_{\text{lb}} \leq \hat{S}_t \leq \text{NOR}_{\text{ub}}$$
$$\ddot{R}_t = \min \left( S_{\text{cap}} \left( \hat{S}_t - \text{NOR}_{\text{ub}} \right) + I_t , R_{\text{max}} \right) \quad \text{if} \quad \hat{S}_t > \text{NOR}_{\text{ub}}$$