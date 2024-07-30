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

$$NOR_{ub} = \min \left( \max \left( A + B \cdot sin \, 2 \pi \omega t + C \cdot cos \, 2 \pi \omega t, \; \hat{S}_{min} \right), \; \hat{S}_{max} \right)$$
$$NOR_{lb} = \min \left( \max \left( a + b \cdot sin \, 2 \pi \omega t + c \cdot cos \, 2 \pi \omega t, \; \hat{s}_{min} \right), \; \hat{s}_{max} \right)$$

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
$$\epsilon_t = h + i \frac{\hat{S}_t - NOR_{lb}}{NOR_{ub} - NOR_{lb}} + j \cdot \bar{I}$$

Those two models refer to the standardised release, which needs to be reconverted to actual release. The release function including the three possible reservoir states (below, within or above NOR) is:

$$\ddot{R}_t = \begin{cases}
    R_{min} & if \quad \hat{S}_t < NOR_{lb} \\
    \min \left( \bar{I} \cdot \left( \tilde{R}_t + \epsilon_t \right) + \bar{I} , R_{max} \right) & if \quad NOR_{lb} \leq \hat{S}_t \leq NOR_{ub} \\
    \min \left( S_{cap} \cdot \left( \hat{S}_t - NOR_{ub} \right) + I_t , R_{max} \right) & if \quad \hat{S}_t > NOR_{ub}
\end{cases}
$$

The last step is to check the feasibility of the release $\ddot{R}_t$ in terms of mass balance:

$$R_t = \max \left( \min \left( \ddot{R}_t, I_t + S_t \right), I_t + S_t - S_{cap} \right)$$

<font color='steelblue'>I have changed the release routine because the original definition generates very noisy releases when the reservoir storage fluctuate around the lower bound of the normal operating rule ($NOR_{lb}$). When the storage is slightly over $NOR_{lb}$, the release is a function of the harmonic and the linear models; however, when the storage is slightly below $NOR_{lb}$, the release is a constant $R_{min}$. It means that from one step to the next there is a big difference in releases.
    
<font color='steelblue'>Instead, I change the release function below $NOR_{lb}$ to be the minimum value between the inflow ($I$) and a linear function of $R_{min}$ and the distance from $NOR_{lb}$. In this way, the changes in release are smooth, and the release is at most equal to the inflow, so the resevoir does not keep emptying:
    
$$R_{NOR} = \bar{I} \cdot \left( \tilde{R}_t + \epsilon_t \right) + \bar{I}$$
$$\ddot{R}_t = \begin{cases}
    \min \left( R_{min} + \frac{\hat{S}_t}{NOR_{lb}} \cdot \left( R_{NOR} - R_{min} \right) , I \right) & if \quad \hat{S}_t < NOR_{lb} \\
    \min \left( R_{NOR} , R_{max} \right) & if \quad NOR_{lb} \leq \hat{S}_t \leq NOR_{ub} \\
    \min \left( S_{cap} \cdot \left( \hat{S}_t - NOR_{ub} \right) + I_t , R_{max} \right) & if \quad \hat{S}_t > NOR_{ub}
\end{cases}
$$