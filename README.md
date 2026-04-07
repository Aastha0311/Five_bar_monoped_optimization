# Five-bar-monoped-optimization

## Pre-requisites
The Following libraries are required: \
`numpy, scipy, matplotlib, cma, pandas`


## Quick Start Guide

## 1. Installation: 
Install the following packages:
```
pip install numpy scipy matplotlib cma pandas
```
## 2. Stage 1: Actuator Optimization: 
Run the python script in the actuator optimization directory to obtain optimal gearbox parameters for all motors:
```
python best_gearbox.py
```
## 3. Stage 2: Co-Design Optimization: 
Run the python script in the components directory:

```
python cmaes.py
```


## Results

Video of the paper along with simulation videos are given here. Optimised mass and efficiency plots for gear ratios 4:1 to 35:1.


## 🎥 Video

[Watch Video](https://youtu.be/s4FwyKOgQYg)

## Gearbox Optimization Results

![Gearbox Plot](results/Act_opt_results/act_opt_plots/gearbox_plots_MAD_M6C12.png)

*Figure: Optimized gearbox parameters (MAD M6C12 configuration).*


![Gearbox Plot](results/Act_opt_results/act_opt_plots/gearbox_plots_VT8020.png)

*Figure: Optimized gearbox parameters (VT8020 configuration).*

![Gearbox Plot](results/Act_opt_results/act_opt_plots/gearbox_plots_MN8014.png)

*Figure: Optimized gearbox parameters (T-Motor MN8014 configuration).*


![Gearbox Plot](results/Act_opt_results/act_opt_plots/gearbox_plots_U8.png)

*Figure: Optimized gearbox parameters (T-Motor U8 configuration).*


![Gearbox Plot](results/Act_opt_results/act_opt_plots/gearbox_plots_U10.png)

*Figure: Optimized gearbox parameters (T-Motor U10 configuration).*



![Gearbox Plot](results/Act_opt_results/act_opt_plots/gearbox_plots_U12.png)

*Figure: Optimized gearbox parameters (T-Motor U12 configuration).*
