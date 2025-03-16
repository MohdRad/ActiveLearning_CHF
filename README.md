Aided Active Learning for Enhanced Critical Heat Flux Prediction

NURETH-21: Aided Active Learning for Enhanced Critical Heat Flux Prediction

Nabila, U.M., Radaideh, M.I., Burnett, L.A., Lin, L., Radaideh, M.I. (2025). “Aided Active Learning for Enhanced Critical Heat Flux Prediction,” In: 21st International Topical Meeting on Nuclear Reactor Thermal Hydraulics (NURETH-21), Busan, Korea, August 31 – September 5, 2025.
 
This repository contains the implementation of Active Learning (AL) with Variational Inference (VI) in Feedforward Neural Networks (vFNN) for Critical Heat Flux (CHF) prediction. 
We evaluate the performance of:

    Variational Feedforward Neural Networks (vFNN)
    Standard Feedforward Neural Networks (FNN)
    Random Forest (RF) models
These models are assessed on a Critical Heat Flux (CHF) dataset, using Active Learning Query Strategies.

Active Learning Framework:
Active Learning (AL) is used to select the most informative samples instead of random sampling.
Two query strategies are used:

    Query Strategy 1 (Uncertainty-based Selection):
        Selects samples where the model is least confident.
        Uses Variational Inference (VI) in Bayesian Neural Networks to estimate uncertainty.
    Query Strategy 2 (Error-based Selection):
        Selects samples where the prediction error is highest.
        Applied to FNN, and RF models.
