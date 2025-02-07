# ActiveLearning
Active Learning with Variational Inference for Enhanced CHF Prediction
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
