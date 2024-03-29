# RandMVLearn
Scalable Randomized Kernel Methods for Multiview Data Integration and Prediction with Individual and Group Variable Selection

We develop scalable randomized kernel methods for jointly associating data from multiple sources and simultaneously predicting an outcome or classifying a unit into one of two or more classes. The proposed methods model nonlinear relationships in multiview data together with predicting a clinical outcome and are capable of identifying variables or groups of variables that best contribute to the relationships among the views. We use the idea that random Fourier bases can approximate shift-invariant kernel functions to construct nonlinear mappings of each view and we use these mappings and the outcome variable to learn view-independent low-dimensional representations. Through simulation studies, we show that the proposed methods outperform several other linear and nonlinear methods for multiview data integration. When the proposed methods were applied to gene expression, metabolomics, proteomics, and lipidomics data pertaining to COVID-19, we identified several molecular signatures for COVID-19 status and severity. Results from our real data application and simulations with small sample sizes suggest that the proposed methods may be useful for small sample size problems. 

This package depends on the following Python modules:
- torch
- numpy
- random
- itertools
- math
- time
- joblib
- matplotlib
- scikit-learn

Please ensure these modules are downloaded and available for use.

Paper can be found here: https://arxiv.org/abs/2304.04692. 

Send an email to ssafo@umn.edu if you have any questions or notice a bug. Thanks!
