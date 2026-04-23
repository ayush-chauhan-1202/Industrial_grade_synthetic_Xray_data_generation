# Industrial_grade_synthetic_Xray_data_generation

This work intends to address a major challenges that many researchers face while building AI solutions for xray systems, which is:
1. Avaialability of industrial grade CT dataset that captures realistic defects like micro-porosity, delamination, crach etc
2. Provides a control of standardization and customization of dataset


In this work, i have tried to define a customizable xray dataset and related synthetic but realistic data. Here's the broad outline:
1. 3D Phantom creation with custom shape, defects, materials etc
2. Noise modelling including application of poisson and gaussian noise, scatter and beam hardening artifacts.
3. Cone-Beam Geometry Forward projection model using trilinear interploation to generate Projection set from Phantom Geometry.
4. Visualization and saving options.


