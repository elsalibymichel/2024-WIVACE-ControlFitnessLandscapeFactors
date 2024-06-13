# 2024-WIVACE-ControlFitnessLandscapeFactors

This repo contains the framework needed to replicate the results of the research paper 
[*Factors Impacting Landscape Ruggedness
in Control Problems: a Case Study*](TO_APPEAR)
submitted at the WIVACE conference 2024.

In this repo we provide:

- A Java class for sampling the genotype space of the Rastrigin function using the random segment based sampling procedure.
The ```main``` of ```io.github.elsalibymichel.core/src/main/RastriginRandomSampling.java``` takes several arguments, ```-h``` shows all the options.

- A Java class for sampling the genotype space of the 2D navigation task using the sampling procedure based on the random segments.
The ```main``` of ```io.github.elsalibymichel.core/src/main/NavRandomSampling.java``` takes several arguments, ```-h``` shows all the options.

- A folder, ```2DNav_Optima-Based-Sampling/```, containing the configuration files for evolving controllers for the 2D navigation task with
[```JGEA```](https://github.com/ericmedvet/jgea.git) (Develop branch, 2.6.2-SNAPSHOT).

- A Java class for sampling the genotype space of the 2d navigation task using the sampling procedure based on the local optima and adaptive sampling rate.
The ```main``` of ```io.github.elsalibymichel.core/src/main/BestGenotypesLandscapeCharacterizer.java``` takes
as arguments a file```.csv``` containing the data obtained using the configuration files in the ```2DNav_Optima-Based-Sampling/```, and a string specifying the factor to be considered.

- A Python file, ```RuggednessIndices.py```, containing the functions that we used to compute the ruggedness indices.

More details can be found in the paper.
