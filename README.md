# Modelling-Crystallization-Using-ML-Augmented-Simulations
In this project we attempt to study crystallization and make an attempt to distinguish an Ordered State from Unordered State, by obtaining collective variables which we would exploit to study difference between two states.
# Progress Sequence

 1. # Code can be found in distance.py
We first compute a Distance Matrix where each element represents the distance between two atoms of a Simple NaCl crystal ( we consider 108 atoms each of Na and Cl in our crystal).
Next we calculate the eigen values and corresponding eigen vectors for each of 102 frames .
We Reduce these eigen values to 2 using PCA technique.

