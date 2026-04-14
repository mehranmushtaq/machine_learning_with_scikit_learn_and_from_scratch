## Unsupervised Machine Learning

This directory contains implementations and experiments with core unsupervised learning algorithms using Scikit-Learn.



## Contents

## K-Means Clustering
	•	Cluster formation using centroid-based approach
	•	Optimal K selection using:
	•	Elbow Method
	•	Silhouette Score



## DBSCAN
	•	Density-based clustering
	•	Handles:
	•	Non-linear cluster shapes
	•	Noise (outliers)


## Hierarchical Clustering
	•	Agglomerative clustering
	•	Visualized using dendrograms
	•	Helps understand cluster hierarchy



## Datasets Used

	•	make_blobs – for clear cluster separation
	•	make_moons – for non-linear clustering
	•	Iris dataset



## Key Observations

	•	K-Means performs well on spherical clusters but fails on non-linear data
	•	DBSCAN successfully identifies complex shapes and noise
	•	Hierarchical clustering provides better interpretability via dendrograms
	•	Feature scaling significantly impacts clustering performance



## How to Run
	1.	Open any notebook (.ipynb)
	2.	Run all cells sequentially
	3.	Observe clustering behavior and visualizations



## Files
	•	k_means.ipynb
	•	dbscan.ipynb
	•	hierarchical_clustering.ipynb



## Purpose

This module is part of a larger Machine Learning learning repository, focused on building strong intuition in clustering techniques through visualization and experimentation.
