# Unsupervised Machine Learning

This directory contains hands-on implementations of unsupervised learning clustering algorithms using scikit-learn, covering theory, practical usage, and visual evaluation.

## Notebooks

### 1. k_means.ipynb
Introduction to K-Means clustering on the Iris dataset.
- Loads and visualizes the Iris dataset with seaborn scatterplots
- Applies StandardScaler for feature normalization
- Reduces dimensionality to 2D using PCA (sklearn.decomposition.PCA)
- Uses the Elbow Method (WCSS) to explore values of k from 1–10
- Fits K-Means with n_clusters=3 and plots cluster assignments with centroids marked as red ✕

### 2. k_means_clustering.ipynb
Deeper dive into K-Means with synthetic blob data and optimal k selection.
- Generates 1000-sample, 4-center blob data using make_blobs
- Visualizes raw unlabeled data and post-clustering color-coded assignments (K=4)
- Elbow Method over k=1–20 with WCSS plot
- Finds optimal k programmatically using the kneed library (KneeLocator) → optimal K = 4
- Silhouette Score analysis over k=2–20 to validate cluster quality
- Silhouette score peaks at k=4, confirming the elbow result

### 3. hiearchichal_clustering.ipynb
Hierarchical (Agglomerative) Clustering on the Iris dataset.
- Loads and scales Iris data with StandardScaler
- Visualizes scaled data as a scatterplot
- Builds a linkage matrix using Ward's method (scipy.cluster.hierarchy.linkage)
- Renders a full dendrogram to visually determine the number of clusters
- Applies AgglomerativeClustering(n_clusters=2) and plots results

### 4. dbscan.ipynb
DBSCAN (Density-Based Spatial Clustering) — great for non-linear shapes.

**Linear Data (Iris):**
- Loads and scales Iris data
- Applies DBSCAN(eps=0.8, min_samples=5) → correctly separates 2 clusters

**Non-Linear Data (Moons):**
- Generates crescent/moon-shaped data using make_moons(n_samples=300, noise=0.05)
- Demonstrates that K-Means fails on non-linear shapes (splits along the wrong axis)
- Shows DBSCAN succeeds with eps=0.5, min_samples=5 → correctly identifies the two moon shapes

### 5. pca.ipynb
Principal Component Analysis (PCA) for dimensionality reduction on the Iris dataset.
- Loads the Iris dataset and creates a DataFrame using pandas
- Applies StandardScaler to normalize all features before decomposition
- Fits PCA with n_components=2, reducing from 4 features to 2 principal components
- PC1 and PC2 together explain ~95.8% of total variance (72.96% + 22.85%)
- Prints PCA component loadings to interpret feature contributions per principal component
- Produces a 2D scatter plot ("PCA For Iris Dataset") color-coded by species, showing clear separation of setosa from versicolor/virginica

## Concepts Covered

| Concept | Notebooks |
|---|---|
| K-Means Clustering | k_means.ipynb, k_means_clustering.ipynb |
| PCA Dimensionality Reduction | k_means.ipynb, pca.ipynb |
| Elbow Method (WCSS) | k_means.ipynb, k_means_clustering.ipynb |
| Silhouette Score | k_means_clustering.ipynb |
| KneeLocator (kneed) | k_means_clustering.ipynb |
| Hierarchical / Agglomerative Clustering | hiearchichal_clustering.ipynb |
| Dendrogram Visualization | hiearchichal_clustering.ipynb |
| DBSCAN | dbscan.ipynb |
| Non-Linear Clustering | dbscan.ipynb |
| Explained Variance Ratio | pca.ipynb |
| Feature Loading Analysis | pca.ipynb |

## Libraries Used
- scikit-learn
- seaborn
- matplotlib
- pandas
- numpy
- scipy
- kneed

## Getting Started

```bash
# Clone the parent repo
git clone https://github.com/mehranmushtaq/Machine-Learning-with-scikit-learn-and-from-scratch.git

cd "Machine-Learning-with-scikit-learn-and-from-scratch/unsupervised ml"

# Install dependencies
pip install scikit-learn seaborn matplotlib pandas numpy scipy kneed

# Launch Jupyter
jupyter notebook
```

## Key Takeaways

	•	K-Means works well on convex, blob-shaped clusters but requires choosing k upfront
	•	The Elbow Method and Silhouette Score are complementary tools for selecting k
	•	Hierarchical clustering doesn’t require specifying k and produces interpretable dendrograms
	•	DBSCAN handles arbitrary shapes and noise, outperforming K-Means on non-linear data
	•	PCA reduces high-dimensional data to fewer components while preserving most variance, making it ideal for visualization and preprocessing before clustering
