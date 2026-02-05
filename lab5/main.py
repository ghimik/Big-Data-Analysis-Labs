import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import KernelPCA
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score

from scipy.stats import zscore
import joblib

from KMeansSearch import MyKMeans

DATA_PATH = "train.csv"
df = pd.read_csv(DATA_PATH)

target = df["Class"]
X = df.drop(columns=["id", "Class"])

print("Source EDA (117k rows):")
print(df.info())
print(df.describe())

z_scores = np.abs(zscore(X))
X_no_outliers = X[(z_scores < 3).all(axis=1)]
target_no_outliers = target.loc[X_no_outliers.index]

SAMPLE_SIZE = 10000
rng = np.random.default_rng(42)
sample_idx = rng.choice(X_no_outliers.shape[0], min(SAMPLE_SIZE, X_no_outliers.shape[0]), replace=False)

X_no_outliers = X_no_outliers.iloc[sample_idx]
target_no_outliers = target_no_outliers.iloc[sample_idx]

print("\nSample EDA (10k rows):")
print(X_no_outliers.describe())
print(X_no_outliers.info())

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_no_outliers)

kernels = ["linear", "poly", "rbf", "sigmoid", "cosine"]
kpca_results = {}
metrics_table = []


for kernel in kernels:
    kpca = KernelPCA(n_components=2, kernel=kernel)
    X_kpca = kpca.fit_transform(X_scaled)
    kpca_results[kernel] = (kpca, X_kpca)
    sil = silhouette_score(X_kpca, target_no_outliers)
    ch = calinski_harabasz_score(X_kpca, target_no_outliers)
    db = davies_bouldin_score(X_kpca, target_no_outliers)
    
    metrics_table.append({
        "Kernel": kernel,
        "Silhouette": sil,
        "Calinski-Harabasz": ch,
        "Davies-Bouldin": db
    })

metrics_df = pd.DataFrame(metrics_table)
print(metrics_df)

plt.figure(figsize=(15, 8))
for i, kernel in enumerate(kernels, 1):
    plt.subplot(2, 3, i)
    plt.scatter(kpca_results[kernel][1][:, 0],
                kpca_results[kernel][1][:, 1],
                c=target_no_outliers, cmap="coolwarm", s=10)
    plt.title(f"KernelPCA ({kernel})")
plt.tight_layout()
plt.show()

kpca_linear = kpca_results["linear"][0]
lambdas = kpca_linear.eigenvalues_
explained_variance_ratio = lambdas / np.sum(lambdas)
lost_variance = 1 - np.sum(explained_variance_ratio[:2])
print("Explained variance ratio (first 2 components):", explained_variance_ratio[:2])
print("Lost variance:", lost_variance)

tsne = TSNE(n_components=2, perplexity=30)
X_tsne = tsne.fit_transform(X_scaled)
sil_tsne = silhouette_score(X_tsne, target_no_outliers)
ch_tsne = calinski_harabasz_score(X_tsne, target_no_outliers)
db_tsne = davies_bouldin_score(X_tsne, target_no_outliers)

print("t-SNE Metrics:")
print(f"Silhouette: {sil_tsne:.3f}")
print(f"Calinski-Harabasz: {ch_tsne:.3f}")
print(f"Davies-Bouldin: {db_tsne:.3f}")


plt.figure(figsize=(6, 5))
plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=target_no_outliers, cmap="coolwarm", s=10)
plt.title("t-SNE")
plt.show()

n_clusters = 2  
my_kmeans = MyKMeans(n_clusters=n_clusters)
my_kmeans.fit(X_scaled)
labels_my = my_kmeans.labels_

plt.figure(figsize=(6, 5))
plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=labels_my, cmap="viridis", s=10)
plt.title("MyKMeans Clusters")
plt.show()

kmeans_lib = KMeans(n_clusters=n_clusters)
kmeans_lib.fit(X_scaled)
labels_lib = kmeans_lib.labels_

plt.figure(figsize=(6, 5))
plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=labels_lib, cmap="viridis", s=10)
plt.title("Library KMeans Clusters")
plt.show()

sil_my = silhouette_score(X_scaled, labels_my)
ch_my = calinski_harabasz_score(X_scaled, labels_my)
db_my = davies_bouldin_score(X_scaled, labels_my)

sil_lib = silhouette_score(X_scaled, labels_lib)
ch_lib = calinski_harabasz_score(X_scaled, labels_lib)
db_lib = davies_bouldin_score(X_scaled, labels_lib)

print("\nMyKMeans Metrics:")
print(f"Silhouette: {sil_my:.3f}, Calinski-Harabasz: {ch_my:.3f}, Davies-Bouldin: {db_my:.3f}")

print("\nLibrary KMeans Metrics:")
print(f"Silhouette: {sil_lib:.3f}, Calinski-Harabasz: {ch_lib:.3f}, Davies-Bouldin: {db_lib:.3f}")


joblib.dump(kpca_results["rbf"][0], "kpca_rbf_model.joblib")
loaded_kpca = joblib.load("kpca_rbf_model.joblib")
X_loaded = loaded_kpca.transform(X_scaled)
print("Model loaded and applied successfully.")