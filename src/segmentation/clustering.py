import joblib
from sklearn.cluster import KMeans

def train_kmeans_segmentation(X_scaled, n_clusters=3, save_path=None):
    """Trains K-Means clustering model for customer persona segmentation."""
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(X_scaled)
    if save_path:
        joblib.dump(kmeans, save_path)
    return kmeans, cluster_labels