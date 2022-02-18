from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from yellowbrick.cluster import KElbowVisualizer
import os 

def get_pca_model(data: pd.DataFrame) -> PCA:

    pca = PCA(n_components=3)
    pca.fit(data)
    return pca


def reduce_dimension(df: pd.DataFrame, pca: PCA) -> pd.DataFrame:
    return pd.DataFrame(pca.transform(df), columns=["col1", "col2", "col3"])


def get_3d_projection(pca_df: pd.DataFrame) -> dict:
    """A 3D Projection Of Data In The Reduced Dimensionality Space"""
    return {"x": pca_df["col1"], "y": pca_df["col2"], "z": pca_df["col3"]}


def get_best_k_cluster(pca_df: pd.DataFrame, image_path: str) -> pd.DataFrame:

    fig = plt.figure(figsize=(10, 8))
    fig.add_subplot(111)

    elbow = KElbowVisualizer(KMeans(), metric="distortion")
    elbow.fit(pca_df)

    os.makedirs('image', exist_ok=True)
    elbow.fig.savefig(image_path)

    k_best = elbow.elbow_value_

    return k_best


def get_clusters(
    pca_df: pd.DataFrame, k: int
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    model = KMeans(n_clusters=k)

    # Fit and predict model
    return model.fit_predict(pca_df)


def plot_clusters(
    pca_df: pd.DataFrame, preds: np.ndarray, projections: dict, image_path: str
) -> None:
    pca_df["clusters"] = preds

    plt.figure(figsize=(10, 8))
    ax = plt.subplot(111, projection="3d")
    ax.scatter(
        projections["x"],
        projections["y"],
        projections["z"],
        s=40,
        c=pca_df["clusters"],
        marker="o",
        cmap="Accent",
    )
    ax.set_title("The Plot Of The Clusters")

    plt.savefig(image_path)


def segment() -> None:

    data = pd.read_csv("data/intermediate/processed.csv")
    pca = get_pca_model(data)
    pca_df = reduce_dimension(data, pca)

    projections = get_3d_projection(pca_df)

    k_best = get_best_k_cluster(
        pca_df,
        image_path="image/elbow.png",
    )
    preds = get_clusters(pca_df, k_best)

    plot_clusters(
        pca_df,
        preds,
        projections,
        image_path="image/cluster.png",
    )


if __name__ == "__main__":
    segment()
