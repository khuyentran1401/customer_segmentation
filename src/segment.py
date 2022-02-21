import pickle
from typing import Tuple

import bentoml.sklearn
import matplotlib.pyplot as plt
import pandas as pd
from omegaconf import DictConfig
from sklearn.cluster import KMeans
from yellowbrick.cluster import KElbowVisualizer


def get_best_k_cluster(pca_df: pd.DataFrame) -> pd.DataFrame:

    fig = plt.figure(figsize=(10, 8))
    fig.add_subplot(111)

    elbow = KElbowVisualizer(KMeans(), metric="distortion")
    elbow.fit(pca_df)

    k_best = elbow.elbow_value_

    return k_best


def get_clusters_model(
    pca_df: pd.DataFrame, k: int
) -> Tuple[pd.DataFrame, pd.DataFrame]:

    model = KMeans(n_clusters=k)

    # Fit model
    return model.fit(pca_df)


def save_model(model):
    bentoml.sklearn.save("customer_segmentation_kmeans", model)


def segment(config: DictConfig) -> None:

    pca_df = pd.read_csv(config.final.path)

    k_best = get_best_k_cluster(pca_df)
    model = get_clusters_model(pca_df, k_best)

    save_model(model)
