import pickle
from typing import Tuple

import bentoml
import bentoml.sklearn
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from hydra.utils import to_absolute_path
from omegaconf import DictConfig
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from yellowbrick.cluster import KElbowVisualizer


def get_pca_model(data: pd.DataFrame) -> PCA:
    pca = PCA(n_components=3)
    pca.fit(data)

    save_path = to_absolute_path("processors/PCA.pkl")
    pickle.dump(pca, open(save_path, "wb"))

    return pca


def reduce_dimension(df: pd.DataFrame, pca: PCA) -> pd.DataFrame:
    return pd.DataFrame(pca.transform(df), columns=["col1", "col2", "col3"])


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

    data = pd.read_csv(
        to_absolute_path(config.intermediate.path),
    )

    pca = get_pca_model(data)
    pca_df = reduce_dimension(data, pca)

    k_best = get_best_k_cluster(pca_df)
    model = get_clusters_model(pca_df, k_best)

    save_model(model)
