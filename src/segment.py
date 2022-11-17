from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from helper import load_config
from prefect import flow, task
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from yellowbrick.cluster import KElbowVisualizer
from omegaconf import DictConfig
import matplotlib as mpl
from sqlalchemy import create_engine


@task
def read_processed_data(config: DictConfig):
    connection = config.connection
    engine = create_engine(
        f"postgresql://{connection.user}:{connection.password}@{connection.host}/{connection.database}",
    )
    query = f'SELECT * FROM "{config.data.intermediate}"'
    df = pd.read_sql(query, con=engine)
    return df


@task
def get_pca_model(data: pd.DataFrame) -> PCA:

    pca = PCA(n_components=3)
    pca.fit(data)
    return pca


@task
def reduce_dimension(df: pd.DataFrame, pca: PCA) -> pd.DataFrame:
    return pd.DataFrame(pca.transform(df), columns=["col1", "col2", "col3"])


@task
def get_3d_projection(pca_df: pd.DataFrame) -> dict:
    """A 3D Projection Of Data In The Reduced Dimensionality Space"""
    return {"x": pca_df["col1"], "y": pca_df["col2"], "z": pca_df["col3"]}



@task
def get_best_k_cluster(
    pca_df: pd.DataFrame, image_path: str, elbow_metric: str
) -> pd.DataFrame:

    fig = plt.figure(figsize=(10, 8))
    fig.add_subplot(111)

    elbow = KElbowVisualizer(KMeans(), metric=elbow_metric)
    elbow.fit(pca_df)
    elbow.fig.savefig(image_path)

    k_best = elbow.elbow_value_
    return k_best


@task
def predict(
    pca_df: pd.DataFrame, k: int, model: dict
) -> Tuple[pd.DataFrame, pd.DataFrame]:

    model_args = dict(model.args)
    model_args["n_clusters"] = k

    model = KMeans(**model_args)

    # Predict
    return model.fit_predict(pca_df)


@task
def insert_clusters_to_df(df: pd.DataFrame, clusters: np.ndarray) -> pd.DataFrame:
    return df.assign(clusters=clusters)


@task
def save_segmented_df(df: pd.DataFrame, config: DictConfig):
    connection = config.connection
    engine = create_engine(
        f"postgresql://{connection.user}:{connection.password}@{connection.host}/{connection.database}",
    )

    df.to_sql(name=config.data.segmented, con=engine, if_exists="replace", index=False)


@task
def plot_clusters(
    pca_df: pd.DataFrame, preds: np.ndarray, projections: dict, image_path: str
) -> None:
    pca_df["clusters"] = preds

    plt.figure(figsize=(10, 8))
    ax = plt.subplot(111, projection="3d")
    ax.set_title("the plot of the clusters")
    ax.scatter(
        projections["x"],
        projections["y"],
        projections["z"],
        s=40,
        c=pca_df["clusters"],
        marker="o",
        cmap="Accent",
    )
    plt.savefig(image_path)


@flow
def segment() -> None:
    mpl.use("Agg")
    config = load_config()
    data = read_processed_data(config)
    pca = get_pca_model(data)
    pca_df = reduce_dimension(data, pca)

    projections = get_3d_projection(pca_df)

    k_best = get_best_k_cluster(pca_df, config.image.elbow, config.elbow_metric)
    prediction = predict(pca_df, k_best, config.segment)

    data = insert_clusters_to_df(data, prediction)
    save_segmented_df(data, config)

    plot_clusters(pca_df, prediction, projections, config.image.clusters)


if __name__ == "__main__":
    segment()
