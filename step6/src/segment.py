from typing import Tuple

import hydra
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from omegaconf import DictConfig
from prefect import Flow, task
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from yellowbrick.cluster import KElbowVisualizer
import os 
from prefect.engine.results import LocalResult
import wandb 
from prefect.engine.serializers import PandasSerializer

FINAL_OUTPUT = LocalResult(
    "data/final/",
    location="{task_name}.csv",
    serializer=PandasSerializer("csv", serialize_kwargs={"index": False}),
)

@task(result=LocalResult("models/", location="pca.pkl"))
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
def get_best_k_cluster(pca_df: pd.DataFrame, image_path: str) -> pd.DataFrame:

    fig = plt.figure(figsize=(10, 8))
    fig.add_subplot(111)

    elbow = KElbowVisualizer(KMeans(), metric="distortion")
    elbow.fit(pca_df)

    os.makedirs('image', exist_ok=True)
    elbow.fig.savefig(image_path)

    k_best = elbow.elbow_value_

    # Log
    wandb.log(
        {
            "elbow": wandb.Image(image_path),
            "k_best": k_best,
            "score_best": elbow.elbow_score_,
        }
    )
    return k_best


@task(result=LocalResult("models/", location="cluster.pkl"))
def get_cluster_model(
    pca_df: pd.DataFrame, k: int
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    model = KMeans(n_clusters=k)

    # Fit model
    model.fit(pca_df)
    return model 

@task 
def get_clusters(pca_df, model: KMeans):
    return model.predict(pca_df)

@task
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

    # Log plot
    wandb.log({"clusters": wandb.Image(image_path)})

@task(result=FINAL_OUTPUT)
def insert_clusters_to_df(
    df: pd.DataFrame, clusters: np.ndarray
) -> pd.DataFrame:
    return df.assign(clusters=clusters)  

@task 
def wandb_log(config: DictConfig):

    # log models
    models = config.models

    for name, path in models.items():
        wandb.log_artifact(path, name=name, type='model')
    
    # log data
    wandb.log_artifact(config.raw_data.path, name='raw_data', type='data')
    wandb.log_artifact(config.intermediate.path, name='intermediate_data', type='data')

    # log number of columns
    wandb.log({"num_cols": len(config.process.keep_columns)})

@hydra.main(config_path="../config", config_name="main")
def segment(config: DictConfig) -> None:

    with Flow("segmentation") as flow:
        data = pd.read_csv(config.intermediate.path)
        pca = get_pca_model(data)
        pca_df = reduce_dimension(data, pca)

        projections = get_3d_projection(pca_df)

        k_best = get_best_k_cluster(pca_df, image_path=config.image.kmeans)
        model = get_cluster_model(pca_df, k_best)
        preds = get_clusters(pca_df, model)
 
        plot_clusters(
            pca_df, preds, projections, image_path=config.image.clusters
        )
        data = insert_clusters_to_df(data, preds)

        wandb_log(config)

    flow.run()


if __name__ == "__main__":
    segment()
