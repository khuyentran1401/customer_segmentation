from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering, KMeans, SpectralClustering
from sklearn.decomposition import PCA
from yellowbrick.cluster import KElbowVisualizer
from prefect import Flow, task
from prefect.engine.results import LocalResult
from prefect.engine.serializers import PandasSerializer

OUTPUT_DIR = "data/final/"
OUTPUT_FILE = "segmented.csv"


@task
def reduce_dimension(
    df: pd.DataFrame, n_components: int, columns: list
) -> pd.DataFrame:
    pca = PCA(n_components=n_components)
    return pd.DataFrame(pca.fit_transform(df), columns=columns)


@task
def get_3d_projection(pca_df: pd.DataFrame) -> dict:
    """A 3D Projection Of Data In The Reduced Dimensionality Space"""
    return {"x": pca_df["col1"], "y": pca_df["col2"], "z": pca_df["col3"]}


@task
def create_3d_plot(projection: dict, image_path: str) -> None:

    # To plot
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(
        projection["x"],
        projection["y"],
        projection["z"],
        cmap="Accent",
        marker="o",
    )
    ax.set_title("A 3D Projection Of Data In The Reduced Dimension")
    plt.savefig(image_path)


@task
def get_best_k_cluster(
    pca_df: pd.DataFrame, cluster_config: dict, image_path: str
) -> pd.DataFrame:

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111)

    model = eval(cluster_config["algorithm"])()
    elbow = KElbowVisualizer(model, metric=cluster_config["metric"])

    elbow.fit(pca_df)
    elbow.fig.savefig(image_path)

    k_best = elbow.elbow_value_

    return k_best


@task
def get_clusters(
    pca_df: pd.DataFrame, algorithm: str, k: int
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    model = eval(algorithm)(n_clusters=k)

    # Fit model and predict clusters
    return model.fit_predict(pca_df)


@task(
    result=LocalResult(
        OUTPUT_DIR,
        location=OUTPUT_FILE,
        serializer=PandasSerializer("csv"),
    ),
)
def insert_clusters_to_df(
    df: pd.DataFrame, clusters: np.ndarray
) -> pd.DataFrame:
    df = df.assign(clusters=clusters)
    df.to_csv(OUTPUT_DIR + OUTPUT_FILE)

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


def segment() -> None:

    with Flow(
        "segmentation",
    ) as flow:
        data = (
            LocalResult(
                dir="data/intermediate",
                serializer=PandasSerializer(
                    "csv",
                    deserialize_kwargs={"index_col": 0},
                ),
            )
            .read(location="processed.csv")
            .value
        )

        pca_df = reduce_dimension(
            data, n_components=3, columns=["col1", "col2", "col3"]
        )

        projections = get_3d_projection(pca_df)

        create_3d_plot(projections, image_path="image/3d_projection.png")

        k_best = get_best_k_cluster(
            pca_df,
            cluster_config={
                "k": 10,
                "metric": "distortion",
                "algorithm": "KMeans",
            },
            image_path="image/elbow.png",
        )

        preds = get_clusters(pca_df, algorithm="KMeans", k=k_best)

        data = insert_clusters_to_df(data, clusters=preds)

        plot_clusters(
            pca_df,
            preds,
            projections,
            image_path="image/cluster.png",
        )

    # flow.run()
    flow.register(project_name="Customer segmentation")


if __name__ == "__main__":
    segment()
