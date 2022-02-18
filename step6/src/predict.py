import pandas as pd
import wandb
from prefect import task, Flow
import pickle
import numpy as np


@task
def get_data():
    return pd.DataFrame(
        {
            "Income": [58138],
            "Recency": [58],
            "NumWebVisitsMonth": [10],
            "Complain": [0],
            "age": [64],
            "total_purchases": [25],
            "enrollment_years": [10],
            "family_size": [1],
        }
    )


@task
def initialize():
    return wandb.init(project="customer_segmentation")


@task
def scale(run, version: str, df: pd.DataFrame) -> pd.DataFrame:
    scaler_dir = run.use_artifact(
        f"khuyentran1401/customer_segmentation/scaler:{version}", type="model"
    ).download()
    scaler = pickle.load(open(f"{scaler_dir}/scaler.pkl", "rb"))
    return pd.DataFrame(scaler.transform(df), columns=df.columns)


@task
def pca(run, version: str, df: pd.DataFrame) -> pd.DataFrame:
    pca_dir = run.use_artifact(
        f"khuyentran1401/customer_segmentation/pca:{version}", type="model"
    ).download()
    pca = pickle.load(open(f"{pca_dir}/pca.pkl", "rb"))
    return pd.DataFrame(pca.transform(df), columns=["col1", "col2", "col3"])


@task
def predict(run, version: str, df: pd.DataFrame) -> np.ndarray:
    model_dir = run.use_artifact(
        f"khuyentran1401/customer_segmentation/cluster:{version}", type="model"
    ).download()
    model = pickle.load(open(f"{model_dir}/cluster.pkl", "rb"))
    return model.predict(df)


with Flow("predict") as flow:
    run = initialize()
    df = get_data()
    df = scale(run, "v5", df)
    df = pca(run, "v6", df)
    pred = predict(run, "v7", df)

flow.run()
