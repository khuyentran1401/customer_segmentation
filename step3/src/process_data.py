from email.policy import default
import pandas as pd
from feature_engine.wrappers import SklearnTransformerWrapper
from sklearn.preprocessing import StandardScaler
from prefect import Flow, task, Parameter
from prefect.engine.results import LocalResult
from prefect.engine.serializers import PandasSerializer
from helper import artifact_task

@artifact_task
def load_data(data_name: str, load_kwargs: dict) -> pd.DataFrame:
    df = pd.read_csv(data_name, **load_kwargs)
    return df


@artifact_task
def drop_na(df: pd.DataFrame) -> pd.DataFrame:
    return df.dropna()

@artifact_task
def get_age(df: pd.DataFrame) -> pd.DataFrame:
    return df.assign(age=df["Year_Birth"].apply(lambda row: 2021 - row))

@artifact_task
def get_total_children(df: pd.DataFrame) -> pd.DataFrame:
    return df.assign(total_children=df["Kidhome"] + df["Teenhome"])

@artifact_task
def get_total_purchases(df: pd.DataFrame) -> pd.DataFrame:
    purchases_columns = df.filter(like="Purchases", axis=1).columns
    return df.assign(total_purchases=df[purchases_columns].sum(axis=1))

@artifact_task
def get_enrollment_years(df: pd.DataFrame) -> pd.DataFrame:
    df["Dt_Customer"] = pd.to_datetime(df["Dt_Customer"])
    return df.assign(enrollment_years=2022 - df["Dt_Customer"].dt.year)

@artifact_task
def get_family_size(df: pd.DataFrame, size_map: dict) -> pd.DataFrame:
    return df.assign(
        family_size=df["Marital_Status"].map(size_map) + df["total_children"]
    )

def drop_features(df: pd.DataFrame, keep_columns: list):
    df = df[keep_columns]
    return df

def drop_outliers(df: pd.DataFrame, column_threshold: dict):
    for col, threshold in column_threshold.items():
        df = df[df[col] < threshold]
    return df.reset_index(drop=True)


@artifact_task
def drop_columns_and_rows(
    df: pd.DataFrame, keep_columns: list, remove_outliers_threshold: dict
):
    df = df.pipe(drop_features, keep_columns=keep_columns).pipe(
        drop_outliers, column_threshold=remove_outliers_threshold
    )

    return df


@task
def scale_features(df: pd.DataFrame):
    scaler = SklearnTransformerWrapper(transformer=StandardScaler())
    return scaler.fit_transform(df)


def process_data():

    family_size = Parameter(
        "family_size",
        default={
            "Married": 2,
            "Together": 2,
            "Absurd": 1,
            "Widow": 1,
            "YOLO": 1,
            "Divorced": 1,
            "Single": 1,
            "Alone": 1,
        },
    )

    keep_columns = Parameter(
        "keep_columns",
        default=[
            "Income",
            "Recency",
            "NumWebVisitsMonth",
            "AcceptedCmp3",
            "AcceptedCmp4",
            "AcceptedCmp5",
            "AcceptedCmp1",
            "AcceptedCmp2",
            "Complain",
            "Response",
            "age",
            "total_purchases",
            "enrollment_years",
            "family_size",
        ],
    )

    remove_outliers_threshold = Parameter(
        "remove_outliers_threshold",
        default={
            "age": 90,
            "Income": 600000,
        },
    )

    with Flow(
        "process_data",
        result=LocalResult(
            "data/intermediate",
            location="processed.csv",
            serializer=PandasSerializer("csv"),
        ),
    ) as flow:
        df = load_data(
            "data/raw/marketing_campaign.csv",
            {"sep": "\t"},
        )
        df = drop_na(df)
        df = get_age(df)
        df = get_total_children(df)
        df = get_total_purchases(df)
        df = get_enrollment_years(df)
        df = get_family_size(df, family_size)
        df = drop_columns_and_rows(df, keep_columns, remove_outliers_threshold)
        df = scale_features(df)

    # flow.run()
    flow.register(project_name="Customer segmentation")


if __name__ == "__main__":
    process_data()
