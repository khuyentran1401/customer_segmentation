from sqlalchemy import create_engine
from omegaconf import DictConfig
import pandas as pd
from prefect import task, flow
from sklearn.preprocessing import StandardScaler

from helper import load_config


@task
def load_data(config: DictConfig) -> pd.DataFrame:
    connection = config.connection
    engine = create_engine(
        f"postgresql://{connection.user}:{connection.password}@{connection.host}/{connection.database}",
    )
    query = f'SELECT * FROM "{config.data.raw}"'
    df = pd.read_sql(query, con=engine)

    return df


@task
def drop_na(df: pd.DataFrame) -> pd.DataFrame:
    return df.dropna()


@task
def get_age(df: pd.DataFrame) -> pd.DataFrame:
    return df.assign(age=df["Year_Birth"].apply(lambda row: 2021 - row))


@task
def get_total_children(df: pd.DataFrame) -> pd.DataFrame:
    return df.assign(total_children=df["Kidhome"] + df["Teenhome"])


@task
def get_total_purchases(df: pd.DataFrame) -> pd.DataFrame:
    purchases_columns = df.filter(like="Purchases", axis=1).columns
    return df.assign(total_purchases=df[purchases_columns].sum(axis=1))


@task
def get_enrollment_years(df: pd.DataFrame) -> pd.DataFrame:
    df["Dt_Customer"] = pd.to_datetime(df["Dt_Customer"], format="%d-%m-%Y")
    return df.assign(enrollment_years=2022 - df["Dt_Customer"].dt.year)


@task
def get_family_size(df: pd.DataFrame, size_map: dict) -> pd.DataFrame:
    return df.assign(
        family_size=df["Marital_Status"].map(size_map) + df["total_children"]
    )


@task
def drop_features(df: pd.DataFrame, keep_columns: list):
    df = df[keep_columns]
    return df


@task
def drop_outliers(df: pd.DataFrame, column_threshold: dict):
    for col, threshold in column_threshold.items():
        df = df[df[col] < threshold]
    return df.reset_index(drop=True)


@task
def get_scaler(df: pd.DataFrame):
    scaler = StandardScaler()
    scaler.fit(df)

    return scaler


@task
def scale_features(df: pd.DataFrame, scaler: StandardScaler):
    return pd.DataFrame(scaler.transform(df), columns=df.columns)


@task
def save_processed_data(df: pd.DataFrame, config: DictConfig):
    connection = config.connection
    engine = create_engine(
        f"postgresql://{connection.user}:{connection.password}@{connection.host}/{connection.database}",
    )

    df.to_sql(
        name=config.data.intermediate, con=engine, if_exists="replace", index=False
    )


@flow
def process_data():
    config = load_config()
    df = load_data(config)
    df = (
        df.pipe(drop_na)
        .pipe(get_age)
        .pipe(get_total_children)
        .pipe(get_total_purchases)
        .pipe(get_enrollment_years)
        .pipe(get_family_size, config.process.encode.family_size)
        .pipe(drop_features, keep_columns=config.process.keep_columns)
        .pipe(drop_outliers, column_threshold=config.process.remove_outliers_threshold)
    )
    scaler = get_scaler(df)
    df = scale_features(df, scaler)
    save_processed_data(df, config)


process_data()
