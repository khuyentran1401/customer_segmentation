import bentoml 
import pandas as pd 
from bentoml.io import NumpyNdarray, PandasDataFrame
import numpy as np 

# Load transformers and model
scaler = bentoml.sklearn.load_runner("scaler:latest", function_name="transform")
pca = bentoml.sklearn.load_runner("pca:latest", function_name="transform")

model = bentoml.sklearn.load_runner("customer_segmentation_kmeans:latest")

service = bentoml.Service("customer_segmentation_kmeans", runners=[scaler, pca, model])

@service.api(input=PandasDataFrame(), output=NumpyNdarray())
def predict(df: pd.DataFrame) -> np.array:

    # Process data
    scaled_df  = pd.DataFrame([scaler.run(df)], columns=df.columns)
    processed_df = pd.DataFrame([pca.run(df)], columns=['col1', 'col2', 'col3'])

    # Predict data
    result = model.run(processed_df)
    return np.array(result)