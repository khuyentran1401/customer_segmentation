import pickle

import bentoml
import bentoml.sklearn
import numpy as np
import pandas as pd
from bentoml.io import NumpyNdarray, PandasDataFrame

# Load processors and model
scaler = bentoml.sklearn.load_runner(
    "scaler:latest", function_name="transform"
)
pca = bentoml.sklearn.load_runner("pca:latest", function_name="transform")
classifier = bentoml.sklearn.load_runner("customer_segmentation_kmeans:latest", function_name="predict")

# Create service with the model
service = bentoml.Service("customer_segmentation_kmeans", runners=[scaler, pca, classifier])

# Create an API function
@service.api(input=PandasDataFrame(), output=NumpyNdarray())
def predict(df: pd.DataFrame) -> np.ndarray:

    # Process data
    scaled_df = pd.DataFrame([scaler.run(df)], columns=df.columns)
    processed = pd.DataFrame(
        [pca.run(scaled_df)], columns=["col1", "col2", "col3"]
    )

    # Predict
    result = classifier.run(processed)
    return np.array(result)
