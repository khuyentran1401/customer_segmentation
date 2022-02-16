import pickle

import bentoml
import bentoml.sklearn
import numpy as np
import pandas as pd
from bentoml.io import NumpyNdarray, PandasDataFrame

# Load model
classifier = bentoml.sklearn.load_runner("customer_segmentation_kmeans:latest")

# Create service with the model
service = bentoml.Service("customer_segmentation_kmeans", runners=[classifier])

# Create an API function
@service.api(input=PandasDataFrame(), output=NumpyNdarray())
def predict(df: pd.DataFrame) -> np.ndarray:

    # Process data
    scaler = pickle.load(open("processors/scaler.pkl", "rb"))

    scaled_df = pd.DataFrame(scaler.transform(df), columns=df.columns)

    pca = pickle.load(open("processors/PCA.pkl", "rb"))
    processed = pd.DataFrame(
        pca.transform(scaled_df), columns=["col1", "col2", "col3"]
    )

    # Predict
    result = classifier.run(processed)
    return np.array(result)
