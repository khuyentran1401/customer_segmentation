import bentoml
import numpy as np
import pandas as pd
from bentoml.io import JSON, NumpyNdarray
from pydantic import BaseModel

# Load processors and model
scaler = bentoml.sklearn.load_runner(
    "scaler:latest", function_name="transform"
)
pca = bentoml.sklearn.load_runner("pca:latest", function_name="transform")
classifier = bentoml.sklearn.load_runner("customer_segmentation_kmeans:latest")

# Create service with the model
service = bentoml.Service(
    "customer_segmentation_kmeans", runners=[scaler, pca, classifier]
)


class Customer(BaseModel):

    Income: float = 58138
    Recency: int = 58
    NumWebVisitsMonth: int = 7
    Complain: int = 0
    age: int = 64
    total_purchases: int = 25
    enrollment_years: int = 10
    family_size: int = 1


# Create an API function
@service.api(input=JSON(pydantic_model=Customer), output=NumpyNdarray())
def predict(customer: Customer) -> np.ndarray:

    df = pd.DataFrame(customer.dict(), index=[0])

    # Process data
    scaled_df = pd.DataFrame([scaler.run(df)], columns=df.columns)
    processed = pd.DataFrame(
        [pca.run(scaled_df)], columns=["col1", "col2", "col3"]
    )

    # Predict
    result = classifier.run(processed)
    return np.array(result)
