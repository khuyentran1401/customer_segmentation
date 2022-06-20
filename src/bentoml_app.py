import bentoml 
import pandas as pd 
from bentoml.io import NumpyNdarray, PandasDataFrame, JSON 
import numpy as np 
from pydantic import BaseModel

# Load transformers and model
scaler = bentoml.sklearn.load_runner("scaler:latest", function_name="transform")
pca = bentoml.sklearn.load_runner("pca:latest", function_name="transform")

model = bentoml.sklearn.load_runner("customer_segmentation_kmeans:latest")

service = bentoml.Service("customer_segmentation_kmeans", runners=[scaler, pca, model])

class Customer(BaseModel):
    Income: float = 53138
    Recency: int = 58
    NumWebVisitsMonth: int = 7
    Complain: int = 0
    age: int = 64
    total_purchases: int = 25
    enrollment_years: int = 10
    family_size: int = 1

@service.api(input=JSON(pydantic_model=Customer), output=NumpyNdarray())
def predict(data: Customer) -> np.array:

    df = pd.DataFrame(data.dict(), index=[0])
    
    # Process data
    scaled_df  = pd.DataFrame([scaler.run(df)], columns=df.columns)
    processed_df = pd.DataFrame([pca.run(df)], columns=['col1', 'col2', 'col3'])

    # Predict data
    result = model.run(processed_df)
    return np.array(result)