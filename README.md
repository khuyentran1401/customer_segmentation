# End-to-end Customer Segmentation Project

## Dataset
The data is downloaded from [Kaggle](https://www.kaggle.com/imakash3011/customer-personality-analysis).

## Project Structure
* `src`: consists of Python scripts
* `config`: consists of configuration files
* `data`: consists of data
* `processors`: consists of all scikit-learn's transformers used to process the new input
## Set Up the Project
1. Install [Poetry](https://python-poetry.org/docs/#installation)
2. Set up the environment:
```bash
make setup
source activate_env.sh
```
## Run the Project
To run all flows, type:
```bash
python src/main.py
```

## Serve Machine Learning Models with BentoML

To serve the trained model, run:
```bash
bentoml serve src/bentoml_app.py:service --reload
```
Now you should be able to interact with the API by going to http://127.0.0.1:5000 and clicking the "Try it out" button:
![](image/api.gif?raw=true)

To send requests to the newly started service in Python, run:
```bash
python src/predict.py
```

Details of `predict.py`:
```python
import requests

prediction = requests.post(
    "http://127.0.0.1:5000/predict",
    headers={"content-type": "application/json"},
    data='{"Income": 58138, "Recency": 58, "Complain": 0,"age": 64,"total_purchases": 25,"enrollment_years": 10,"family_size": 1}',
).text

print(prediction)
```
Output:
```bash
1
```

## Run a Streamlit app
To open a Streamlit app, run:
```bash
streamlit run src/streamlit_app.py
```
then go to http://localhost:8501. You should see a web app like below:

![](image/streamlit.gif?raw=true)

