import requests

prediction = requests.post(
    "https://bentoml-her0ku-mty0ndg3mza0ngo.herokuapp.com/predict",
    headers={"content-type": "application/json"},
    data='{"Income": 58138, "Recency": 58, "NumWebVisitsMonth": 2, "Complain": 0,"age": 64,"total_purchases": 25,"enrollment_years": 10,"family_size": 1}',
).text

print(prediction)
