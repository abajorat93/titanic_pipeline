# Titanic Pipeline
==============================

## Training
Podemos configurar el training en src-config-config.py
Podemos iniciar el training con `python -m src.training.train_model`
## Serving/Inference
Podemos iniciar el servidor para las predicciones con `uvicorn src.inference.predict:app --reload`.
Podemos rquerir una prediccion mandando un POST al http://127.0.0.1:8000/prediction con un JSON en el Body en el siguiente formato:
```
{
    "pclass": 1,
    "name": "Allen, Miss. Elisabeth Walton",
    "sex": "female",
    "age": "29",
    "sibsp": 0,
    "parch": 0,
    "ticket": "24160",
    "fare": "211.3375",
    "cabin": "B5",
    "embarked": "S",
    "boat": "2",
    "body": "?",
    "home_dest": "St Louis, MO"
}
```
