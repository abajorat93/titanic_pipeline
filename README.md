# Titanic Pipeline
==============================

## Training
* Podemos configurar el training en src-config-config.py
* Podemos iniciar el training con `python -m src.training.train_model`
* Posteriormente, dirigirse a la carpeta `/models`, luego identificar el modelo recién entrenado, copiar el nombre y registrarlo en `/src/config/config.py` linea 10
    **Ejemplo**

    Este modelo: `PRODUCTION_MODEL = BASE_DIR + '/models/RandomForest_Best.sav'`

    Por este: `PRODUCTION_MODEL = BASE_DIR + '/models/RandomForest_20220505205301.sav`

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

## Lista de Logs
| Path          | Description   |
| ------------- | ------------- |
| /src/training/train_model/   | Logging for the model training  |
| Content Cell  | Content Cell  |
* 