# Titanic Pipeline
==============================

## Training
* Podemos configurar el training en src-config-config.py
* Podemos iniciar el training con `python -m src.training.train_model`
* Posteriormente, dirigirse a la carpeta `/models`, luego identificar el modelo recién entrenado, copiar el nombre y registrarlo en `/src/config/config.py`, linea **10**.

    **Ejemplo**

    Este modelo: `PRODUCTION_MODEL = BASE_DIR + '/models/RandomForest_Best.sav'`

    Por este: `PRODUCTION_MODEL = BASE_DIR + '/models/RandomForest_20220505205301.sav`

## Serving/Inference
* Podemos iniciar el servidor para las predicciones con `uvicorn src.inference.predict:app --reload` .
* Podemos rquerir una prediccion mandando un `POST` (se puede usar Postman) a la url http://127.0.0.1:8000/prediction con un JSON en el Body en el siguiente formato:
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

##
Para usar los notebooks, instala primero el paquete de manera local usando `python -m build` y despues `pip install -e .`
## Lista de Logs
| Path          | Description   | Severity   |
| ------------- | ------------- | ------------- |
| `/src/training/train_model.py`  | Log para registrar que el modelo haya sido entrenado. Se registran **predicciones, accuracy y lugar de guardado**. | DEBUG |
| `src/inference/predict.py`  | Log para registrar para el modelo: id, version, datos de entrada, datos procesados y prediccion. Para el sistema: timepo de ejecución y memoria utilizada. | INFO |
| `src/inference/predict.py`  | Carga del modelo | CRITICAL |
* 

## Opcional
* Falta agregar una página inicial con la documentación
