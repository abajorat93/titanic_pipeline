import os

BASE_DIR = os.path.realpath(os.path.join(os.path.dirname(__file__), "..", ".."))
DATASETS_DIR = BASE_DIR + "/datasets/"
URL = "https://www.openml.org/data/get_csv/16826755/phpMYEkMl"

SEED_SPLIT = 404
SEED_MODEL = 404

PRODUCTION_MODEL = BASE_DIR + "/models/RandomForest_20220505205301.sav"
DATASET_FILE = DATASETS_DIR + "data.csv"

MODEL_NAME = "RandomForest"

TARGET = "survived"
FEATURES = [
    "pclass",
    "sex",
    "age",
    "sibsp",
    "parch",
    "fare",
    "cabin",
    "embarked",
    "title",
]
NUMERICAL_VARS = ["pclass", "age", "sibsp", "parch", "fare"]
CATEGORICAL_VARS = ["sex", "cabin", "embarked", "title"]
