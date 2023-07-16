import pandas as pd
import numpy as np
from sklearn.linear_model import SGDRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import pickle as pkl

class split_dataset:
    def __init__(self) -> None:
        pass

    def split(df, tg):
        X_train, X_test, y_train, y_test = train_test_split(df.drop(tg, axis=1), tg, test_size=0.2, random_state=3)
        return X_train, X_test, y_train, y_test


pipe1 = Pipeline([
    ('splitting', split_dataset()),
    ('sgd_regressor', SGDRegressor(penalty='l2', eta0=0.01, learning_rate='constant', alpha=0.1))
])