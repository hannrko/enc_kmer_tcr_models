import ml_utils.ml_eval as mlev
from sklearn.linear_model import LogisticRegression
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

class TrainTestSklearnModel:
    def __init__(self, model):
        self.model = model
    def train(self, X, y):
        self.model.fit(X, y)

    def test(self, X, y):
        return self.model.predict(X), self.model.predict_proba(X)[:, 1]

def null_func(model, n, p):
    print(f"N is {n}, p is {p}")
skm = LogisticRegression()
model = TrainTestSklearnModel(skm)
d = pd.DataFrame(np.random.rand(100,10))
l = np.concatenate((np.zeros(50), np.ones(50)))
skf = StratifiedKFold(n_splits=5)
test_eval = mlev.MLClasEval(usr_model_func=null_func, usr_model_kwargs={"n":9, "p":"hiya"})
print(test_eval.cross_validation(model, d, l, skf))

