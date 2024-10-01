import numpy as np
import xgboost as xgb
from sklearn.linear_model import LogisticRegression

class XGB:
    def __init__(self, n_estimators, max_depth, learning_rate, reg_lambda, class_weight=None):
        # some free parameters but keep binary logistic for binary classification model
        self.model_kwargs = {"n_estimators": n_estimators, "max_depth": max_depth, "learning_rate": learning_rate,
                             "reg_lambda": reg_lambda, "objective": "binary:logistic"}
        # need to set balanced class weight later when labels are available
        self.class_weight = class_weight
        # don't initialise model yet due to class weight
        self.model = None

    def train(self, data, labels):
        if self.class_weight == "balanced":
            n_pos = sum(labels == 1)
            sc_pos = (len(labels) - n_pos) / n_pos
            self.model_kwargs.update({"scale_pos_weight": sc_pos})
        # init model with params
        self.model = xgb.XGBClassifier(**self.model_kwargs)
        self.model.fit(data, labels)

    def test(self, data):
        return self.model.predict(data), self.model.predict_proba(data)[:, 1]

    def feat_imp(self, imp_type):
        return self.model.get_booster().get_score(importance_type=imp_type)


class L1LR:
    def __init__(self, C=1, max_iter=100, class_weight=None):
        # saga supported for l1 and should be faster than liblinear
        self.model_kwargs = {"penalty": "l1", "C": C, "solver": "liblinear","max_iter": max_iter, "class_weight": class_weight}
        self.model = LogisticRegression(**self.model_kwargs)
        self.fit_feat = None

    def train(self, data, labels):
        self.model.fit(data, labels)
        self.fit_feat = data.columns

    def test(self, data):
        return self.model.predict(data), self.model.predict_proba(data)[:, 1]

    def feat_imp(self):
        coef = self.model.coef_.reshape(-1)
        coef_dict = dict(zip(self.fit_feat, np.abs(coef)))
        return coef_dict
