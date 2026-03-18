import numpy as np
import xgboost as xgb
import optuna
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from bench_model import Standardise

def get_sc_pos(labels):
    n_pos = sum(labels == 1)
    return (len(labels) - n_pos) / n_pos

class XGB:
    def __init__(self, n_estimators=100, max_depth=6, learning_rate=0.3, reg_lambda=1, class_weight=None):
        # some free parameters but keep binary logistic for binary classification model
        self.model_kwargs = {"n_estimators": n_estimators, "max_depth": max_depth, "learning_rate": learning_rate,
                             "reg_lambda": reg_lambda, "objective": "binary:logistic"}
        # need to set balanced class weight later when labels are available
        self.class_weight = class_weight
        # don't initialise model yet due to class weight
        self.model = None

    def train(self, data, labels):
        if self.class_weight == "balanced":
            sc_pos = get_sc_pos(labels)
            self.model_kwargs.update({"scale_pos_weight": sc_pos})
        # init model with params
        self.model = xgb.XGBClassifier(**self.model_kwargs)
        self.model.fit(data, labels)

    def test(self, data):
        return self.model.predict(data), self.model.predict_proba(data)[:, 1]

    def feat_imp(self, imp_type):
        return self.model.get_booster().get_score(importance_type=imp_type)
    
class XGBBayesOpt:
    # must be run without standard scaling within framework
    # apply it here to avoid leakage from splitting
    def __init__(self, class_weight=None, train_val_split_rs=0, n_trials=50, n_estimators=100, test_prop=0.2,
                 max_depth_min=3, max_depth_max=10, learning_rate_min=0.01, learning_rate_max=1, reg_lambda_min=1,
                 reg_lambda_max=100):
        self.class_weight = class_weight
        self.model = None
        self.train_val_split_rs =  train_val_split_rs
        self.full_std = None
        self.n_trials = n_trials
        self.top_params = None
        self.top_auc = None
        self.test_prop = test_prop
        self.default_kwargs = {"objective": "binary:logistic",
                "tree_method": "hist",
                "eval_metric": "logloss",
                "n_estimators": n_estimators}
        # ranges of non-defaults
        self.max_depth_range = [max_depth_min, max_depth_max]
        self.learning_rate_range = [learning_rate_min, learning_rate_max]
        self.reg_lambda_range = [reg_lambda_min, reg_lambda_max]


    def _make_objective(self, train_data, train_labels, val_data, val_labels):

        def objective(trial):
            model_kwargs = {
                "max_depth": trial.suggest_int("max_depth", self.max_depth_range[0], self.max_depth_range[1]),
                "learning_rate": trial.suggest_float("learning_rate", self.learning_rate_range[0], self.learning_rate_range[1], log=True),
                "reg_lambda": trial.suggest_float("reg_lambda", self.reg_lambda_range[0], self.reg_lambda_range[1], log=True),
                }
            if self.class_weight == "balanced":
                model_kwargs.update({"scale_pos_weight": get_sc_pos(train_labels)})
            model_kwargs.update(self.default_kwargs)
            model = xgb.XGBClassifier(**model_kwargs)
            model.fit(train_data, train_labels, verbose=False)
            preds = model.predict_proba(val_data)[:, 1]
            return roc_auc_score(val_labels, preds)
        return objective
    
    def train(self, data, labels, test_prop=0.2):
        # split data
        train_data, val_data, train_labels, val_labels = train_test_split(data, labels, test_size=self.test_prop, stratify=labels, random_state=self.train_val_split_rs)
        opt_std = Standardise(train_data)
        train_data_std = opt_std.apply(train_data)
        val_data_std = opt_std.apply(val_data)
        # do opt
        bayes_obj = self._make_objective(train_data_std, train_labels, val_data_std, val_labels)
        study = optuna.create_study(direction="maximize")
        study.optimize(bayes_obj, n_trials=self.n_trials)
        # get best values
        top_kwargs = study.best_params
        self.top_hyp = dict(top_kwargs)
        self.top_hyp["eval_metric"] = study.best_value
        # train model on top parameters
        self.full_std = Standardise(data)
        data_std = self.full_std.apply(data)
        if self.class_weight == "balanced":
            top_kwargs.update({"scale_pos_weight": get_sc_pos(labels)})
        top_kwargs.update(self.default_kwargs)
        self.model = xgb.XGBClassifier(**top_kwargs)
        self.model.fit(data_std, labels)
    
    def test(self, data):
        data_std = self.full_std.apply(data)
        return self.model.predict(data_std), self.model.predict_proba(data_std)[:, 1]

    def feat_imp(self, imp_type):
        return self.model.get_booster().get_score(importance_type=imp_type)


class L1LR:
    def __init__(self, C=1, max_iter=100, class_weight=None):
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

class L1LRBayesOpt:
    def __init__(self, max_iter=100, class_weight=None, train_val_split_rs=0, n_trials=50):
        self.default_kwargs = {"penalty": "l1",
                               "max_iter": max_iter,
                               "class_weight": class_weight,
                               "solver": "liblinear"}
        self.model = None
        self.train_val_split_rs = train_val_split_rs
        self.full_std = None
        self.n_trials = n_trials
        self.top_hyp = None
        self.fit_feat = None

    def _make_objective(self, train_data, train_labels, val_data, val_labels):
        def objective(trial):
            model_kwargs = {"C": trial.suggest_float("C", 0.01, 1, log=True) }
            model_kwargs.update(self.default_kwargs)
            model = LogisticRegression(**model_kwargs)
            model.fit(train_data, train_labels)
            preds = model.predict_proba(val_data)[:, 1]
            return roc_auc_score(val_labels, preds)
        return objective

    def train(self, data, labels):
        # split data
        train_data, val_data, train_labels, val_labels = train_test_split(data, labels, test_size=0.2, stratify=labels, random_state=self.train_val_split_rs)
        opt_std = Standardise(train_data)
        train_data_std = opt_std.apply(train_data)
        val_data_std = opt_std.apply(val_data)
        # do opt
        bayes_obj = self._make_objective(train_data_std, train_labels, val_data_std, val_labels)
        study = optuna.create_study(direction="maximize")
        study.optimize(bayes_obj, n_trials=self.n_trials)
        # get best values
        top_kwargs = study.best_params
        self.top_hyp = dict(top_kwargs)
        self.top_hyp["eval_metric"] = study.best_value
        # train model on top parameters
        self.full_std = Standardise(data)
        data_std = self.full_std.apply(data)
        top_kwargs.update(self.default_kwargs)
        self.model = LogisticRegression(**top_kwargs)
        self.model.fit(data_std, labels)
        self.fit_feat = data.columns

    def test(self, data):
        data_std = self.full_std.apply(data)
        return self.model.predict(data_std), self.model.predict_proba(data_std)[:, 1]

    def feat_imp(self):
        coef = self.model.coef_.reshape(-1)
        coef_dict = dict(zip(self.fit_feat, np.abs(coef)))
        return coef_dict
