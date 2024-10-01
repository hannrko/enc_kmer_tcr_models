import numpy as np
import sklearn.metrics
import copy

class MLClasEval:
    def __init__(self, clas_type="binary", usr_perf_func=None, usr_perf_kwargs=None, usr_model_func=None, usr_model_kwargs=None):
        self.eval_type = None
        self.clas_type = clas_type
        self.usr_perf_flag, self.usr_perf_func, self.usr_perf_kwargs = self._usr_func_setup(usr_perf_func, usr_perf_kwargs)
        self.usr_model_flag, self.usr_model_func, self.usr_model_kwargs = self._usr_func_setup(usr_model_func,
                                                                                            usr_model_kwargs)

    def _usr_func_setup(self, usr_func, usr_kwargs):
        if usr_func is None:
            usr_flag = False
            usr_func = None
            usr_kwargs = {}
        else:
            usr_flag = True
            usr_func = usr_func
            if usr_kwargs is None:
                usr_kwargs = {}
        return usr_flag, usr_func, usr_kwargs

    def _perf(self, res, probs, labs, names):
        tn, fp, fn, tp = sklearn.metrics.confusion_matrix(labs, res).ravel()
        def_cm_perf = self._def_cm_perf(tn, fp, fn, tp)
        def_prob_perf = {"AUC": sklearn.metrics.roc_auc_score(labs, probs)}
        perf = {**def_cm_perf, **def_prob_perf}
        if self.usr_perf_flag:
            usr_perf = self.usr_perf_func(res, probs, labs, names, **self.usr_perf_kwargs)
            perf.update(usr_perf)
        return perf

    def _acc(self, tn, fp, fn, tp):
        return (tp + tn) / (tn + fp + tp + fn)
    def _sens(self, tn, fp, fn, tp):
        return tp / (tp + fn)

    def _spec(self, tn, fp, fn, tp):
        return tn / (tn + fp)

    def _mcc(self, tn, fp, fn, tp):
        return ((tp * tn) - (fp * fn)) / np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))

    def _f1(self, tn, fp, fn, tp):
        return 2 * tp / (2 * tp + fp + fn)

    def _def_cm_perf(self, tn, fp, fn, tp):
        perf = {}
        perf["Accuracy"] = self._acc(tn, fp, fn, tp)
        perf["Sensitivity"] = self._sens(tn, fp, fn, tp)
        perf["Specificity"] = self._spec(tn, fp, fn, tp)
        perf["MCC"] = self._mcc(tn, fp, fn, tp)
        perf["scMCC"] = (perf["MCC"] + 1) / 2
        return perf

    def get_perf(self, res, probs, labs, names):
        if self.eval_type == "cv":
            perf = {}
            # get perf for all iterations
            for i in res.keys():
                perf_i = self._perf(res[i], probs[i], labs[i], names[i])
                for key, val in perf_i.items():
                    if key in perf.keys():
                        perf[key] = np.append(perf[key], val)
                    else:
                        perf[key] = val
        else:
            perf = self._perf(res, probs, labs, names)
        return perf

    def _train_test(self, model, trn_data, trn_labs, tst_data):
        model.train(trn_data, trn_labs)
        # if model func, evaluate it
        if self.usr_model_flag:
            usr_func_eval = self.usr_model_func(model, **self.usr_model_kwargs)
        else:
            usr_func_eval = None
        res, probs = model.test(tst_data)
        return res, probs, usr_func_eval

    def train(self, model, trn_data, trn_labs):
        # does this need to use train_test before we can integrate meta func with it?
        self.eval_type = "train"
        res, probs, umf = self._train_test(model, trn_data, trn_labs, trn_data)
        # get performance
        trn_perf = self.get_perf(res, probs, trn_labs, trn_data.index)
        return trn_perf, umf

    def train_test(self, model, trn_data, trn_labs, tst_data, tst_labs):
        self.eval_type = "traintest"
        res, probs, umf = self._train_test(model, trn_data, trn_labs, tst_data)
        # get performance
        tst_perf = self.get_perf(res, probs, tst_labs, tst_data.index)
        return tst_perf, umf

    def cross_validation(self, orig_model, data, labs, cv_obj):
        self.eval_type = "cv"
        # initialise dicts to store results
        it_res = {}
        it_labs = {}
        it_probs = {}
        it_names = {}
        it_func_res = {}
        # make sure labels are numpy arrays
        labs = np.array(labs)
        for i, (trn_i, tst_i) in enumerate(cv_obj.split(data.values, labs)):
            print("CV: "+ str(i+1))
            # copy model
            model = copy.deepcopy(orig_model)
            # training
            trn_data = data.iloc[trn_i]
            trn_labs = labs[trn_i]
            # testing
            tst_data = data.iloc[tst_i]
            tst_labs = labs[tst_i]
            # each iteration we get result and probabilities
            res, probs, umf = self._train_test(model, trn_data, trn_labs, tst_data)
            it_res[i] = res
            it_probs[i] = probs
            it_labs[i] = tst_labs
            it_names[i] = tst_data.index
            it_func_res[i] = umf
        # calculate performance
        perf = self.get_perf(it_res, it_probs, it_labs, it_names)
        mu_perf = [(k, np.mean(v)) for k, v in perf.items()]
        return dict(mu_perf), perf, it_func_res
