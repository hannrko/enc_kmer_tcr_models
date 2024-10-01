import aa_alph_redu.aa_alph_reducer as aaar
import kmer_cluster as ckf
import filters
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold


class KmerTCRrepClassification:
    def __init__(self, model, model_kwargs, tf_steps, tf_kwargs, k, explain):
        self.k = k
        self.model = model(**model_kwargs)
        self.explain = explain
        self.tf_op = AAKmerFeatures(tf_steps, tf_kwargs, self.model, self.k, self.explain)

    def train(self, trn_data, trn_labels):
        tf_trn_data, tf_trn_expl = self.tf_op.fit_transform(trn_data, trn_labels)
        print(tf_trn_data, tf_trn_data.min().min(), tf_trn_data.max().max())
        self.model.train(tf_trn_data, trn_labels)

    def test(self, tst_data):
        tf_tst_data = self.tf_op.transform(tst_data)
        return self.model.test(tf_tst_data)

    def feat_imp(self, **feat_kwargs):
        return self.model.feat_imp(**feat_kwargs)


class AAKmerFeatures:
    def __init__(self, steps, kwargs, model, k, explain):
        self.tf_dict = {"ra": self.ra, "cf": self.cf, "filt": self.filt, "pgen": self.pgen_norm,
                        "stnd": self.stnd, "stnd_f": self.flt_stnd, "repair": self.repair}
        self.steps = steps
        self.kwargs = kwargs
        self.step_objs = {}
        self.k = k
        self.model = model
        # if explaining the model, leave out unimportant transformations
        if explain:
            self.explain = dict(zip(self.tf_dict.keys(), [True]*len(self.tf_dict)))
            self.explain.update({"stnd_f": False, "stnd": False, "repair": False})
        else:
            self.explain = dict(zip(self.tf_dict.keys(), [False]*len(self.tf_dict)))
        self.expl_dict = {}

    def tf_step(self, stp, kwargs, data, labels):
        stp_obj, tf_data, tf_expl = self.tf_dict[stp](data, labels, **kwargs)
        self.step_objs[stp] = stp_obj
        return tf_data, tf_expl

    def fit_transform(self, data, labels):
        # remove features with no entries
        data = data[data.columns[data.sum() > 0]]
        expl = {}
        for stp in self.steps:
            # do preprocessing
            data, stp_expl = self.tf_step(stp, self.kwargs[stp], data, labels)
            if self.explain[stp]:
                expl[stp] = stp_expl
        self.expl_dict = expl
        return data, expl

    def transform(self, data):
        for stp in self.steps:
            data = self.step_objs[stp].apply(data)
        return data

    def ra(self, data, labels, n_alph, aa_enc, min_ra_size, n_solns, clus_mode):
        skf = StratifiedKFold(n_splits=5, shuffle=True)
        ra_kwargs = {"aa_enc": aa_enc, "cmetric": "euclidean", "cmethod": "average"}
        ra = aaar.ReducedAAAlphabet(ra_kwargs, n_alph, self.k, model=self.model, d=data, l=labels, val_obj=skf,
                               min_opt_size=min_ra_size, n_solns=n_solns, clus_mode=clus_mode)
        ra_expl = dict(zip(data.columns, ra.converts(data.columns)))
        return ra, ra.apply(data), ra_expl

    def cf(self, data, labels, aa_enc, n_clus, clus_mthd):
        skf = StratifiedKFold(n_splits=5, shuffle=True)
        km_clus = ckf.KmerCluster(data, aa_enc, n_clus, clus_mthd, self.k, self.model, labels, val_obj=skf)
        km_clus_expl = dict(zip(data.columns, km_clus.converts(data.columns)))
        return km_clus, km_clus.apply(data), km_clus_expl

    def filt(self, data, labels, filt_func, seq_func, num, n_seq):
        filter = filters.SeqNumFilter(data, keep_high=True, filt_func=filt_func, l_seq=self.k, seq_func=seq_func,
                                      num_arg=num, n_seq=n_seq)
        filt_expl = dict(zip(data.columns, np.column_stack((filter.trn_meas, [kmer not in filter.rem_seqs for kmer in data.columns]))))
        return filter, filter.apply(data), filt_expl

    def stnd(self, data, labels):
        stnd_sc = Standardise(data)
        return stnd_sc, stnd_sc.apply(data), {}

    def flt_stnd(self, data, labels):
        fstnd_sc = FlatStandardise(data)
        return fstnd_sc, fstnd_sc.apply(data), {}

    def repair(self, data, labels):
        rep = FeatureRepair(data)
        return rep, rep.apply(data), {}



class Standardise:
    def __init__(self, fit_data):
        mu = fit_data.mean(axis=0)
        sd = fit_data.std(axis=0)
        # if sd is zero, we should leave data as is
        # by setting sd = 1, mu = 0
        to_change = sd == 0
        mu[to_change] = 0
        sd[to_change] = 1
        self.mu = mu
        self.sd = sd

    def apply(self, any_data):
        return (any_data - self.mu) / self.sd

class FlatStandardise:
    def __init__(self, fit_data):
        mu = fit_data.mean(axis=None)
        sd = np.std(fit_data.values)
        # if sd is zero, we should leave data as is
        # by setting sd = 1, mu = 0
        self.mu = mu
        self.sd = sd

    def apply(self, any_data):
        return (any_data - self.mu) / self.sd

class FeatureRepair:
    def __init__(self, data):
        self.fit_features = data.columns

    def apply(self, data):
        in_both = [f in data.columns for f in self.fit_features]
        fitfeat_in_data = data[self.fit_features[in_both]]
        # add any missing features
        missing_feat = self.fit_features[np.logical_not(in_both)]
        feat_to_add = pd.DataFrame(index=data.index, columns=missing_feat, data=np.zeros((len(data), len(missing_feat))))
        rep_data = pd.concat([fitfeat_in_data, feat_to_add], axis=1)
        # same order of features
        return rep_data[self.fit_features]
