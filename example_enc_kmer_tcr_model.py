import run_enc_kmer_tcr_model
from itertools import product
import pandas as pd
import numpy as np

class KmerMatrixGenerator:
    def __init__(self, k, n_sam, split=0.5, vals=None, noise=None, sam_initial="S", rs=0):
        aa = "ACDEFGHIKLMNPQRSTVWY"
        poss_kmers = [''.join(comb) for comb in product(aa, repeat=k)]
        sams = [sam_initial + str(i+1) for i in range(n_sam)]
        # if a base value or array of base values is supplied, use that
        # if not, make an array of zeros
        if vals is None:
            base_signal = np.zeros((n_sam,len(poss_kmers)))
        else:
            if isinstance(vals, int) or isinstance(vals, float):
                base_signal = np.ones((n_sam, len(poss_kmers)))*vals
            else:
                base_signal = vals
        # if we specify a value for noise, get uniform noise with mean 0 and magnitude specified
        if noise is None:
            noise_to_add = np.zeros((n_sam, len(poss_kmers)))
        else:
            np.random.seed(rs)
            noise_to_add = (np.random.rand(n_sam, len(poss_kmers)) - 0.5)*noise
        data = base_signal + noise_to_add
        self.kmer_mat = pd.DataFrame(columns=poss_kmers, index=sams, data=data)
        n_pos = round(n_sam*split)
        n_neg = n_sam - n_pos
        self.labels = np.concatenate((np.repeat(0, n_neg), np.repeat(1, n_pos)))

    def change_counts(self, vals, row=None, column=None, label=None):
        # define columns
        if column is None:
            c = self.kmer_mat.columns
        else:
            c = column
        # define rows
        if row is None:
            r = self.kmer_mat.index
        else:
            r = row
        if label is not None:
            # refine rows
            l = self.kmer_mat.index[self.labels==label]
            r = [rs for rs in r if rs in l]
        self.kmer_mat[c].loc[r] = vals

# 4mers
k = 4

km_train = KmerMatrixGenerator(k, 20, noise=1, rs=0)

km_test = KmerMatrixGenerator(k, 20, noise=1, sam_initial="ST", rs=1)

# model takes kmer matrices with one kmer per row, one sample per column, so transpose
print(km_train.kmer_mat.T, km_train.labels)
print(km_test.kmer_mat.T, km_test.labels)

# Model
m_key = "xgbbo" # choose model XGBoost with Bayesian optimisation (xgbbo), options are "xgb", "xgbbo", "l1lr", "l1lrbo"
m_kw = {"class_weight": "balanced"}

# Features
feat = "ra" # reduced alphabet features
enc = "Atchley" # Atchley factor encoding
n_alph = 0 # set alphabet size using model performance

res = run_enc_kmer_tcr_model.trntst(km_train.kmer_mat.T, km_train.labels, km_test.kmer_mat.T, km_test.labels,
                              m_key, m_kw, feat,
                                    {"aa_enc": enc, "n_alph": n_alph, "min_ra_size": 1},
                                    k=k,  runtime_exp=False, res_dir=None, sv=False)

print(res)