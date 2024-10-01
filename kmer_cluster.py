import ast
import itertools
import os
import pandas as pd
import numpy as np
import aa_alph_redu.aa_cluster as aac
import scipy as sp
from sklearn.manifold import MDS
from aa_alph_redu.ml_utils import ml_eval as mlev
import ast

class KmerClusterSingle:
    def __init__(self, kmer_df, aa_enc, n_clus, clus_mthd, k):
        self.aa = np.array(list("ACDEFGHIKLMNPQRSTVWY"))
        self.inp_types = ["factor", "substitution"]
        self.enc_dict = {"atchley": "factor", "blosum62": "substitution"}
        self.k = k
        self.aa_enc = aa_enc.lower()
        self.inp_type = self.enc_dict[self.aa_enc]
        kmers = list(kmer_df.columns[kmer_df.sum().values > 0])
        # load aa_matrix
        fdir = os.path.dirname(os.path.abspath(__file__))
        self.aa_matrix = pd.read_csv(os.path.join(fdir, "aa_alph_redu", "encodings", f"{self.aa_enc}.csv"), index_col=0)
        if self.inp_type == "substitution":
            sub = np.array(self.aa_matrix[self.aa].loc[self.aa])
            sub = sub - np.amin(sub)
            diag = np.diag(sub)
            # array of squared diagonals
            sq_denom = np.sqrt(diag.reshape(-1, 1) * diag.reshape(1, -1))
            dist = 1 - sub / sq_denom
            # 5 components to match Atchley
            embedding = MDS(n_components=5, n_init=100, max_iter=1000, eps=0.00001, dissimilarity="precomputed")
            self.aa_factors = pd.DataFrame(data=embedding.fit_transform(dist), index=self.aa)
        elif self.inp_type == "factor":
            self.aa_factors = self.aa_matrix
        # standardise
        fmu = self.aa_factors.mean(axis=0)
        fsd = self.aa_factors.std(axis=0)
        self.aa_factors = (self.aa_factors - fmu)/fsd
        self.enc_kmers = np.array(list(map(self.AA_to_factors, kmers)))
        self.clus_mthd = clus_mthd
        self.n_clus = n_clus
        self.clus_obj = self._cluster_kmers(self.n_clus, self.clus_mthd)
        self.kmer_clus_converter = dict(zip(kmers, self.clus_obj.labels_))
        ulabs = np.unique(self.clus_obj.labels_)
        kmers_by_cluster = [np.array(kmers)[self.clus_obj.labels_ == l] for l in ulabs]
        self.kmer_clus_explainer = dict(zip(ulabs, kmers_by_cluster))

    def AA_to_factors(self, seq):
        # convert amino acid sequences to factors based on conversion table
        # returns numpy array (length of seq)*(number of factors)
        return np.array([np.array(self.aa_factors.loc[aa]) for aa in seq]).flatten()

    def _cluster_kmers(self, nc, cm, rs=0):
        if cm == "kmeans":
            from sklearn.cluster import KMeans
            clus_obj = KMeans(n_clusters=nc, random_state=rs)
            clus_obj.fit(self.enc_kmers)
        return clus_obj

    def convert(self, kmer):
        # if in converter, convert
        if kmer in self.kmer_clus_converter.keys():
            return self.kmer_clus_converter[kmer]
        else:
            enc_kmer = self.AA_to_factors(kmer)
            return self.clus_obj.predict([enc_kmer])[0]

    def converts(self, kmers):
        return list(map(self.convert, kmers))

    def apply(self, kmer_df):
        # kmer_df should have columns as kmers
        kmer_dft = kmer_df.T
        #kmer_clus = list(map(self.convert, kmer_dft.index))
        kmer_dft.index = self.converts(kmer_dft.index)#kmer_clus
        clus_df = kmer_dft.groupby(kmer_dft.index).agg("sum")
        return clus_df.T


class KmerCluster:
    def __init__(self, kmer_df, aa_enc, n_clus, clus_mthd, k, model=None, l=None, val_obj=None):
        self.nclus_perf = None
        #self.kmers = kmers
        self.aa_enc = aa_enc
        self.clus_mthd = clus_mthd
        self.k = k
        if isinstance(n_clus,str):
            n_clus = ast.literal_eval(n_clus)
        self.init_n_clus = n_clus
        if np.array(n_clus).ndim == 0:
            self.km_clus = KmerClusterSingle(kmer_df, aa_enc, n_clus, clus_mthd, k)
        else:
            opt_nclus = self._opt_nclus(n_clus, model, kmer_df, l, val_obj)
            self.km_clus = KmerClusterSingle(kmer_df, aa_enc, opt_nclus, clus_mthd, k)
        self.kmer_clus_converter = self.km_clus.kmer_clus_converter
        self.kmer_clus_explainer = self.km_clus.kmer_clus_explainer

    def _opt_nclus(self, n_clus, model, d, l, val_obj, perf_name="AUC"):
        self.nclus_perf = [self._calc_nclus_perf(nc, model, d, l, val_obj)[perf_name] for nc in n_clus]
        return n_clus[np.argmax(self.nclus_perf)]

    def _calc_nclus_perf(self, n_clus, model, d, labs, val_obj):
        # need to do it this way because of leakage
        try_nclus = BasicKmerClusterModel(model, self.aa_enc, n_clus, self.clus_mthd, self.k)
        nc_eval = mlev.MLClasEval()
        cv_res = nc_eval.cross_validation(try_nclus, d, labs, val_obj)
        a_perf = cv_res[0]
        return a_perf

    def convert(self, kmer):
        return self.km_clus.convert(kmer)

    def converts(self, kmers):
        return self.km_clus.converts(kmers)

    def apply(self, kmer_df):
        return self.km_clus.apply(kmer_df)

class BasicKmerClusterModel:
    def __init__(self, model, aa_enc, n_clus, clus_mthd, k):
        # model should have sklearn format
        self.model = model
        self.n_clus = n_clus
        self.aa_enc = aa_enc
        self.clus_mthd = clus_mthd
        self.k = k
        self.kmer_clus = None

    def train(self, data, labels):
        self.kmer_clus = KmerClusterSingle(data, self.aa_enc, self.n_clus, self.clus_mthd, self.k)
        nc_data = self.kmer_clus.apply(data)
        self.model.train(nc_data, labels)

    def test(self, data):
        nc_data = self.kmer_clus.apply(data)
        return self.model.test(nc_data)#, self.model.predict_proba(nc_data)[:, 1]
