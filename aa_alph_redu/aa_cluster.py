import numpy as np
import pandas as pd
import scipy as sp
import matplotlib.pyplot as plt
import os

class AAClustering:
    def __init__(self, aa_enc=None, aa_matrix=None, aa_matrix_name=None, inp_type=None, cmetric="euclidean", cmethod="average"):
        self.aa = np.array(list("ACDEFGHIKLMNPQRSTVWY"))
        self.inp_types = ["distance", "similarity", "factor", "substitution"]
        self.enc_dict = {"atchley": "factor", "blosum62": "substitution"}
        # if we supply aa_enc in our defined list, infer inp_type
        # if aa_matrix is supplied... infer type? or it must be defined
        if aa_enc.lower() in self.enc_dict.keys():
            self.aa_enc = aa_enc.lower()
            # get input type
            self.inp_type = self.enc_dict[self.aa_enc]
            # load aa_matrix
            fdir = os.path.dirname(os.path.abspath(__file__))
            self.aa_matrix = pd.read_csv(os.path.join(fdir,"encodings",f"{self.aa_enc}.csv"), index_col=0)
        elif aa_matrix is not None:
            # think about what happens if not supplied or if not in possible list
            # could pass function
            self.inp_type = inp_type
            if aa_matrix_name is None:
                self.aa_enc = "unknown"
            else:
                self.aa_enc = aa_matrix_name
            self.enc_dict[self.aa_enc] = self.inp_type
            self.aa_matrix = aa_matrix
        else:
            self.aa_matrix = None
            self.inp_type = None
            self.aa_enc = None
        # get upper triangular matrix of distances
        self.aa_cdist = self._calc_cdist(self.aa_matrix, self.inp_type)
        self.Z = sp.cluster.hierarchy.linkage(self.aa_cdist, metric=cmetric, method=cmethod)

    def _calc_cdist(self, matrix, matrix_type):
        if matrix_type == "distance":
            return self._dist2cdist(matrix)
        elif matrix_type == "similarity":
            return self._sim2cdist(matrix)
        elif matrix_type == "factor":
            return self._factor2cdist(matrix)
        elif matrix_type == "substitution":
            return self._sub2cdist(matrix)
        else:
            return None

    def _dist2cdist(self, dist):
        # assume scaled 0 to 1 with diagonal 0
        return sp.spatial.distance.squareform(dist)

    def _sim2cdist(self, sim):
        # assume scaled 0 to 1 with diagonal 1
        dist = 1 - sim
        # squareform is reversible- get condensed
        return sp.spatial.distance.squareform(dist)

    def _factor2cdist(self, factor):
        factor = factor.loc[self.aa]
        # need to scale
        fmu = factor.mean(axis=0)
        fsd = factor.std(axis=0)
        factor = (factor - fmu)/fsd
        return sp.spatial.distance.pdist(factor)

    def _sub2cdist(self, sub):
        # only take relevant amino acids
        sub = np.array(sub[self.aa].loc[self.aa])
        sub = sub - np.amin(sub)
        diag = np.diag(sub)
        # array of squared diagonals
        sq_denom = np.sqrt(diag.reshape(-1, 1)*diag.reshape(1, -1))
        dist = 1 - sub/sq_denom
        return sp.spatial.distance.squareform(dist)

    def plot_dendro(self, fig_kwargs=None):
        if fig_kwargs is None:
            fig_kwargs={}
        fig, ax = plt.subplots(1, 1, **fig_kwargs)
        den = sp.cluster.hierarchy.dendrogram(self.Z, ax=ax, labels=self.aa)
        plt.show()

    def get_solns(self, mode="all", n_solns=None):
        # add modes, get all or n natural solutions
        # flip so we get one larger cluster first in list
        # this ensures we preserve old behaviour when selecting number of clusters that optimises some value
        # which preferentially selects smaller numbers of clusters if many optima
        # adding zero so we definitely include full alphabet
        soln_threshs = self._get_thresh(mode=mode, n_solns=n_solns)#np.flip(np.unique(np.concatenate(([0], self.Z[:,2]))))
        solns = np.array([sp.cluster.hierarchy.fcluster(self.Z, t=d, criterion="distance") for d in soln_threshs])
        n_clus_soln = np.array([len(np.unique(s)) for s in solns])
        return solns, n_clus_soln

    def _get_thresh(self, mode, n_solns=None):
        if mode == "all":
            return np.flip(np.unique(np.concatenate(([0], self.Z[:,2]))))
        if mode == "restrict":
            return self._restrict_thresh(n_solns)

    def _restrict_thresh(self, n_solns):
        d = self.Z[:, 2]
        d0 = np.concatenate(([0], d))
        # make upper threshold double total dendro so we automatically consider size=1
        d1 = np.concatenate((d, [max(d) * 2]))
        ord_dinds = np.flip(np.argsort(d1 - d0))[:n_solns]
        d_half = (d0 + d1)/2
        return d_half[ord_dinds]
