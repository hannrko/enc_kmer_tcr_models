import itertools
import os

import pandas as pd
import numpy as np
import aa_alph_redu.aa_cluster as aac
from typing import Type
from .ml_utils import ml_eval as mlev

class AAAlphabet:
    # this is a separate class because we only want to cluster once!
    # reduce amino acid alphabet using only a single size
    def __init__(self, clus_soln: Type[aac.AAClustering], a_size: int, n_solns=None, clus_mode="all") -> None:
        self.size = a_size
        self.solns, self.nc_solns = clus_soln.get_solns(mode=clus_mode, n_solns=n_solns)
        if self.size not in self.nc_solns:
            print(f"No alphabet of size {self.size}")
        is_size = self.nc_solns == self.size
        # we could check that only 1 solution is true
        self.isoln = np.squeeze(self.solns[is_size])
        clus_i = list(range(1, self.size + 1))
        # could add x if size is 1
        if self.size == 1:
            self.new_aas = np.array(["X"])
        else:
            self.new_aas = np.array([(clus_soln.aa[self.isoln == i])[0].lower() for i in clus_i])
        na_all = self.new_aas[self.isoln-1]
        self.alph_converter = dict(zip(clus_soln.aa, na_all))
        self.alph = [list(clus_soln.aa[self.isoln == i]) for i in clus_i]
        self.alph_explainer = dict(zip(self.new_aas, self.alph))

class AAAlphKmer:
    def __init__(self, clus_soln: Type[aac.AAClustering], n_alph, k, n_solns=None, clus_mode="all"):
        self.k = k
        self.alph_catg = self._catg_alph(n_alph)
        self.aa_clus = clus_soln
        if (self.alph_catg == "invalid"):
            print("Invalid n_alph, should be an integer or list-like of length k")
        else:
            self.n_alph = n_alph
        if (self.alph_catg == "single"):
            # make sure same alphabet applied to all positions in kmer
            pos2na = dict(zip(range(self.k), np.repeat(self.n_alph, self.k)))
        else:
            pos2na = dict(zip(range(self.k), self.n_alph))
        # get reduced alphabet(s)
        u_n_alph = np.unique(self.n_alph)
        u_aaa = [AAAlphabet(self.aa_clus, na, n_solns, clus_mode) for na in u_n_alph]
        u_aaa_dict = dict(zip(u_n_alph, u_aaa))
        # list of reduced alphabet objects
        print(f"reduced alphabet {n_alph}")
        self.aa_alph_kmer = [u_aaa_dict[pos2na[p]] for p in range(self.k)]

    def _catg_alph(self, n_alph):
        n_alph = np.squeeze(n_alph)
        if (isinstance(n_alph, int) or ((isinstance(n_alph, np.ndarray) and (n_alph).ndim==0))):
            catg = "single"
        elif ((isinstance(n_alph, list) or (isinstance(n_alph, np.ndarray)
                                            and np.squeeze(n_alph).ndim == 1)) and len(n_alph) == self.k):
            catg = "multiple"
        else:
            catg = "invalid"
        return catg

    def convert(self, seq, delim=None):
        if delim is None:
            delim = ""
        # take a sequence, convert amino acids using reduced alphabet, put back together with delimiter
        # different alphabets for different positions requires converter for each alphabet
        redu_seq = delim.join([self.aa_alph_kmer[i].alph_converter[aa] for i, aa in enumerate(list(seq[:self.k]))])
        if len(seq) > self.k:
            # this treats any additional characters as annotation e.g. positional
            # which is preserved and added back onto kmer
            redu_seq = "-".join([redu_seq, seq[self.k:]])
        return redu_seq

    def converts(self, seqs):
        return list(map(self.convert, seqs))

    def convert_df(self, seq_df):
        # df transposed to have sequences on index
        seq_dft = seq_df.T
        #redu = list(map(self.convert, seq_dft.index))
        redu = self.converts(seq_dft.index)
        seq_dft.index = redu
        redu_df = seq_dft.groupby(seq_dft.index).agg("sum")
        return redu_df.T

class ReducedAAAlphabet:
    def __init__(self, aaclus_kwargs, n_alph, k=3, max_opt_size=20, min_opt_size=1, model=None, d=None, l=None,
                 val_obj=None, n_solns=None, clus_mode="all"):
        self.k = k
        # do clustering
        self.aa_clus = aac.AAClustering(**aaclus_kwargs)
        self.solns, self.nc_solns = self.aa_clus.get_solns(mode=clus_mode, n_solns=n_solns)
        # only save alphabet performance if we optimise
        self.alph_perf = []
        # this is where we can optimise
        zero_n_alph = np.atleast_1d(n_alph) == 0
        if not np.any(zero_n_alph):
            self.final_alph = AAAlphKmer(self.aa_clus, n_alph, self.k, n_solns, clus_mode)
        else:
            # get aa_clus numbers between max and min inclusive
            self.nc_solns = np.array(self.nc_solns)
            single_alph_poss = self.nc_solns[np.logical_and(min_opt_size <= self.nc_solns, self.nc_solns <= max_opt_size)]
            zero_expand = np.array(list(itertools.product(single_alph_poss, repeat=sum(zero_n_alph))))
            self.expand_n_alph = np.zeros((len(zero_expand), len(zero_n_alph)))
            # use combinations to get sets of alphabet sizes that are variable wrt position
            # fill in constant columns
            ze_i = 0
            for i, is_zero in enumerate(zero_n_alph):
                if is_zero:
                    self.expand_n_alph[:, i] = zero_expand[:, ze_i]
                    ze_i = ze_i + 1
                else:
                    self.expand_n_alph[:, i] = np.atleast_1d(n_alph)[i]
            self.expand_n_alph = self.expand_n_alph.astype(int)
            print(self.expand_n_alph)
            # now get alphabet performance for some data
            opt_n_alph = self._optimise_alph(model, d, l, val_obj)
            self.final_alph = AAAlphKmer(self.aa_clus, opt_n_alph, self.k, n_solns, clus_mode)

    def convert(self, seq, delim=None):
        return self.final_alph.convert(seq, delim)

    def converts(self, kmers):
        return self.final_alph.converts(kmers)

    def apply(self, seq_df):
        return self.final_alph.convert_df(seq_df)

    def _optimise_alph(self, model, d, l, val_obj, perf_name="AUC"):
        # given set of alphabet options, calc performance and
        self.alph_perf = [self._calc_alph_perf(n_alph, model, d, l, val_obj)[perf_name] for n_alph in self.expand_n_alph]
        # we might want to save this as table
        # how to handle which performance metric?
        # takes first maximum which is larger alphabet currently...
        return self.expand_n_alph[np.argmax(self.alph_perf)]

    def _calc_alph_perf(self, n_alph, model, d, l, val_obj):
        # convert data to reduced alphabet representation
        aaa = AAAlphKmer(self.aa_clus, n_alph, self.k)
        aa_data = aaa.convert_df(d)
        # do cross-validation
        # get performance
        ra_eval = mlev.MLClasEval()
        cv_res = ra_eval.cross_validation(model, aa_data, l, val_obj)
        a_perf = cv_res[0]
        return a_perf
