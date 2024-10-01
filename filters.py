# -*- coding: utf-8 -*-
"""
Created on Thu Mar  9 13:20:15 2023

@author: hannrk
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import kneed

class SeqFilter:
    # filter kmers based on some measure derived from their frequency
    def __init__(self, trn_seqs, seq_func="var", filt_func="mu+sd", thresh_arg=0, keep_abv=True, seq_kwargs=None, l_alpha=20, l_seq=3):
        # determine what type of filter to use 
        self.ff_key = filt_func
        self.ff_opts = {"mu+sd": self._muplussd, "pct": self._percent, "pctp": self._percent_of_poss}
        # save threshold argument
        self.thresh_arg = thresh_arg
        # save flag that indicates if we are keeping sequences that have a 
        # quantity above the theshold set by the filter function
        # do we also need to define what function we use to get our quantity of interest?
        self.seqf_key = seq_func
        # define dict to access sequence frequency operations
        self.seqf_opts = {"var": self._var, "normvar": self._normvar,
                          "sqnormvar": self._sqnormvar, "freq": self._freq, 
                          "raritynormmad": self._raritynormmad}
        self.keep_abv = keep_abv
        # sequence function keyword args
        if seq_kwargs is None:
            self.seq_kwargs = {}
        else:
            self.seq_kwargs = seq_kwargs
            # length of first sequence should be length of all sequences
        self.l_seq = l_seq
        if len(trn_seqs.columns[0]) != self.l_seq:
            postns = [ts[self.l_seq:] for ts in trn_seqs.columns]
            self.p_seq = len(np.unique(postns))
        else:
            self.p_seq = 1
        self.l_alpha = l_alpha
        # trn_seqs should be a pandas dataframe
        # with index indicating sample and columns indicating sequences
        # get sequence quantity we're using
        # trn_meas should be a pandas series
        self.trn_meas = self.seqf_opts[self.seqf_key](trn_seqs, **self.seq_kwargs)
        # get threshold
        self.thresh = self.ff_opts[self.ff_key](self.trn_meas,self.thresh_arg)
        # identify all sequences to keep based on keep above key
        if self.keep_abv:
            # keep sequences with measure above the threshold
            self.keep_mask = self.trn_meas > self.thresh
        else:
            # or below it
            self.keep_mask = self.trn_meas < self.thresh
        # list of sequences to REMOVE
        self.rem_seqs = list(self.trn_meas[~self.keep_mask].index)
        
        
    def apply(self, new_seqs):
        # find the set of sequences that are in new seqs thta need to be removed
        ovlp_seqs = [rk for rk in self.rem_seqs if rk in new_seqs.columns]
        filt_seqs = new_seqs.drop(columns=ovlp_seqs)
        return filt_seqs
    
    @staticmethod
    def _var(freq_df):
        return freq_df.var()
    
    @staticmethod
    def _normvar(freq_df):
        return freq_df.var()/freq_df.sum()
    
    @staticmethod
    def _sqnormvar(freq_df):
        return freq_df.var()/(freq_df.sum())**2
    
    @staticmethod
    def _freq(freq_df):
        return freq_df.sum()    
    
    @staticmethod
    def _raritynormmad(freq_df, nseq, rarity_thresh=0.01):
        l = rarity_thresh*freq_df.shape[0]/nseq
        mad = (freq_df - freq_df.median()).abs().sum()
        rarity_denom = 1 + l*freq_df.sum()
        return mad/rarity_denom
            
    @staticmethod
    def _muplussd(q, n_sd):
        mu = q.mean()
        sd = q.std()
        return mu + n_sd*sd
    
    @staticmethod
    def _percent(q, pct):
        return q.quantile(q=pct/100)
    
    def _percent_of_poss(self, q, pct):
        n_poss = (self.l_alpha**self.l_seq)*self.p_seq
        n_filt = n_poss*(pct)/100
        print(f"{n_filt} kmers filtered")
        # sort q
        qs = q.sort_values(ascending=True)
        # return n_filt(th) value
        if len(qs) < n_filt:
            n_filt = len(qs)
            print(f"fewer kmers than specified by filter, returning {n_filt} kmers")
        return 0 if pct == 0 else qs.iloc[int(n_filt)-1]

    def filt_plot(self, sv_path=None):
        # reorder measure so it's in descending order
        self.meas_ord = self.trn_meas.iloc[np.flip(np.argsort(self.trn_meas))]
        plt.figure(figsize=(20,8))
        plt.plot(self.meas_ord.values)
        plt.xlabel("Sequence rank")
        plt.ylabel(self.seqf_key)
        # fill in the space below removed sequences in red
        # use the actual filter to fill between 
        plt.fill_between(range(len(self.meas_ord.values)),0,self.meas_ord.values,
                         where=(~self.keep_mask[self.meas_ord.index]),color="r")
        if sv_path is not None:
            plt.savefig(os.path.join(sv_path,"filter_thresh.png"))
        else:
            plt.show()


class SeqNumFilter:
    # filter kmers based on some measure derived from their frequency
    def __init__(self, trn_seqs, num_arg=None, seq_func="var", filt_func="num", keep_high=True, n_seq=None,
                 rarity_thresh=0.01, l_alpha=20, l_seq=3):
        # determine what type of filter to use
        self.ff_key = filt_func
        self.ff_opts = {"pctp": self._percent_of_poss, "num": self._num, "pct": self._percent}
        self.keep_high = keep_high
        self.seqf_key = seq_func
        # define dict to access sequence frequency operations
        self.seqf_opts = {"var": self._var, "normvar": self._normvar,
                          "sqnormvar": self._sqnormvar, "freq": self._freq,
                          "normmad": self._normmad, "std": self._std,
                          "normstd": self._normstd, "pgenstd": self._pgen_std}
        print(seq_func)
        # sequence function keyword args
        #if seq_kwargs is None:
            #self.seq_kwargs = {}
        #else:
            #self.seq_kwargs = seq_kwargs
            # length of first sequence should be length of all sequences
        self.l_seq = l_seq
        if len(trn_seqs.columns[0]) != self.l_seq:
            postns = [ts[self.l_seq:] for ts in trn_seqs.columns]
            self.p_seq = len(np.unique(postns))
        else:
            self.p_seq = 1
        self.l_alpha = l_alpha
        # trn_meas should be a pandas series
        self.trn_meas = self.seqf_opts[self.seqf_key](trn_seqs, n_seq=n_seq, rarity_thresh=rarity_thresh)
        # save threshold argument
        if isinstance(num_arg,(int, float)):
            self.num_arg = num_arg
            print("num arg not string")
        else:
            print("Setting filter threshold with elbow")
            self.num_arg = self._set_elbow()
        # get sequences to remove
        self.rem_seqs = self.ff_opts[self.ff_key](self.trn_meas, self.num_arg)

    def _set_elbow(self):
        ord_ind = np.flip(np.argsort(self.trn_meas))
        meas_ord = self.trn_meas.iloc[ord_ind]
        x = kneed.KneeLocator(np.arange(len(meas_ord)), meas_ord.values, S=10.0, curve="convex",
                          direction="decreasing").elbow
        if x is None:
            x = len(meas_ord)
            print("WARNING: filter threshold cannot be set, no filtering applied")
        return x

    def apply(self, new_seqs):
        # find the set of sequences that are in new seqs thta need to be removed
        ovlp_seqs = [rk for rk in self.rem_seqs if rk in new_seqs.columns]
        filt_seqs = new_seqs.drop(columns=ovlp_seqs)
        return filt_seqs

    def _percent_of_poss(self, q, pct, keep_pct=False):
        n_poss = (self.l_alpha**self.l_seq)*self.p_seq
        if keep_pct:
            pct_keep = pct
        else:
            pct_keep = 100 - pct
        # number KEPT
        n_filt = round(n_poss*(pct_keep)/100)
        print(f"{n_filt} kmers kept")
        # sort q
        qs = q.sort_values(ascending=(not self.keep_high))
        if len(qs) < n_filt:
            seq_to_remove = []
            print(f"fewer kmers than specified by filter, don't remove any")
        elif n_filt == 0:
            seq_to_remove = qs.index
        else:
            seq_to_remove = list(qs.iloc[n_filt:].index)
        return seq_to_remove

    def _percent(self, q, pct, keep_pct=False):
        if keep_pct:
            pct_keep = pct
        else:
            pct_keep = 100 - pct
        # number KEPT
        n_filt = round(len(q)*(pct_keep)/100)
        print(f"{n_filt} kmers kept")
        # sort q
        qs = q.sort_values(ascending=(not self.keep_high))
        if len(qs) < n_filt:
            seq_to_remove = []
            print(f"fewer kmers than specified by filter, don't remove any")
        elif n_filt == 0:
            seq_to_remove = qs.index
        else:
            seq_to_remove = list(qs.iloc[n_filt:].index)
        return seq_to_remove

    def _num(self, q, n):
        qs = q.sort_values(ascending=(not self.keep_high))
        if len(qs) < n:
            seq_to_remove = []
            print(f"fewer kmers than specified by filter, don't remove any")
        if n == 0:
            seq_to_remove = qs.index
        else:
            seq_to_remove = list(qs.iloc[int(n):].index)
        return seq_to_remove

    @staticmethod
    def _var(freq_df, **kwargs):
        return freq_df.var()

    @staticmethod
    def _normvar(freq_df, **kwargs):
        return freq_df.var() / freq_df.sum()

    @staticmethod
    def _std(freq_df, **kwargs):
        return freq_df.std()

    @staticmethod
    def _normstd(freq_df, **kwargs):
        return freq_df.std() / freq_df.sum()

    @staticmethod
    def _sqnormvar(freq_df, **kwargs):
        return freq_df.var() / (freq_df.sum()) ** 2

    @staticmethod
    def _freq(freq_df, **kwargs):
        return freq_df.sum()

    @staticmethod
    def _normmad(freq_df, **kwargs):
        mad = ((freq_df - freq_df.median()).abs()).median()#.sum()
        return mad / freq_df.sum()

    @staticmethod
    def _pgen_std(freq_df, n_seq, **kwargs):
        odp = "C:/Users/Hannrk/OneDrive - The University of Liverpool"
        dp = os.path.join(odp, "imrep_data_sim")
        olga_dir = os.path.join(dp, "OLGA/raw_olga_preprocessed")
        olga_path = os.path.join(olga_dir, f"raw_olga_raw_kmers_k4_ignore_dup_kmersTrue.csv")
        olga = pd.read_csv(olga_path, index_col=0).T
        #olga = pd.read_csv("olga_data/raw_olga_raw_kmers_k4_ignore_dup_kmersTrue.csv", index_col=0).T
        pgen_kmer = dict(olga.sum() / 10 ** 8)
        kmer_per_seq = freq_df/(n_seq)
        eps = np.finfo(float).eps
        pgen_denom = np.array([pgen_kmer.get(kmer, eps) for kmer in freq_df.columns])
        pgen_ratio = kmer_per_seq/pgen_denom
        return pgen_ratio.std()

    def filt_plot(self, sv_path=None):
        # reorder measure so it's in descending order for aesthetics
        ord_ind = np.flip(np.argsort(self.trn_meas))
        meas_ord = self.trn_meas.iloc[ord_ind]
        plt.figure()
        plt.plot(meas_ord.values)
        plt.xlabel("Sequence rank")
        plt.ylabel(self.seqf_key)
        # fill in the space below removed sequences in red
        # use the actual filter to fill between
        rem_mask = pd.Series(data=[seq in self.rem_seqs for seq in self.trn_meas.index],
                                 index=self.trn_meas.index)
        plt.fill_between(range(len(meas_ord.values)),0,meas_ord.values,
                         where=(rem_mask.loc[meas_ord.index]),color="r")
        if sv_path is not None:
            plt.savefig(os.path.join(sv_path,"filter_thresh.png"))
        else:
            plt.show()

