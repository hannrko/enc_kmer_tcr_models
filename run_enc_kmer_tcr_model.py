from bench_model import KmerTCRrepClassification
import clas_model_wrappers
import pandas as pd
import numpy as np
import os
import aa_alph_redu.ml_utils.ml_eval as mlev
from sklearn.model_selection import RepeatedStratifiedKFold
import sklearn

def nullfunc(model):
    return pd.Series({})

def get_alphperfs(model):
    ap_list = model.tf_op.step_objs["ra"].alph_perf
    a_list = model.tf_op.step_objs["ra"].expand_n_alph
    ap = pd.DataFrame(data=np.column_stack((a_list, ap_list)), columns=["alph", "perf"])
    return ap

def get_clusperfs(model):
    ncp_list = model.tf_op.step_objs["cf"].nclus_perf
    nc_list = model.tf_op.step_objs["cf"].init_n_clus
    return pd.DataFrame(data=np.column_stack((nc_list, ncp_list)), columns=["nclus", "perf"])

def get_ra_xgb_hyps(model):
    a_list = model.tf_op.step_objs["ra"].expand_n_alph
    dict_list = model.tf_op.step_objs["ra"].alph_itfuncres
    a_hyps = pd.DataFrame(dict_list)
    a_hyps.index = np.squeeze(a_list)
    return a_hyps

def feat_hyp_expl(model, feat, imp_kwargs=None):
    if imp_kwargs is None:
        imp_kwargs = {}
    if feat == "cf":
        ff = get_clusperfs
        hf = nullfunc
        expl = pd.Series(model.tf_op.expl_dict[feat])
        t = model.tf_op.step_objs[feat].time_res
    elif feat == "ra":
        ff = get_alphperfs
        hf = get_ra_xgb_hyps
        expl = pd.Series(model.tf_op.expl_dict[feat])
        t = model.tf_op.step_objs[feat].time_res
    else:
        ff = nullfunc
        hf = nullfunc
        expl = pd.Series({})
        t = {}
    if hasattr(model.model, "top_hyp"):
        hyps = model.model.top_hyp
    else:
        hyps = {}
    return expl, ff(model), pd.Series(model.feat_imp(**imp_kwargs)), t, pd.Series(hyps), hf(model)

def trntst(trn_data, trn_labs, tst_data, tst_labs, m_key, m_kw, feat, feat_kw, k, runtime_exp=False, res_dir=None,
                sv=False):
    m_dict = {"xgbbo": clas_model_wrappers.XGBBayesOpt, "l1lrbo": clas_model_wrappers.L1LRBayesOpt,
              "l1lr": clas_model_wrappers.L1LR, "xgb": clas_model_wrappers.XGB}
    esn_feat = ["repair"]
    if feat != "":
        feat_seq = [feat] + esn_feat
    else:
        feat_seq = esn_feat
        feat_kw = {}

    if res_dir is None:
        res_dir = "results"

    if not os.path.exists(res_dir):
        os.makedirs(res_dir)
    iter_name = os.path.basename(os.path.normpath(res_dir))
    model = m_dict[m_key]
    print(m_kw)
    if m_key in ["xgbbo", "l1lrbo"]:
        ra_func = lambda y: y.top_hyp
    else:
        ra_func = None
    m = KmerTCRrepClassification(model, m_kw, feat_seq, {feat: feat_kw, "repair": {}}, k=k, explain=True,
                                 ra_usr_model_func=ra_func)
    if m_key in ["xgb", "xgbbo"]:
        imp_kwargs = {"imp_type": "gain"}
    else:
        imp_kwargs = None

    trntst = mlev.MLClasEval(usr_model_func=feat_hyp_expl, usr_model_kwargs={"feat": feat, "imp_kwargs": imp_kwargs},
                             timing=True)

    tst_perf, tst_meta = trntst.train_test(m, trn_data.T, trn_labs, tst_data.T, tst_labs)

    time = pd.Series(trntst.time_store)
    tst_perf = pd.Series(tst_perf)
    search = tst_meta[0]
    thresh = tst_meta[1]
    imp = tst_meta[2]
    opttime = tst_meta[3]
    hyps = tst_meta[4]
    opthyps = tst_meta[5]

    if sv:
        time.to_csv(os.path.join(res_dir, "TIME_" + iter_name + ".csv"))
        tst_perf.to_csv(os.path.join(res_dir, "PERF_" + iter_name + ".csv"))
        search.to_csv(os.path.join(res_dir, "SEARCH_" + iter_name + ".csv"))
        thresh.to_csv(os.path.join(res_dir, "THRESH_" + iter_name + ".csv"))
        imp.to_csv(os.path.join(res_dir, "IMP_" + iter_name + ".csv"))
        if runtime_exp:
            pd.DataFrame(opttime).to_csv(os.path.join(res_dir, "OPTTIME_" + iter_name + ".csv"))
        hyps.to_csv(os.path.join(res_dir, "HYPS_" + iter_name + ".csv"))
        opthyps.to_csv(os.path.join(res_dir, "OPTHYPS_" + iter_name + ".csv"))

    return tst_perf, time, search, thresh, imp, opttime, hyps, opthyps