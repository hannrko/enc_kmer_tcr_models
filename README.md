# enc_kmer_tcr_models
Python code for kmer-based TCR repertoire classification accompanying preprint: Kockelbergh, Hannah, Shelley C. Evans, Liam Brierley, et al. ‘Evaluating the Utility of Amino Acid Similarity-Aware Kmers to Represent TCR Repertoires for Classification’. Preprint, bioRxiv, 24 June 2025. https://doi.org/10.1101/2024.12.06.626025. 

Implements reduced alphabet features based on Atchley factor or BLOSUM62 encoding, and subsequent classification using XGBoost or logistic regression. Bayesian optimisation for hyperparameter setting is optional. Also implements kmeans clustering of kmers based on the same set of encodings and whole amino acid alphabet kmer models for comparison purposes.

## Installation

### Preprequisites
- Python3

### Instructions
- Clone or fork and clone repository
- Install dependencies in python virtual environment
    - matplotlib==3.9.2
    - numpy==2.1.3
    - optuna==4.6.0
    - pandas==2.2.3
    - scikit-learn==1.5.2
    - scipy==1.14.1
    - xgboost==2.0.3

## Usage

### Models
run_enc_kmer_tcr_model.py contains function trntst() which given a kmer matrix and corresponding labels each for training and testing, implements a model with optional reduced alphabet features and hyperparameter setting with Bayesian optimisation.
An example of usage is provided in example_enc_kmer_tcr_model.py with randomly-generated kmer matrices.

run_enc_kmer_tcr_model.trntst() options are:
- `trn_data`
    - a kmer matrix with M kmer rows, N sample columns, containing counts of each kmer observed in each TCR repertoire sample
    - includes N samples from training data only
- `trn_labs`
    - an array of length N indicating class of each training sample
    - e.g. 1 indicating presence of a specific disease, 0 for absence of that disease for donor of TCR repertoire sample
- `tst_data`
    - a kmer matrix with Mt kmer rows, Nt sample columns, containing counts of each kmer observed in each TCR repertoire sample
    - includes Nt samples from testing data only
- `tst_labs`
    - an array of length Nt indicating class of each testing sample
    - consistent with training labels
- `m_key`
    - a string indicating which classification model to apply
  
    | m_key    | Description                        |
    | -------- | ---------------------------------- |
    | `xgb`     | XGBoost classification        |
    | `xgbbo`     | XGBoost classification with Bayesian optimisation of hyperparameters     |
    | `l1lr`   | Logistic regression with L1 norm (LASSO penalty)                 |
    | `l1lrbo`   | Logistic regression with L1 norm (LASSO penalty) and Bayesian optimisation of hyperparameters   |

- `m_kw`
    - A dict of model parameters depending on m_key chosen
    - `xgb`
      - `class_weight`: `balanced` to set XGBoost scale_pos_weight or None
      - `n_estimators`: int, from XGBoost default 100
      - `max_depth`: int, from XGBoost default 6
      - `learning_rate`: float, from XGBoost default 0.3
      - `reg_lambda`: float, from XGBoost default 1
    - `xgbbo`
      - `class_weight`: `balanced` to set XGBoost scale_pos_weight or default None
      - `train_val_split_rs`: int, set random seed of train test split within Bayesian optimisation, default 0
      - `n_trials`: int, Bayesian optimisation iterations, default  50
      - `n_estimators`: int, from XGBoost, default  100
      - `test_prop`: float<1, proportion of training data used to evaulate trials in Bayesian optimisation, default 0.2
      - `max_depth_min`: int, minimum max_depth value to consider, default 3
      - `max_depth_max`: int, maxmimum max_depth value to consider, default 10
      - `learning_rate_min`: float, minimum learning_rate value to consider, default 0.01
      - `learning_rate_max`: float, maximum learning_rate value to consider, default 1
      - `reg_lambda_min`: float, minimum reg_lambda value to consider, default 1
      - `reg_lambda_max`: float, maximum reg_lambda value to consider, default 100
    - `l1lr`
      - `C`: float, from sklearn.linear_model.LogisticRegression default 1
      - `max_iter`: int, from sklearn.linear_model.LogisticRegression default 100
      - `class_weight`, from sklearn.linear_model.LogisticRegression, `balanced` or default None
    - `l1lrbo`
      - `C_min`: float, minimum C value to consider, default 0.01
      - `C_max`: float, maximum C value to consider, default 1
      - `max_iter`: int, from sklearn.linear_model.LogisticRegression default 100
      - `class_weight`, from sklearn.linear_model.LogisticRegression, `balanced` or default None
      - `train_val_split_rs`: int, set random seed of train test split within Bayesian optimisation, default 0
      - `n_trials`: int, Bayesian optimisation iterations, default  50
      - `test_prop`: float<1, proportion of training data used to evaulate trials in Bayesian optimisation, default 0.2
     
- `feat`
  - Which type of features to use
    
  | Step     | Description                        |
  | -------- | ---------------------------------- |
  | `ra`     | Reduced amino acid alphabet        |
  | `cf`     | Kmer clustering                   |
  | ``   | Kmers (no further featurisation)                  |

- `feat_kw`
  - dict containing feature type-specific arguments
  - `ra`
    - `n_alph`: int, reduced alphabet size, 0 to optimise acording to AUC
    - `aa_enc`: str, `Atchley` or `BLOSUM62`
    - `min_ra_size`: int, smallest alphabet size to consider if `n_alph=0`
  - `cf`
    - `aa_enc`: str, `Atchley` or `BLOSUM62`
    - `n_clus`: int, number of clusters
    - `clus_mthd`: str, `kmeans` for sklearn.cluster.KMeans 
  - ``
    - can be an empty dict

- `k`
  - length of kmers in data

- `runtime_exp`
  - bool, indicate whether to save time taken for each amino acid similarity threshold if using `feat=ra` or `feat=cf`, default False
    
- `res_dir`
  - str or None, path of directory to save result if `sv=True`
  - if None but `sv=True`, make new directory `results` in current working directory
    
- `sv`
  - bool, whether to save results
  - saves in `res_dir` or  new dir called `results` in current working directory
    
### Data preparation
TCR repertoire kmer matrices can be obtained using code at https://github.com/hannrko/imrep_utils
