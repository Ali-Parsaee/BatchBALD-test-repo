"""
Dataset loading and preprocessing.
Combines: datasets.py, Making_and_getting_datasets.py
"""

import numpy as np
import pandas as pd
import torch
import argparse
import sys
import os
from datetime import datetime
from sklearn.utils import shuffle as sklearn_shuffle
from sklearn.utils import check_random_state
from sklearn.datasets import make_friedman1
from sklearn.preprocessing import StandardScaler, LabelEncoder


# =============================================================================
# DATA GENERATION FUNCTIONS
# =============================================================================

def make_data(dataset: str):
    """Load or generate dataset."""
    if dataset == "Synthetic-I":
        return make_synthetic_data()
    elif dataset == "Synthetic-II":
        return generate_synthetic_data(type='poly')
    elif dataset == "SUPPORT":
        return make_support_data(), None
    elif dataset == "NACD":
        return make_nacd_data(), None
    elif dataset == "MIMIC":
        return make_mimic_data(), None
    else:
        raise ValueError(f"Dataset '{dataset}' not recognized.")


def make_synthetic_data(n_samples=10000, n_noise_features=47, base_hazard=0.1, percent_censor=0.3):
    """Generates synthetic survival dataset with linear hazard."""
    x = np.random.standard_normal((n_samples, n_noise_features + 3))
    hazards = x[:, 0] + 2 * x[:, 1] - 0.5 * x[:, 2]
    event_time = np.random.exponential(1 / (base_hazard * np.exp(hazards)))
    censor_time = np.quantile(event_time, 1 - percent_censor)
    time = np.minimum(event_time, censor_time)
    event = (event_time < censor_time).astype(np.int)

    df = pd.DataFrame({
        "time": time, "event": event, "true_time": event_time,
        **{f"x{i+1}": x[:, i] for i in range(x.shape[1])}
    })
    return df, np.array([1, 2, -0.5])


def generate_synthetic_data(censor_dist='Uniform', n_samples=10000, n_features=10, type='linear'):
    """Generate synthetic regression data."""
    if type == "linear":
        X, true_times, coef = make_regression(n_samples=n_samples, n_features=n_features)
    elif type == "poly":
        X, true_times = make_friedman1(n_samples, n_features=n_features, noise=0.05)
        coef = None
    else:
        raise NotImplementedError

    true_times = true_times.round(decimals=1)
    X = X.round(decimals=1)
    if true_times.min() < 0:
        true_times += -true_times.min() + 0.1
    times = np.copy(true_times)

    if censor_dist == "Uniform":
        event_status = np.ones(n_samples)
        censor_time = np.random.uniform(low=true_times.min(), high=true_times.max(), size=n_samples).round(decimals=1)
        event_status[censor_time < true_times] = 0
        times[event_status == 0] = censor_time[event_status == 0]
        df = pd.DataFrame({
            "time": times, "event": event_status, "true_time": true_times,
            **{f"x{i+1}": X[:, i] for i in range(X.shape[1])}
        })
        return df, coef
    raise NotImplementedError


def make_regression(n_samples=10000, n_features=100, n_informative=20, bias=0.0, noise=0.05, shuffle=True, random_state=None):
    """Generate random regression problem."""
    generator = check_random_state(random_state)
    X = generator.randn(n_samples, n_features)
    ground_truth = np.zeros((n_features, 1))
    ground_truth[:n_informative, :] = 2.5 * generator.rand(n_informative, 1)
    y = np.dot(X, ground_truth) + bias
    y += generator.normal(loc=0.0, scale=noise, size=y.shape)

    if shuffle:
        X, y = sklearn_shuffle(X, y, random_state=generator)
        indices = np.arange(n_features)
        generator.shuffle(indices)
        X[:, :] = X[:, indices]
        ground_truth = ground_truth[indices]

    return X, np.squeeze(y), np.squeeze(ground_truth)


# =============================================================================
# REAL DATASET LOADERS
# =============================================================================

def make_support_data():
    """Downloads and preprocesses the SUPPORT dataset."""
    url = "https://biostat.app.vumc.org/wiki/pub/Main/DataSets/support2csv.zip"
    cols_to_drop = ["hospdead", "slos", "charges", "totcst", "totmcst", "avtisst", "sfdm2",
                    "adlp", "adls", "dzgroup", "sps", "aps", "surv2m", "surv6m",
                    "prg2m", "prg6m", "dnr", "dnrday", "hday"]

    data = (pd.read_csv(url).drop(cols_to_drop, axis=1)
            .rename(columns={"d.time": "time", "death": "event"}))
    data["event"] = data["event"].astype(int)
    data["ca"] = (data["ca"] == "metastatic").astype(int)

    fill_vals = {
        "alb": 3.5, "pafi": 333.3, "bili": 1.01, "crea": 1.01, "bun": 6.51,
        "wblc": 9, "urine": 2502, "edu": data["edu"].mean(), "ph": data["ph"].mean(),
        "glucose": data["glucose"].mean(), "scoma": data["scoma"].mean(),
        "meanbp": data["meanbp"].mean(), "hrt": data["hrt"].mean(),
        "resp": data["resp"].mean(), "temp": data["temp"].mean(),
        "sod": data["sod"].mean(), "income": data["income"].mode()[0],
        "race": data["race"].mode()[0]
    }
    data = data.fillna(fill_vals)
    data.sex.replace({'male': 1, 'female': 0}, inplace=True)
    data.income.replace({'under $11k': 0, '$11-$25k': 1, '$25-$50k': 2, '>$50k': 3}, inplace=True)
    
    skip_cols = ['event', 'sex', 'time', 'dzclass', 'race', 'diabetes', 'dementia', 'ca']
    cols_standardize = list(set(data.columns.to_list()).symmetric_difference(skip_cols))
    data[cols_standardize] = data[cols_standardize].apply(lambda x: (x - x.mean()) / x.std())

    onehot_cols = ["dzclass", "race"]
    data = pd.get_dummies(data, columns=onehot_cols, drop_first=True)
    data = data.rename(columns={"dzclass_COPD/CHF/Cirrhosis": "dzclass_COPD"})
    return data


def make_nacd_data():
    """Load and preprocess NACD dataset."""
    # Try multiple possible paths
    possible_paths = [
        "data/NACD/NACD_Full.csv",
        "BNN-ISD-main/data/NACD/NACD_Full.csv",
        "../data/NACD/NACD_Full.csv",
    ]
    
    data = None
    for path in possible_paths:
        if os.path.exists(path):
            data = pd.read_csv(path)
            break
    
    if data is None:
        raise FileNotFoundError(f"NACD_Full.csv not found in any of: {possible_paths}")
    
    cols_to_drop = ['PERFORMANCE_STATUS', 'STAGE_NUMERICAL', 'AGE65']
    data = data.drop([c for c in cols_to_drop if c in data.columns], axis=1)
    
    # CENSORED=1 means censored (no death) -> event=0
    # CENSORED=0 means died -> event=1
    if "CENSORED" in data.columns:
        data["event"] = 1 - data["CENSORED"]
        data = data.drop(columns=["CENSORED"])
    if "SURVIVAL" in data.columns:
        data = data.rename(columns={"SURVIVAL": "time"})

    cols_standardize = ['BOX1_SCORE', 'BOX2_SCORE', 'BOX3_SCORE', 'BMI', 'WEIGHT_CHANGEPOINT',
                        'AGE', 'GRANULOCYTES', 'LDH_SERUM', 'LYMPHOCYTES',
                        'PLATELET', 'WBC_COUNT', 'CALCIUM_SERUM', 'HGB', 'CREATININE_SERUM', 'ALBUMIN']
    cols_standardize = [c for c in cols_standardize if c in data.columns]
    data[cols_standardize] = data[cols_standardize].apply(lambda x: (x - x.mean()) / x.std())
    return data


def make_mimic_data():
    """Load and preprocess MIMIC dataset."""
    possible_paths = [
        "data/MIMIC/MIMIC_IV_v2.0_preprocessed.csv",
        "BNN-ISD-main/data/MIMIC/MIMIC_IV_v2.0_preprocessed.csv",
        "../data/MIMIC/MIMIC_IV_v2.0_preprocessed.csv",
    ]
    
    data = None
    for path in possible_paths:
        if os.path.exists(path):
            data = pd.read_csv(path)
            break
    
    if data is None:
        raise FileNotFoundError(f"MIMIC_IV_v2.0_preprocessed.csv not found in any of: {possible_paths}")
    
    skip_cols = ['event', 'is_male', 'time', 'is_white', 'renal', 'cns', 'coagulation', 'cardiovascular']
    cols_standardize = list(set(data.columns.to_list()).symmetric_difference(skip_cols))
    data[cols_standardize] = data[cols_standardize].apply(lambda x: (x - x.mean()) / x.std())
    return data


def make_SUPPORT(args):
    """Alternative SUPPORT loader using ucimlrepo."""
    try:
        from ucimlrepo import fetch_ucirepo
    except ImportError:
        print("ucimlrepo not installed, falling back to URL-based loader")
        return make_support_data()
    
    support2 = fetch_ucirepo(id=880)
    d = pd.DataFrame(support2.data['original'])
    d.rename(columns={'death': 'event', 'age': 'time'}, inplace=True)
    data = d

    cols_to_drop = ["hospdead", "slos", "charges", "totcst", "totmcst", "avtisst", "sfdm2",
                    "adlp", "adls", "dzgroup", "sps", "aps", "surv2m", "surv6m",
                    "prg2m", "prg6m", "dnr", "dnrday", "hday"]
    data = data.drop(columns=[c for c in cols_to_drop if c in data.columns])
    data["ca"] = (data["ca"] == "metastatic").astype(int)

    fill_vals = {
        "alb": 3.5, "pafi": 333.3, "bili": 1.01, "crea": 1.01, "bun": 6.51,
        "wblc": 9, "urine": 2502, "edu": data["edu"].mean(), "ph": data["ph"].mean(),
        "glucose": data["glucose"].mean(), "scoma": data["scoma"].mean(),
        "meanbp": data["meanbp"].mean(), "hrt": data["hrt"].mean(),
        "resp": data["resp"].mean(), "temp": data["temp"].mean(),
        "sod": data["sod"].mean(), "income": data["income"].mode()[0],
        "race": data["race"].mode()[0]
    }
    data = data.fillna(fill_vals)
    data['sex'] = data['sex'].map({'male': 1, 'female': 0})
    data['income'] = data['income'].map({'under $11k': 0, '$11-$25k': 1, '$25-$50k': 2, '>$50k': 3})

    skip_cols = ['event', 'time', 'sex', 'dzclass', 'race', 'diabetes', 'dementia', 'ca', 'income']
    cols_standardize = [col for col in data.columns if col not in skip_cols]
    
    scaler = StandardScaler()
    data[cols_standardize] = scaler.fit_transform(data[cols_standardize])
    data['event'] = data['event'].astype(int)
    
    onehot_cols = ["dzclass", "race"]
    data = pd.get_dummies(data, columns=[c for c in onehot_cols if c in data.columns], drop_first=True)
    if "dzclass_COPD/CHF/Cirrhosis" in data.columns:
        data = data.rename(columns={"dzclass_COPD/CHF/Cirrhosis": "dzclass_COPD"})
    
    data = data.astype(float)
    if len(data.columns) > 14:
        data = data.drop(data.columns[14], axis=1)
    return data


# =============================================================================
# DATASET GETTER
# =============================================================================

def Get_Dataset(dataset):
    """
    Returns prepared dataset and args for the specified dataset.
    """
    from utils import generate_parser, save_params
    
    model_type = 'BayesianMTLR'
    
    # Simulate command-line arguments
    sys.argv = ['run_models.py', '--dataset', dataset, '--model', model_type,
                '--lr', '0.00008', '--batch_method', 'batchbald_true']
    
    args = generate_parser()
    
    my_r = 20
    args.seed = my_r
    np.random.seed(my_r)
    torch.manual_seed(my_r)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    if dataset == 'SUPPORT':
        data = make_SUPPORT(args)
    else:
        data, coef = make_data(dataset)

    if dataset == 'NACD':
        if 'SURVIVAL' in data.columns:
            data = data.rename(columns={'SURVIVAL': 'time'})
        if 'CENSORED' in data.columns:
            data = data.rename(columns={'CENSORED': 'event'})

    assert "time" in data.columns and "event" in data.columns, \
        "time and event columns required"
    
    args.n_features = data.shape[1]
    args.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    args.device = "cuda" if torch.cuda.is_available() else "cpu"
    args.num_epochs = 100

    save_params(args)
    data = data.astype(float)
    print(f"Dataset shape: {data.shape}")
    
    return data, args
