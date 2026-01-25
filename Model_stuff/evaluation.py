"""
Evaluation metrics for survival analysis.
"""

import numpy as np
from lifelines import KaplanMeierFitter
from sklearn.metrics import mean_absolute_error


def check_and_convert(event_times, event_indicators):
    """Convert arrays to proper format."""
    et = np.asarray(event_times).astype(float)
    ei = np.asarray(event_indicators).astype(int)
    return et, ei


class KaplanMeierArea:
    """Kaplan-Meier estimator with area computation."""
    
    def __init__(self, durations, events):
        self.durations, self.events = check_and_convert(durations, events)
        self.kmf = KaplanMeierFitter()
        self.kmf.fit(self.durations, event_observed=self.events)
        self._times = self.kmf.survival_function_.index.values
        self._surv = self.kmf.survival_function_['KM_estimate'].values

    def predict(self, times):
        times = np.asarray(times)
        return np.interp(times, self._times, self._surv, left=1.0, right=self._surv[-1])

    def mean(self):
        if len(self._times) < 2:
            return float(self._times[0]) if len(self._times) == 1 else 0.0
        return float(np.trapz(self._surv, self._times))


def Mae_Po(predicted_times, event_times, event_indicators,
           train_event_times=None, train_event_indicators=None,
           method="Hinge", log_scale=False, error_func=np.abs):
    """
    Evaluates MAE-PO (Mean Absolute Error with Pseudo-Observations).
    """
    event_times, event_indicators = check_and_convert(event_times, event_indicators)
    if train_event_times is not None and train_event_indicators is not None:
        train_event_times, train_event_indicators = check_and_convert(train_event_times, train_event_indicators)
    else:
        print("Train set information missing for MAE-PO calculation")
        return float('inf')

    event_indicators = np.asarray(event_indicators, dtype=np.bool_)
    km_model = KaplanMeierArea(train_event_times, train_event_indicators)
    censor_times = event_times[~event_indicators]

    weights = np.ones(event_times.size)
    weights[~event_indicators] = 1 - km_model.predict(censor_times)

    # Calculate best guess times (surrogate times)
    best_guesses = np.empty(shape=event_times.size)
    test_data_size = event_times.size
    sub_expect_time = km_model.mean()
    train_data_size = train_event_times.size
    total_event_time = np.empty(shape=train_data_size + 1)
    total_event_indicator = np.empty(shape=train_data_size + 1)
    total_event_time[0:-1] = train_event_times
    total_event_indicator[0:-1] = train_event_indicators

    for i in range(test_data_size):
        if event_indicators[i] == 1:
            best_guesses[i] = event_times[i]
        else:
            total_event_time[-1] = event_times[i]
            total_event_indicator[-1] = event_indicators[i]
            total_km_model = KaplanMeierArea(total_event_time, total_event_indicator)
            total_expect_time = total_km_model.mean()
            best_guesses[i] = (train_data_size + 1) * total_expect_time - train_data_size * sub_expect_time

    if log_scale:
        errors = np.log(best_guesses) - np.log(predicted_times)
    else:
        errors = best_guesses - predicted_times

    return np.average(error_func(errors), weights=weights)


def concordance(y_pred, y_test, cens):
    """
    Calculate the concordance index (C-index) for survival analysis.
    
    Args:
        y_pred: Risk scores (higher = shorter survival)
        y_test: Actual event times
        cens: Censoring indicators (1 = event, 0 = censored)
    
    Returns:
        float: Concordance index
    """
    n = 0
    n_concordant = 0

    for i in range(len(y_test)):
        for j in range(i + 1, len(y_test)):
            if (y_test[i] < y_test[j] and cens[i] == 1) or (y_test[j] < y_test[i] and cens[j] == 1):
                n += 1
                if y_test[i] < y_test[j]:
                    if y_pred[i] > y_pred[j]:
                        n_concordant += 1
                else:
                    if y_pred[j] > y_pred[i]:
                        n_concordant += 1

    if n == 0:
        return 0.0
    return n_concordant / n


def brier_score(y_pred, y_test):
    """Calculate Brier score."""
    y_pred = np.array(y_pred)
    y_test = np.array(y_test)
    return np.mean((y_pred - y_test) ** 2) / len(y_pred)


def brier_score_at_t(survival_probs, time_bins, true_times, true_events, t):
    """
    Calculate Brier score at time t using IPCW.
    """
    kmf = KaplanMeierFitter()
    kmf.fit(true_times, 1 - true_events)
    
    idx = np.searchsorted(time_bins, t)
    if idx == 0:
        St = survival_probs[:, 0]
    elif idx == len(time_bins):
        St = survival_probs[:, -1]
    else:
        w = (t - time_bins[idx-1]) / (time_bins[idx] - time_bins[idx-1])
        St = (1-w) * survival_probs[:, idx-1] + w * survival_probs[:, idx]
    
    I1 = (true_times <= t) & (true_events == 1)
    I2 = (true_times > t)
    
    Gt = np.zeros(len(true_times))
    for i, ti in enumerate(true_times):
        pred_time = min(t, ti)
        pred = kmf.predict(pred_time)
        Gt[i] = pred[0] if isinstance(pred, (np.ndarray, list)) else pred
    
    w1 = I1 / Gt
    w2 = I2 / kmf.predict(t)
    
    score1 = w1 * (0 - St)**2
    score2 = w2 * (1 - St)**2
    
    return np.mean(score1 + score2)


def integrated_brier_score(survival_probs, time_bins, true_times, true_events, max_time=None):
    """
    Calculate Integrated Brier Score (IBS).
    """
    if max_time is None:
        max_time = np.max(true_times)
    
    eval_times = np.linspace(0, max_time, 100)
    scores = [brier_score_at_t(survival_probs, time_bins, true_times, true_events, t) 
              for t in eval_times]
    
    return np.trapz(scores, eval_times) / max_time
