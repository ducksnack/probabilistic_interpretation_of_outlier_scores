import numpy as np
import pandas as pd
from scipy.stats import binom
import math

def ExCeeD(train_scores, test_scores, prediction, contamination):
    
    """
    Estimate the example-wise confidence according to the model ExCeeD provided in the paper.
    First, this method estimates the outlier probability through a Bayesian approach.
    Second, it computes the example-wise confidence by simulating to draw n other examples from the population.

    Parameters
    ----------
    train_scores   : list of shape (n_train,) containing the anomaly scores of the training set (by selected model).
    test_scores    : list of shape (n_test,) containing the anomaly scores of the test set (by selected model).
    prediction     : list of shape (n_test,) assuming 1 if the example has been classified as anomaly, 0 as normal.
    contamination  : float regarding the expected proportion of anomalies in the training set. It is the contamination factor.

    Returns
    ----------
    exWise_conf    : np.array of shape (n_test,) with the example-wise confidence for all the examples in the test set.
    
    """
    
    n = len(train_scores)
    n_anom = np.int(n*contamination) #expected anomalies
    
    count_instances = np.vectorize(lambda x: np.count_nonzero(train_scores <= x)) 
    n_instances = count_instances(test_scores)

    prob_func = np.vectorize(lambda x: (1+x)/(2+n)) 
    posterior_prob = prob_func(n_instances) #Outlier probability according to ExCeeD
    
    conf_func = np.vectorize(lambda p: 1 - binom.cdf(n - n_anom, n, p))
    exWise_conf = conf_func(posterior_prob)
    np.place(exWise_conf, prediction == 0, 1 - exWise_conf[prediction == 0]) # if the example is classified as normal,
                                                                             # use 1 - confidence.
    
    return exWise_conf

def cantelli_inequality(a, mean, var):
    k = (a-mean)/math.sqrt(var)
    if k > 0:
        p = min(1, (1/(1+k**2)))
    else:
        p = 1
    return 1-p

def ExCeeD_alt(train_scores, test_scores, prediction, contamination, alg='bayes'):
    
    """
    THIS IS AN ALTERED VERSION OF THE ExCeeD FUNCTION.
    THE ONLY DIFFERENCES ARE: 
    1) IT RETURNS AN (OUTLIER_PROBABILITY, EXAMPLEWISE_CONFIDENCE)-TUPLE
    RATHER THAN JUST THE EXAMPLEWISE_CONFIDENCE
    2) IT TAKES AN ALG ARGUMENT NOW. THE ALG ARGUMENT DEFINES HOW THE FUNCTION
    DETERMINES OUTLIER PROBABILITY. THE OPTIONS ARE:
        I 'bayes' FOR THE BAYESIAN APPROACH DESCRIBED IN THE ORIGINAL PAPER
        II 'cant' FOR OUR CANTELLI VARIATION OF ESTIMATING OUTLIER PROBABLITIES
    
    Estimate the example-wise confidence according to the model ExCeeD provided in the paper.
    First, this method estimates the outlier probability through a Bayesian approach.
    Second, it computes the example-wise confidence by simulating to draw n other examples from the population.

    Parameters
    ----------
    train_scores   : list of shape (n_train,) containing the anomaly scores of the training set (by selected model).
    test_scores    : list of shape (n_test,) containing the anomaly scores of the test set (by selected model).
    prediction     : list of shape (n_test,) assuming 1 if the example has been classified as anomaly, 0 as normal.
    contamination  : float regarding the expected proportion of anomalies in the training set. It is the contamination factor.

    Returns
    ----------
    exWise_conf    : np.array of shape (n_test,) with the example-wise confidence for all the examples in the test set.
    
    """
    n = len(train_scores)
    n_anom = np.int(n*contamination) #expected anomalies
    
    count_instances = np.vectorize(lambda x: np.count_nonzero(train_scores <= x)) 
    n_instances = count_instances(test_scores)
    
    if alg == 'bayes':
        prob_func = np.vectorize(lambda x: (1+x)/(2+n)) 
        posterior_prob = prob_func(n_instances) #Outlier probability according to ExCeeD
    elif alg == 'cant':
        score_mean, score_var = test_scores.mean(), test_scores.var()
        posterior_prob = [cantelli_inequality(s, score_mean, score_var) for s in test_scores]
        
    conf_func = np.vectorize(lambda p: 1 - binom.cdf(n - n_anom, n, p))
    exWise_conf = conf_func(posterior_prob)
    np.place(exWise_conf, prediction == 0, 1 - exWise_conf[prediction == 0]) # if the example is classified as normal,
                                                                             # use 1 - confidence.
    
    return posterior_prob, exWise_conf
    