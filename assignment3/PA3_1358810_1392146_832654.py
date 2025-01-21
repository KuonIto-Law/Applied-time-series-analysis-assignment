# I, (1358810) worked with the students with number 1392146 and 832654


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import acf as sm_acf 
from statsmodels.stats import diagnostic
from scipy.stats import chi2 
from scipy import stats
import statsmodels.api as sm
import statsmodels.tsa as tsm
import statsmodels.stats as sms
from scipy.optimize import minimize



### Task 1

def modified_portmanteau_test(residuals, p, q, lags, significance_level):
    """
    Computes the Modified Portmanteau test statistic for a given residual series.

    Parameters:
    - residuals: T * 1 vector of residuals
    - p: lag order of AR part of the fitted ARMA(p, q) process
    - q: lag order of MA part of the fitted ARMA(p, q) process
    - lags: number of residual autocorrelations to be tested
    - significance_level: significance level for the test

    Returns:
    - Q: Portmanteau test statistic
    - critical_value: Critical value at the given significance level
    """
    T = len(residuals)
    acf_vals = sm_acf(residuals, nlags=lags, fft=False)[1:]  # [1:] to exclude lag 0
    acf_sq = acf_vals**2
    result_dataframe = []

    for lag in range(5, lags+1):
        Q = T * (T+2) * np.sum(acf_sq[:lag] / np.arange(T-1, T-lag-1, -1))
        
        # Degrees of freedom
        dof = lag - p - q
    
        # Critical value from chi-squared distribution
        critical_value = chi2.ppf(1 - significance_level, dof)
    
        # Testing
        reject = Q > critical_value
        
        # Saving the result as a dictionary to make a dataframe
        result_dataframe.append({
            "lag": lag,
            "ACF": acf_vals[lag-1]**2, 
            "QLB": Q, 
            "critical_value": critical_value, 
            "reject": reject
        })

    return pd.DataFrame(result_dataframe)



### Task 2

# simulate a time serires with length T = 1000 from an AR(2) process with c = 0.1, α1 = 0.6, α2 = 0.2 and εt iid N(0, σ^2), σ = 1            
def ar2(alpha1, alpha2, T, c, sigma):            
    ar2_data = np.zeros(T)
    wn = np.random.normal(0, sigma, size=T)
    for ii in range(2,T):
        ar2_data[ii] = c + alpha1*ar2_data[ii-1] + alpha2*ar2_data[ii-2] + wn[ii] 
    return ar2_data


# generating the time series data                                                  
ar2_data = ar2(0.6, 0.2, 1000, 0.1, 1)


# This function is from tutorial 
def est_arp(data, p, c):
    T = len(data) - p
    if (c == 1):
       X = np.zeros(shape=(T, p+1))
       X[:,0] = np.array(np.ones(T))
       for ii in range(1, p+1):
           X[:,ii] = data[p-ii:-ii]
    else:
        X = np.zeros(shape=(T, p))
        for ii in range(1, p+1):
            X[:,ii-1] = data[p-ii:-ii]
    res_arp = sm.OLS(data[p:], X).fit()
    alpha_est = res_arp.params
    y_hat = res_arp.fittedvalues
    res = res_arp.resid
    sig_est = res_arp.cov_params()
    return alpha_est, sig_est, res, y_hat, res_arp

alpha_est1, sig_est1, res1, y_hat1, res_arp1 = est_arp(ar2_data, 1, 1)
alpha_est2, sig_est2, res2, y_hat2, res_arp2 = est_arp(ar2_data, 2, 1)

result_1 = modified_portmanteau_test(res1, 1, 0, 16, 0.05)
result_2 = modified_portmanteau_test(res2, 2, 0, 16, 0.05)

print(result_1, result_2)

# compare the function I created with the module 
print(diagnostic.acorr_ljungbox(res1, lags=16, model_df=1))    
print(diagnostic.acorr_ljungbox(res2, lags=16, model_df=2)) 

"""
Interpretation of the result 
For AR(1) model, the null hypothesis that there is no autocorrelation between residuals  was rejected. 
Since the simulated data follow a true AR(2) process, the AR(1) model does not fully explain the autocorrelation. 
Therefore, significant autocorrelation remains in the residuals and is likely to be rejected by the portmanteau test.  

For AR(2) model, the nullhypothesis is not rejected. 
The autocorrelation the simulated AR(2) process can be properly fitted so that no significant autocorrelations remain in the residuals. 
As a result, it is likely not rejected.
"""



### Task 3

def log_likelihood(params, y):
    """
    Calculate the negative log-likelihood for an AR(1) model with intercept.

    Parameters:
    params : array-like
        A list or array containing the parameters of the AR(1) model.
        - c: The intercept term.
        - alpha: The AR(1) coefficient.
        - sigma2: The variance of the white noise error term.
    y : array-like
        A 1-dimensional array of time series data.

    Returns: float
      The negative log-likelihood of the AR(1) model given the parameters and data.
    """
    c, alpha, sigma2 = params
    T = len(y)
    
    sum_term = np.sum((y[1:] - c - alpha * y[:-1]) ** 2) / (2 * sigma2)
    
    log_like = (
        -0.5 * np.log(2 * np.pi)
        - 0.5 * np.log(sigma2 / (1 - alpha**2))
        - (y[0] - c / (1 - alpha)) ** 2 / (2 * sigma2 / (1 - alpha**2))
        - ((T - 1) / 2) * np.log(2 * np.pi)
        - ((T - 1) / 2) * np.log(sigma2)
        - sum_term
        )
    return -log_like  


def ar1_mle(y):
    """
    Estimate the parameters of an AR(1) model with intercept using Maximum Likelihood Estimation (MLE).

    Parameters:
    y : array-like
        A 1-dimensional array of time series data.

    Returns:
    dict
        A dictionary containing the results of the MLE estimation:
        - 'Hessian Matrix': The inverse of the Hessian matrix at the optimal point.
        - 'c': The estimated intercept term.
        - 'alpha1': The estimated AR(1) coefficient.
        - 'sigma2': The estimated variance of the white noise error term.
        - 'std_errors': The standard errors of the estimated parameters.
        - 'max_log_likelihood': The maximum log-likelihood value at the optimal point.
    """
    
    # Initial parameter estimates
    c_start = np.mean(y)
    alpha_start = 0.5
    sigma2_start = np.var(y)
    start_params = np.array([c_start, alpha_start, sigma2_start])

    bounds=[(None, None), (-0.99999, 0.99999), (1e-10, None)]

    # Optimize the log-likelihood function
    opt_res = minimize(log_likelihood, start_params, args=(y,), bounds=bounds, method='L-BFGS-B', options={'gtol': 1e-8, 'eps': 1e-8})

    return {
        'Hessian Matrix': opt_res.hess_inv.todense(),
        'c': opt_res.x[0],
        'alpha1': opt_res.x[1],
        'sigma2': opt_res.x[2],
        'std_errors': np.sqrt(np.diag(opt_res.hess_inv.todense())),
        'max_log_likelihood': -log_likelihood(opt_res.x, y)
    }



### Task 4

# loading the dataset
data = pd.read_csv('./awm19up18.csv')

# transforming array and calculate the GDP growth rate
gdp = np.array(data['YER'])
gdp_growth = np.log(gdp[1:]) - np.log(gdp[:-1])

# run the function 
ar1_mle(gdp_growth)

# Comparing the result that was generated by the function I created with that from the ARIMA module
model = ARIMA(gdp_growth, order=(1, 0, 0)).fit()
print(model.summary())

# The result
# The estimator for alpha and sigma squared differ from those in the ARIMA model by less than one standard deviation each, but differ by more than one standard deviation for the constant