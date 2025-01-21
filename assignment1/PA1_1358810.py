# importing libraries
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns


# Task1
# loading the file
df = pd.read_csv("C:/Users/Nutzer/Desktop/Applied time series analysis/programming_assignment/CLVMNACSCAB1GQDE.csv")
print(df)

# figure size
plt.figure(figsize=(15,5))

# time series plot
plt.plot(df["DATE"], df["CLVMNACSCAB1GQDE"])

# adjusting the x axis
T = len(df)
plt.xticks(range(0,T,28),df["DATE"][list(range(0,T,28))])


# Task2
def transformation(data, transformation):
    
    """
    Transforming the time series data into user-specified time serires
    
    Args:
        data: the time series data
        transformation: log, first differences of logs, first differences, seasonal differences, seasonal differences of logs
    
    Returns:
        specified transformed data
    """
    
    if transformation == 'log':
        return np.log(data)
    
    elif transformation == 'log_diff':    
        log_data = np.log(data)
        return [log_data[i] - log_data[i-1] for i in range(1, len(data))]
    
    elif transformation == "diff":
        return [data[i] - data[i-1] for i in range(1, len(data))]
    
    elif transformation == 'seasonal_diff':
        return [data[i] - data[i-4] for i in range(4, len(data))]
    
    elif transformation == 'log_seasonal_diff':
        log_data = np.log(data)
        return [log_data[i] - log_data[i-4] for i in range(4, len(data))]

# plots for the seasonal growth rate of real GDP
seasonal_growth_rate = transformation(df["CLVMNACSCAB1GQDE"], "log_diff")
plt.figure()
plt.title("Seasonal growth rate of real GDP")
plt.plot(seasonal_growth_rate)

# plots for the yearly growth rate of real GDP
yearly_growth_rate = transformation(df["CLVMNACSCAB1GQDE"], "log_seasonal_diff")
plt.figure()
plt.title("Yearly growth rate of real GDP")
plt.plot(yearly_growth_rate)


# Task3

def autocorrelation(data, lag):
    """
    calculating the autocorrelation
    
    data: time series data list
    lag: user-spesified lag
    
    returning the autocorrelation
    """
    mean = np.mean(data)
    numerator = [(data[i] - mean) * (data[i-lag] - mean) for i in range(lag, len(data))]
    denominator = [(data[i] - mean) ** 2 for i in range(len(data))] 
    autocorrelation = sum(numerator) / sum(denominator)
    return autocorrelation

# a plot of the sample autocorrelations for the quarterly growth rate of German real GDP. 
auto_seasonal = [autocorrelation(seasonal_growth_rate, i) for i in range(0,50)]
plt.figure()
plt.title("autocorrelation for seaonal growth rate")
plt.stem(auto_seasonal)

# a plot of the sample autocorrelations for the yearly growth rate of German real GDP.
auto_yearly = [autocorrelation(yearly_growth_rate, i) for i in range(0,50)]
plt.figure()
plt.title("autocorrelation for yearly growth rate")
plt.stem(auto_yearly)

#Compare the plots and explain 

#My answer 
#The autocorrelation at seasonal growth rate showa less volatiliy and shorter-time dependencies compared to yearly data.
#On the other hand, the autocorrelation for yearly growth rate 
#exhibits cyclical patterns possibly linked to broader economic trends or cycles.


# Task4

# HP filter function
def hp_filter(y):
    """
    This function divides the time series data into trend and cyclical component.
    
    y: time series data as a numpy array
    
    return: trend
    """
    T = len(y)

    # Generate G matrix
    def G_matrix(T):
        matrix = np.zeros((T, T))
        
        if T > 0:
            matrix[0, :3] = [1, -2, 1]
        if T > 1:
            matrix[1, :4] = [-2, 5, -4, 1]
        for i in range(2, T-2):
            matrix[i, i-2:i+3] = [1, -4, 6, -4, 1]
        if T > 2:
            matrix[T-2, -4:] = [1, -4, 5, -2]
        if T > 1:
            matrix[T-1, -3:] = [1, -2, 1]
        
        return matrix
    
    # Convert y to a column vector
    y_matrix = y.values.reshape(-1, 1)
    
    # Calculate trend component
    identity_matrix = np.eye(T)
    lam = 1600
    trend = np.dot(np.linalg.inv(identity_matrix + lam * G_matrix(T)), y_matrix)
    
    return trend

# plot time serires and trend

# figure size
plt.figure(figsize=(15,5))

# title
plt.title("The log of real German GDP and the trend component")

# time series plot
plt.plot(df["DATE"], np.log(df["CLVMNACSCAB1GQDE"]), label="The log of real German GDP")
plt.plot(hp_filter(np.log(df["CLVMNACSCAB1GQDE"])), label="Hodrick-Prescott Trend Estimate")

# legend
plt.legend()

# adjusting the x axis
T = len(df)
plt.xticks(range(0,T,28),df["DATE"][list(range(0,T,28))])

plt.show()

# plot cyclical component
cyclical = np.log(df["CLVMNACSCAB1GQDE"]).values.reshape(-1, 1) - hp_filter(np.log(df["CLVMNACSCAB1GQDE"]))

plt.figure(figsize=(15,5))

plt.title("The cyclical component")

plt.plot(cyclical)
plt.xticks(range(0,T,28),df["DATE"][list(range(0,T,28))])


# Task5
def ma(theta, T):
    ma_data = np.zeros(T)
    wn = np.random.normal(size=T)
    for i in range(1,T):
        ma_data[i] = wn[i] + theta * wn[i-1]
    return ma_data

import statsmodels.api as sm
from statsmodels.tsa.stattools import acf
from statsmodels.tsa.stattools import pacf

# plot ACF and PACF for theta=0.3
sm.graphics.tsa.plot_acf(ma(0.3, 500), lags=50)
plt.title("Autocorrelation for theta=0.3")
sm.graphics.tsa.plot_pacf(ma(0.3, 500), lags=50)
plt.title("Partial autocorrelation for theta=0.3")
plt.show()

# plot ACF and PACF for theta= -0.5
sm.graphics.tsa.plot_acf(ma(-0.5, 500), lags=50)
plt.title("Autocorrelation for theta=-0.5")
sm.graphics.tsa.plot_pacf(ma(-0.5, 500), lags=50)
plt.title("Partial autocorrelation for theta=-0.5")
plt.show()


#Compare your results with the expected patterns of the theoretical
#ACFs and PACFs of the processes.

#My answer
#Both time series exhibit the expected patterns for an MA(1) process:
#The ACF is significant at the first lag and aligns with the value of θ, decaying quickly beyond that.
#The PACF shows a clear cutoff after the first lag, which characterizes the MA(1) model.
#These observations confirm the theoretical properties of the MA(1) process where the ACF should show a significant correlation at lag 1 and almost none thereafter, while the PACF should exhibit a sharp cutoff after lag 1. The plots align well with these expectations, validating the correctness of the generated data and the applied statistical methods.