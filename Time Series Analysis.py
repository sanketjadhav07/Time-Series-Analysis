# -*- coding: utf-8 -*-
"""
@author: user - Sanket Jadhav
"""

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns

data = pd.read_csv(r'D:\07-SANKET\DATA SCIENCE\0.A - DS Projects\Time Series (Regression)\AirPassengers.csv')

data.head()

data.columns = ['Date', 'Number of Passengers']

data.head()

# Visualize the Time Series
def plot_df(df, x, y, title="", xlabel='Date', ylabel='Number of Passangers', 
dpi=100):
    plt.figure(figsize=(15,4), dpi=dpi)
    plt.plot(x,y, color='tab:red')
    plt.gca().set(title=title, xlabel=xlabel, ylabel=ylabel)
    plt.show()   
plot_df(data, x=data['Date'], y=data['Number of Passengers'], title='Number of US Airline passengers from 1949 to 1960') 

x = data['Date'].values
y1 = data['Number of Passengers'].values
fig, ax = plt.subplots(1, 1, figsize=(16,5), dpi= 120)
plt.fill_between(x, y1=y1, y2=-y1, alpha=0.5, linewidth=2, color='seagreen')
plt.ylim(-800, 800)
plt.title('Air Passengers (Two Side View)', fontsize=16)
plt.hlines(y=0, xmin=np.min(data['Date']), xmax=np.max(data['Date']), linewidth=.5)
plt.show()

# Patterns in a Time Series
def plot_df(data, x, y, title="", xlabel='Date', ylabel='Number of Passengers', dpi=100):
    plt.figure(figsize=(15,4), dpi=dpi)
    plt.plot(x, y, color='blue')
    plt.gca().set(title=title, xlabel=xlabel, ylabel=ylabel)
    plt.show()
    
plot_df(data, x=data['Date'], y=data['Number of Passengers'], title='Trend and Seasonality')
  
# Decomposition of a Time Series
from statsmodels.tsa.seasonal import seasonal_decompose
from dateutil.parser import parse

# Multiplicative Decomposition 
multiplicative_decomposition = seasonal_decompose(data['Number of Passengers'], model='multiplicative', period=30)

# Additive Decomposition
additive_decomposition = seasonal_decompose(data['Number of Passengers'], model='additive', period=30)

# Plot
plt.rcParams.update({'figure.figsize': (16,12)})
multiplicative_decomposition.plot().suptitle('Multiplicative Decomposition', fontsize=16)
plt.tight_layout(rect=[0, 0.03, 1, 0.95])

additive_decomposition.plot().suptitle('Additive Decomposition', fontsize=16)
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show() 

rand_numbers = np.random.randn(1000)
pd.Series(rand_numbers).plot(title='Random White Noise', color='b')

# Using scipy: Subtract the line of best fit
from scipy import signal
detrended = signal.detrend(data['Number of Passengers'].values)
plt.plot(detrended)
plt.title('Air Passengers detrended by subtracting the least squares fit', fontsize=16)

# Using statmodels: Subtracting the Trend Component
from statsmodels.tsa.seasonal import seasonal_decompose
result_mul = seasonal_decompose(data['Number of Passengers'], model='multiplicative', period=30)
detrended = data['Number of Passengers'].values - result_mul.trend
plt.plot(detrended)
plt.title('Air Passengers detrended by subtracting the trend component', fontsize=16)

# Subtracting the Trend Component
# Time Series Decomposition
result_mul = seasonal_decompose(data['Number of Passengers'], model='multiplicative', period=30)
# Deseasonalize
deseasonalized = data['Number of Passengers'].values / result_mul.seasonal
# Plot
plt.plot(deseasonalized)
plt.title('Air Passengers Deseasonalized', fontsize=16)
plt.plot()

# Test for seasonality
from pandas.plotting import autocorrelation_plot
# Draw Plot
plt.rcParams.update({'figure.figsize':(10,6), 'figure.dpi':120})
autocorrelation_plot(data['Number of Passengers'].tolist())

from statsmodels.tsa.stattools import acf, pacf
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
# Draw Plot
fig, axes = plt.subplots(1,2,figsize=(16,3), dpi= 100)
plot_acf(data['Number of Passengers'].tolist(), lags=50, ax=axes[0])
plot_pacf(data['Number of Passengers'].tolist(), lags=50, ax=axes[1])

from pandas.plotting import lag_plot
plt.rcParams.update({'ytick.left' : False, 'axes.titlepad':10})
# Plot
fig, axes = plt.subplots(1, 4, figsize=(10,3), sharex=True, sharey=True, dpi=100)
for i, ax in enumerate(axes.flatten()[:4]):
    lag_plot(data['Number of Passengers'], lag=i+1, ax=ax, c='firebrick')
    ax.set_title('Lag ' + str(i+1))
    
fig.suptitle('Lag Plots of Air Passengers', y=1.05)    
plt.show()


