import pandas as pd
import numpy as np
import datetime
import statsmodels.api as sm
from sklearn import datasets
from sklearn import linear_model
from scipy import stats
from scipy.signal import argrelextrema

# load csv, skip 6 rows since they are junk
data = pd.read_csv('LOG14_2012_timeUnset corey no stress.csv', skiprows=6)
# remove the row that is all dashes
data = data.drop(data.index[0])
# remove all columns except Time and EDA(uS)
data = data.drop(data.columns[[1,2,3,4,5,7]], axis=1)

# window numer is current window we are iterating at
window_number = 0
# the length of a window (constant)
window_duration = 10

# convert all times to datetime (will be set to today's date, but that is ignored)
data['Time'] = pd.to_datetime(data['Time'])

# iterate windows until window is empty
while True:

    # window start/end time are the seconds when the window starts/ends
    window_start_time = 5 * window_number
    window_end_time = window_start_time + window_duration

    # creates a datetime (random date) with the start/end times (used for formatting in window)
    start_time = datetime.datetime(1,1,1,0,0,0) + datetime.timedelta(0,window_start_time)
    end_time = datetime.datetime(1,1,1,0,0,0) + datetime.timedelta(0,window_end_time)

    # pull the current window using start/end time range times only (date is ignored)
    window = data[(data['Time'] >= str(start_time.time())) & (data['Time'] <= str(end_time.time()))]

    # if the window we get is empty we reached end of csv, break from loop
    if window.empty:
        break

    # get features for current window

    # linear regression
    # X values will be time, it must be in seconds
    # y are the EDA values
    X = []
    y = window['EDA(uS)']

    # convert each datetime to seconds, append to X
    for x in window['Time']:
        t = x.time()
        X.append(float("%0.3f" % ((t.hour * 60 + t.minute) * 60 + t.second + (t.microsecond*0.000001))))

    # get linear regression values (slope is what we weed)
    slope, intercept, r_value, p_value, std_err = stats.linregress(X, y)

    # get features
    window_slope = slope
    window_mean = window['EDA(uS)'].mean()
    window_max = window['EDA(uS)'].max()
    window_min = window['EDA(uS)'].min()

    # get peaks (local maxima)
    peak_indexes = argrelextrema(window['EDA(uS)'].as_matrix(), np.greater)[0].tolist()
    window_peaks_number = len(peak_indexes)
    window_peaks = []

    # get peaks from their indexed locations and append to window_peaks
    arr = window['EDA(uS)'].values.tolist()
    for i in peak_indexes:
        window_peaks.append(arr[i])

    

    break

    # iterate to next window
    window_number += 1