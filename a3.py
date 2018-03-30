import pandas as pd
import numpy as np
import datetime
import aubio
import statsmodels.api as sm
from sklearn import datasets
from sklearn import linear_model
from scipy.stats import kurtosis, skew
from scipy import stats
from scipy.signal import argrelextrema
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_fscore_support
import csv
import matplotlib.pyplot as plt
import os

def GSR_Class(filename_csv, filename_mp3, stress_label, file_label):
    # load csv, skip 6 rows since they are junk
    #'LOG14_2012_timeUnset corey no stress.csv'
    data = pd.read_csv(filename_csv, skiprows=6)
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

    # instantiate dataframe to contain features and labels
    data_columns = {'mean_gsr': [], 'max_gsr': [], 'min_gsr': [], 'slope_gsr': [], 'mean_gsr_peak_height': [], 'gsr_peak_amount': [], 'gsr_25_peak_quantile': [], 'gsr_50_peak_quantile': [], 'gsr_75_peak_quantile': [], 'labels': []}
    gsr_dataframe = pd.DataFrame(data=data_columns)
    first_datetime = data['Time'].iloc[0]

    # iterate windows until window is empty
    while True:
        # window start/end time are the seconds when the window starts/ends
        window_start_time = 5 * window_number
        window_end_time = window_start_time + window_duration

        # creates a datetime (random date) with the start/end times (used for formatting in window)
        start_time = datetime.datetime(1,1,1,0,0,0) + datetime.timedelta(0,window_start_time) + datetime.timedelta(hours=data['Time'].iloc[0].hour,minutes=data['Time'].iloc[0].minute, seconds=data['Time'].iloc[0].second)
        end_time = datetime.datetime(1,1,1,0,0,0) + datetime.timedelta(0,window_end_time) +  datetime.timedelta(hours=data['Time'].iloc[0].hour,minutes=data['Time'].iloc[0].minute, seconds=data['Time'].iloc[0].second)

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

        window_peak_differences = np.diff(peak_indexes)

        window_peak_percentiles = []

        if not window_peak_differences.any():
            window_peak_differences = [0]

        window_peak_percentiles.append([np.percentile(window_peak_differences, 25), np.percentile(window_peak_differences, 50), np.percentile(window_peak_differences, 75)])

        window_mean_peak_height = np.mean(window_peaks)

        # 0 for unstressed
        #window_label = 0

        # gsr_25, gsr_50, gsr_75, gsr_peak_number, label, max_gsr, mean_gsr, mean_gsr_peak_height, min_gsr, slope_gsr in that order
        gsr_dataframe.loc[window_number] = [window_peak_percentiles[0][0],window_peak_percentiles[0][1],window_peak_percentiles[0][2],window_peaks_number,stress_label,window_max,window_mean,window_mean_peak_height,window_min,window_slope]



        # iterate to next window
        window_number += 1

    #audio
    SAMPLE_RATE = 32

    def extract_pitch(file, rate=44096):
        window = int(rate / SAMPLE_RATE)
        audio_src = aubio.source(file, rate, window)
        pitch_obj = aubio.pitch('yin', window, window, rate)
        pitch_obj.set_unit("Hz")
        pitch_obj.set_silence(-40)
        pitches = []
        total_frames = 0
        while True:
            samples, read = audio_src()
            pitch = pitch_obj(samples)[0]
            if pitch > 1000:  # Not likely, just zero it out...
                pitch = 0
            pitches.append(pitch)
            total_frames += read
            if read < window: break
        return pitches

    #'corey no stress.wav'
    pitches = extract_pitch(filename_mp3)

    # 320 pitches = 1 second,
    # 5 second window.

    def split_windows (array):
        counter = 0
        ret_array = []
        for i, pitch in enumerate(array):
            if i == len(array)-1:
                yield ret_array
            elif counter < 157:
                ret_array.append(pitch)
                counter+=1
            else:
                counter = 0
                yield ret_array
                ret_array = []

    audio_5sec = list(split_windows(pitches))

    std_dev = []


    def calculations_std(array):
        for set_pitch in range(1,len(array)):
            yield np.std(np.asarray(array[set_pitch-1] + array[set_pitch]))

    def calculations_min(array):
        for set_pitch in range(1,len(array)):
            yield np.min(np.asarray(array[set_pitch-1] + array[set_pitch]))

    def calculations_max(array):
        for set_pitch in range(1,len(array)):
            yield np.max(np.asarray(array[set_pitch-1] + array[set_pitch]))

    def calculations_mean(array):
        for set_pitch in range(1,len(array)):
            yield np.mean(np.asarray(array[set_pitch-1] + array[set_pitch]))

    def calculations_kurtosis(array):
        for set_pitch in range(1,len(array)):
            yield kurtosis(array[set_pitch-1] + array[set_pitch])

    def calculations_skew(array):
        for set_pitch in range(1,len(array)):
            yield skew(array[set_pitch-1] + array[set_pitch])

    # slope, intercept, r_value, p_value, std_err = stats.linregress(X, y)
    def calculations_linreg(array):
        for set_pitch in range(1,len(array)):
            total_length = len(array[set_pitch]) + len(array[set_pitch-1])
            X = list(range(0, total_length))
            slope, intercept, r_value, p_value, std_err = stats.linregress(X, array[set_pitch-1] + array[set_pitch])
            yield slope

    std_array = list(calculations_std(audio_5sec))
    min_array = list(calculations_min(audio_5sec))
    max_array = list(calculations_max(audio_5sec))
    mean_array = list(calculations_mean(audio_5sec))
    kurtosis_array = list(calculations_kurtosis(audio_5sec))
    skew_array = list(calculations_skew(audio_5sec))
    linreg_array = list(calculations_linreg(audio_5sec))



    # instantiate dataframe to contain features and labels
    audio_data_columns = {'audio_mean': [], 'audio_max': [], 'audio_min': [], 'audio_slope': [], 'audio_std': [], 'audio_skew': [], 'audio_kurtosis': [], 'labels': []}
    audio_dataframe = pd.DataFrame(data=audio_data_columns)

    #Columns: [audio_kurtosis, audio_max, audio_mean, audio_min, audio_skew, audio_slope, audio_std, labels]
    for i in range(0, len(std_array)):
        audio_dataframe.loc[i] = [kurtosis_array[i], max_array[i], mean_array[i], min_array[i], skew_array[i], linreg_array[i], std_array[i], stress_label]

    audio_rows, audio_cols = audio_dataframe.shape
    #print('ROW COUNT FOR AUDIO CSV: ', audio_rows)

    gsr_rows, gsr_cols = gsr_dataframe.shape
    #print('ROW COUNT FOR GSR CSV: ', gsr_rows)


    if gsr_rows > audio_rows:
        diff = gsr_rows - audio_rows
        gsr_dataframe.drop(gsr_dataframe.tail(diff).index,inplace=True)
    elif audio_rows > gsr_rows:
        diff = audio_rows - gsr_rows
        audio_dataframe.drop(gsr_dataframe.tail(diff).index,inplace=True)

    # save final audio dataframe to csv
    audio_dataframe.to_csv('audio_features_' + file_label + '.csv', sep=' ', encoding='utf-8')

    # save final gsr dataframe to csv
    gsr_dataframe.to_csv('gsr_features_' + file_label + '.csv', sep=' ', encoding='utf-8')

    #print('gsr data saved in gsr_features.csv')
    #print('audio data saved in audio_features.csv')

    return (audio_dataframe, gsr_dataframe)


def get_dataframes():
    audio_data_columns = {'audio_mean': [], 'audio_max': [], 'audio_min': [], 'audio_slope': [], 'audio_std': [], 'audio_skew': [], 'audio_kurtosis': [], 'labels': []}
    gsr_data_columns = {'mean_gsr': [], 'max_gsr': [], 'min_gsr': [], 'slope_gsr': [], 'mean_gsr_peak_height': [], 'gsr_peak_amount': [], 'gsr_25_peak_quantile': [], 'gsr_50_peak_quantile': [], 'gsr_75_peak_quantile': [], 'labels': []}
    audio_dataframe = pd.DataFrame(data=audio_data_columns)
    gsr_dataframe = pd.DataFrame(data=gsr_data_columns)
    for participant in os.listdir('data'):
        stress_csv = ''
        stress_audio = ''
        unstress_csv = ''
        unstress_audio = ''

        #print('Stress')
        for stress_file in os.listdir('data/' + participant + '/Stressed'):
            #print (stress_file)
            if '.csv' in stress_file:
                stress_csv = stress_file
            if '.mp3' in stress_file or '.wav' in stress_file:
                stress_audio = stress_file

        #print('NO STRESS')
        for no_stress_file in os.listdir('data/' + participant + '/Unstressed'):
            #print (no_stress_file)
            if '.csv' in no_stress_file:
                unstress_csv = no_stress_file
            if '.mp3' in no_stress_file or '.wav' in no_stress_file:
                unstress_audio = no_stress_file

        #print(unstress_audio)

        no_stress_audio_dataframe, no_stress_gsr_dataframe = GSR_Class('data/' + participant + '/Unstressed/' + unstress_csv, 'data/' + participant + '/Unstressed/' + unstress_audio, 0, participant)
        stress_audio_dataframe, stress_gsr_dataframe = GSR_Class('data/' + participant + '/Stressed/' + stress_csv, 'data/' + participant + '/Stressed/' + stress_audio, 1, participant)

        audio_dataframe = audio_dataframe.append(no_stress_audio_dataframe, ignore_index=True)
        audio_dataframe = audio_dataframe.append(stress_audio_dataframe, ignore_index=True)

        gsr_dataframe = gsr_dataframe.append(no_stress_gsr_dataframe, ignore_index=True)
        gsr_dataframe = gsr_dataframe.append(stress_gsr_dataframe, ignore_index=True)

        #no_stress_audio_dataframe, no_stress_gsr_dataframe = GSR_Class('LOG14_2012_timeUnset corey no stress.csv', 'corey no stress.wav', 0, 'corey')

        #stress_audio_dataframe, stress_gsr_dataframe = GSR_Class(filename, 'corey no stress.wav', 0, 'corey')
        #audio_dataframe = no_stress_audio_dataframe.append(stress_audio_dataframe, ignore_index=True)

    return (audio_dataframe, gsr_dataframe)

audio_dataframe, gsr_dataframe = get_dataframes()
#no_stress_audio_dataframe, no_stress_gsr_dataframe = GSR_Class('LOG14_2012_timeUnset corey no stress.csv', 'corey no stress.wav', 0, 'corey')
#stress_audio_dataframe, stress_gsr_dataframe = GSR_Class('LOG13_2012_timeUnset corey stress.csv', 'Corey stress voice.wav', 1, 'corey_stress')

#audio_dataframe = no_stress_audio_dataframe.append(stress_audio_dataframe, ignore_index=True)
#gsr_dataframe = stress_gsr_dataframe.append(no_stress_gsr_dataframe, ignore_index=True)

def to2d (array):
    for a in array:
        if not np.isnan(a):
            yield [a]
        else:
            # print('FOUND NAN')
            yield [0]

voice_data = {'max':[], 'min':[], 'mean':[], 'slope':[], 'std':[], 'kurtosis':[], 'skew':[]}
voice_data['max'] += list(to2d(audio_dataframe['audio_max'].as_matrix().tolist()))
voice_data['min'] += list(to2d(audio_dataframe['audio_min'].as_matrix().tolist()))
voice_data['mean'] += list(to2d(audio_dataframe['audio_mean'].as_matrix().tolist()))
voice_data['slope'] += list(to2d(audio_dataframe['audio_slope'].as_matrix().tolist()))
voice_data['std'] += list(to2d(audio_dataframe['audio_std'].as_matrix().tolist()))
voice_data['kurtosis'] += list(to2d(audio_dataframe['audio_kurtosis'].as_matrix().tolist()))
voice_data['skew'] += list(to2d(audio_dataframe['audio_skew'].as_matrix().tolist()))

label_data = []
label_data += audio_dataframe['labels'].as_matrix().tolist()

#  gsr_25_peak_quantile gsr_50_peak_quantile gsr_75_peak_quantile gsr_peak_amount labels max_gsr mean_gsr mean_gsr_peak_height min_gsr slope_gsr
GSP_data = {'max':[], 'min':[], 'mean':[], 'slope':[], 'peak mean':[], 'peak num':[], '25 quantile':[], '50 quantile':[], '75 quantile':[]}
#print('List: ', len(gsr_dataframe['max_gsr'].as_matrix()))
GSP_data['max'] += list(to2d(gsr_dataframe['max_gsr'].as_matrix().tolist()))
GSP_data['min'] += list(to2d(gsr_dataframe['min_gsr'].as_matrix().tolist()))#gsr_dataframe['min_gsr'].as_matrix().tolist()
GSP_data['mean'] += list(to2d(gsr_dataframe['mean_gsr'].as_matrix().tolist()))#gsr_dataframe['mean_gsr'].as_matrix().tolist()
GSP_data['slope'] += list(to2d(gsr_dataframe['slope_gsr'].as_matrix().tolist()))#gsr_dataframe['slope_gsr'].as_matrix().tolist()
GSP_data['peak mean'] += list(to2d(gsr_dataframe['mean_gsr_peak_height'].as_matrix().tolist()))#gsr_dataframe['mean_gsr_peak_height'].as_matrix().tolist()
GSP_data['peak num'] += list(to2d(gsr_dataframe['gsr_peak_amount'].as_matrix().tolist()))#gsr_dataframe['gsr_peak_amount'].as_matrix().tolist()
GSP_data['25 quantile'] += list(to2d(gsr_dataframe['gsr_25_peak_quantile'].as_matrix().tolist()))
GSP_data['50 quantile'] += list(to2d(gsr_dataframe['gsr_50_peak_quantile'].as_matrix().tolist()))
GSP_data['75 quantile'] += list(to2d(gsr_dataframe['gsr_75_peak_quantile'].as_matrix().tolist()))


def makeModels(set1, set2, key):
    finished = []
    results = {}

    for name, data in set1.items(): #Make models and test the model
        for name2, data2 in set2.items():
            #print(name2, data2)
            if name2 not in finished:
                merged_data = [data[x] + data2[x] for x in range(len(data))]
                #print('MERGED DATA ',merged_data )
                if name == name2:
                    results[name] = modelData(merged_data, key)
                else:
                    results[name + '+' + name2] = modelData(merged_data, key)
        finished.append(name)
    return results

def modelData(data, key):
    model = GaussianNB()
    data_train, data_test, labels_train, labels_test = train_test_split(data, key, test_size=0.3)
    model.fit(data_train, labels_train)
    predicted = model.predict(data_test)

    (precision, recall, fscore, support) = precision_recall_fscore_support(labels_test, predicted)

    return([sum(precision) / float(len(precision)), sum(recall) / float(len(recall)), sum(fscore) / float(len(fscore))])

def output(name, results):
    fig, ax = plt.subplots()
    index = np.arange(len(results))
    bar_width = 0.3

    precision = ax.bar(index - bar_width,[int(x[0] * 100) for x in results.values()], bar_width,
                 color='y',
                 label='precision')

    recall = ax.bar(index, [int(x[1] * 100) for x in results.values()], bar_width,
                 color='g',
                 label='recall')

    fscore = ax.bar(index + bar_width, [int(x[2] * 100) for x in results.values()], bar_width,
                 color='b',
                 label='fscore')

    ax.autoscale(tight=True)

    plt.xticks(index, results.keys(), fontsize = 8)
    plt.title(name)
    plt.legend()
    plt.show()

output('GSP Data Results', makeModels(GSP_data, GSP_data, label_data))
output('Voice Data Results', makeModels(voice_data, voice_data, label_data))
output('Combined Data Results', makeModels(GSP_data, voice_data, label_data))
