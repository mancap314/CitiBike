#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import division
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
import os.path
import time
import itertools
import random
from numpy import fft

def get_weekday(date):
    weekdays = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    return weekdays[date.weekday()]

def deg2rad(deg):
  return deg * (math.pi / 180)

def get_distance(lat1,lon1,lat2,lon2):
    """
    gets the distance in km between two points given their latitude and longitude
    :param lat1: latitude of point 1
    :param lon1: longitude of point 2
    :param lat2: latitude of point 2
    :param lon2: longitude of point 2
    :return: distance in km between point 1 and point 2
    """
    radius = 6371 # Radius of the earth in km
    dLat = deg2rad(lat2-lat1)
    dLon = deg2rad(lon2-lon1)
    a = math.sin(dLat/2) * math.sin(dLat/2) + math.cos(deg2rad(lat1)) * math.cos(deg2rad(lat2)) * math.sin(dLon/2) * math.sin(dLon/2)
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a));
    d = radius * c # Distance in km
    return d;

class Station:
    def __init__(self, id, name, latitude, longitude):
        self.id = id
        self.name = name
        self.latitude = latitude
        self.longitude = longitude

    def get_id(self):
        return self.id

    def get_name(self):
        return self.name

    def get_latitude(self):
        return self.latitude

    def get_longitude(self):
        return self.longitude

    def get_distance(self, station):
        return get_distance(self.latitude, self.longitude, station.get_latitude(), station.get_longitude())

def get_stations(new=False):
    """
    :param new: should be recomputed or not
    :return: dataframe of the stations
    """
    if new == False and os.path.exists('stations.pkl'):
        return pd.read_pickle('stations.pkl')

    data = pd.read_csv('201602-citibike-tripdata.csv')
    print 'data columns:'
    print data.columns

    stations_start = data[['start station id', 'start station name', 'start station latitude', 'start station longitude']]\
        .drop_duplicates()
    stations_start.columns = ['id', 'name', 'latitude', 'longitude']

    stations_end = data[
        ['end station id', 'end station name', 'end station latitude', 'end station longitude']] \
        .drop_duplicates()
    stations_end.columns = ['id', 'name', 'latitude', 'longitude']

    dfs = [stations_start, stations_end]
    stations = pd.concat(dfs).drop_duplicates()

    print 'stations...:'
    print stations

    station_instances = [Station(stations.ix[idx]['id'], stations.ix[idx]['name'], stations.ix[idx]['latitude'], stations.ix[idx]['longitude']) for idx in stations.index]

    station_pairs = {'id': [], 'closest': []}
    for station in station_instances:
        current_id = station.get_id()
        other_stations = [st for st in station_instances if st.get_id() != current_id]
        distances = [[st.get_id(), station.get_distance(st)] for st in other_stations]
        min_distance = min([d[1] for d in distances])
        id_closest = [d[0] for d in distances if d[1] == min_distance][0]
        station_pairs['id'].append(current_id)
        station_pairs['closest'].append(id_closest)

    station_pairs = pd.DataFrame(station_pairs)
    stations = stations.merge(station_pairs, on='id')
    stations.to_pickle('stations.pkl')
    return stations

stations = get_stations()

def get_closest_station(station_id):
    """
    gets closest station according to their geographical distance
    :param station_id: id of the station
    :return: station id of the closest station
    """
    global stations
    closest_station = stations.loc[stations['id'] == station_id, 'closest'].values[0]
    return closest_station

def get_station_attr(id, attr):
    """
    gets the given attribute for a station with a given id
    :param id: id of the station
    :param attr: array of attributes
    :return: the attributes for the station
    """
    stations = get_stations()
    return stations[stations['id'] == id][[attr]].values[0][0]


def get_data(new=False):
    if new == False and os.path.exists('data.pkl'):
        data = pd.read_pickle('data.pkl')
    else:
        data = pd.read_csv('201602-citibike-tripdata.csv')
        data = data.drop(['start station name', 'start station latitude', 'start station longitude', 'end station name',
                          'end station latitude', 'end station longitude'], axis=1)


        # Unknow values of 'gender' are encoded by 0 => 0 is replaced by Null
        data.loc[data['gender'] == 0, 'gender'] = np.nan
        data['stoptime'] = [pd.Timestamp(t) for t in data['stoptime']]
        data['starttime'] = data['starttime'].apply(lambda x: pd.Timestamp(x))

        data['start_closest'] = data['start station id'].apply(lambda x: get_closest_station(x))
        data['end_closest'] = data['end station id'].apply(lambda x: get_closest_station(x))

        # NOTICE: First try, needed about 2 hours to run, now done by method get_unbalance: about 20 sec
        # print 'ubalance start...'
        # tstart = time.time()
        # unbalance_start = data.apply(lambda x: get_unbalance(x['start station id'], x['starttime']), axis=1)
        # print 'unbalance start:', str(time.time() - tstart)
        # unbalance_start_closest = data.apply(lambda x: get_unbalance(x['start_closest'], x['starttime']), axis=1)
        # unbalance_end = data.apply(lambda x: get_unbalance(x['end station id'], x['stoptime']), axis=1)
        # unbalance_end_closest = data.apply(lambda x: get_unbalance(x['end_closest'], x['stoptime']), axis=1)
        #
        # data['unbalance_start'] = unbalance_start
        # data['unbalance_start_closest'] = unbalance_start_closest
        # data['unbalance_end'] = unbalance_end
        # data['unbalance_end_closest'] = unbalance_end_closest

        data.to_pickle('data.pkl')
    return data

def get_mean_tripduration_by_startstation():
    data = get_data()
    mean_duration_by_station = data.groupby(['start station id'], as_index=False)[
        'tripduration'].mean()  # as_index=False to get a DataFrame
    mean_duration_by_station['tripduration'] = mean_duration_by_station['tripduration'].apply(
        lambda x: round(x / 60))  # trip duration in minutes
    mean_duration_by_station.columns = ['station id', 'mean tripduration']
    return mean_duration_by_station.sort_values(by='mean tripduration', ascending=False) #biggest values first

def get_tripduration_by_bike():
    """
    :return: total trip duration by bike in hours
    """
    data = get_data()
    trip_duration_by_bike = data.groupby(['bikeid'], as_index=False)['tripduration'].sum()
    trip_duration_by_bike['tripduration'] = trip_duration_by_bike['tripduration'].apply(
        lambda x: x / 3600)  # trip duration in hours
    return trip_duration_by_bike

def get_threshold_outlier():
    """
    :return: total trip duration treshold for overused bikes
    """
    data = get_data()
    trip_duration_by_bike = get_tripduration_by_bike()
    bplot = plt.boxplot(trip_duration_by_bike['tripduration'])
    plt.title('Repartition of Bikes Use Duration')
    plt.ylabel('Use Duration (hour)')
    plt.show()
    return  [item.get_ydata()[1] for item in bplot['whiskers']][1] #outliers: (+) on the graph

def percent_difference(arg1, arg2):
    return 100 * (arg1 - arg2) / arg1

def get_frequentation_diff():
    """
    difference in station frequentation overused / not-overused bikes
    :return: frequentation difference per station in percent
    """
    data = get_data()
    trip_duration_by_bike = get_tripduration_by_bike()
    threshold = get_threshold_outlier()

    # Outliers:
    outlier_bikes = trip_duration_by_bike[trip_duration_by_bike['tripduration'] > threshold]
    outliers_bike_ids = np.array(outlier_bikes[['bikeid']]).flatten()
    data_outliers_bike = data[data['bikeid'].isin(outliers_bike_ids)]
    data_outliers_bike['freq'] = data_outliers_bike.groupby('start station id')['start station id'].transform('count')
    start_station_outliers = data_outliers_bike[['start station id', 'freq']]\
        .drop_duplicates().sort_values(by='freq', ascending=False)
    start_station_outliers['freq'] = [round(1000 * fr / sum(start_station_outliers['freq']), 2) for fr in
                                      start_station_outliers['freq']]

    # Not outliers:
    not_outlier_bikes = trip_duration_by_bike[trip_duration_by_bike['tripduration'] < threshold]
    not_outliers_bike_ids = np.array(not_outlier_bikes[['bikeid']]).flatten()

    data_not_outliers_bike = data[data['bikeid'].isin(not_outliers_bike_ids)]
    data_not_outliers_bike['freq'] = data_not_outliers_bike.groupby('start station id')['start station id'].transform(
        'count')
    start_station_not_outliers = data_not_outliers_bike[['start station id', 'freq']].drop_duplicates().sort_values(
        by='freq', ascending=False)
    start_station_not_outliers['freq'] = [round(1000 * fr / sum(start_station_not_outliers['freq']), 2) for fr in
                                          start_station_not_outliers['freq']]

    # Comparison:
    comparison_outliers = start_station_outliers.merge(start_station_not_outliers, on='start station id')
    comparison_outliers.columns = ['start station id', 'permille_not_outlier', 'permille_outlier']
    comparison_outliers['abs_diff_percent'] = comparison_outliers \
        .apply(lambda row: abs(percent_difference(row['permille_outlier'], row['permille_not_outlier'])), axis=1)
    comparison_outliers['diff_percent'] = comparison_outliers \
        .apply(lambda row: percent_difference(row['permille_outlier'], row['permille_not_outlier']), axis=1)
    comparison_outliers = comparison_outliers.sort_values(by='abs_diff_percent', ascending=False)

    return comparison_outliers


def cartesian(df1, df2):
    rows = itertools.product(df1.iterrows(), df2.iterrows())

    df = pd.DataFrame(left.append(right) for (_, left), (_, right) in rows)
    return df.reset_index(drop=True)

def get_all(new=False):
    """
    :param new: recompute or not
    :return: all dates by hours in the time span of data frame
    """
    if new == False and os.path.exists('all.pkl'):
        return pd.read_pickle('all.pkl')

    all_days = pd.DataFrame({'day': pd.date_range('02.01.2016', periods=29, freq='d')})
    all_days['day'] = all_days['day'].apply(lambda x: pd.Timestamp(x))
    all_days['day'] = all_days['day'].dt.strftime('%m.%d.%Y')
    all_hours = pd.DataFrame({'hour': pd.date_range('02.01.2016', periods=24, freq='H')})
    all_hours['hour'] = all_hours['hour'].apply(lambda x: pd.Timestamp(x))
    all_hours['hour'] = all_hours['hour'].dt.strftime('%H')
    all_dates = cartesian(all_days, all_hours)

    all_stations = get_stations()[['id']] #watch out: global stations
    all_stations.columns = ['station id']

    all_ = cartesian(all_stations, all_dates)
    all_.to_pickle('all.pkl')
    return all_

def get_fdata(new=False):
    """
    :param new: recompute or not
    :return: data with desired features
    """
    if new == False and os.path.exists('fdata.pkl'):
        return pd.read_pickle('fdata.pkl')

    data = get_data()
    features = ['starttime', 'stoptime', 'start station id', 'start_closest', 'end station id', 'end_closest']
    fdata = data[features]
    tstart = time.time()
    print 'stime:'
    fdata['startday'] = fdata['starttime'].dt.strftime('%m.%d.%Y')
    fdata['starthour'] = fdata['starttime'].dt.strftime('%H')
    fdata['stopday'] = fdata['stoptime'].dt.strftime('%m.%d.%Y')
    fdata['stophour'] = fdata['stoptime'].dt.strftime('%H')
    fdata.to_pickle('fdata.pkl')
    return fdata

def get_unbalance(fdata):
    """
    :param fdata: data with desired features
    :return: nbalance by station, day and hour
    """
    fdata['one'] = [1.0]*fdata.shape[0]

    all_ = get_all()

    #Computing cumulated sum of departures by station/day through hours
    departures = fdata[['start station id', 'startday', 'starthour', 'one']]
    departures.columns = ['station id', 'day', 'hour', 'one']
    departures = all_.merge(departures, on=['station id', 'day', 'hour'], how='outer', sort=True)
    departures['one'] = departures['one'].fillna(0.0)
    departures = departures.groupby(['station id', 'day', 'hour'])['one'].sum().groupby(level=[0, 1]).cumsum()
    departures = departures.reset_index()
    departures.columns = ['station id', 'day', 'hour', 'cumday_departures']
    departures = departures.sort(columns=['station id', 'day', 'hour'])

    #Same for arrivals
    arrivals = fdata[['end station id', 'stopday', 'stophour', 'one']]
    arrivals.columns = ['station id', 'day', 'hour', 'one']
    arrivals = all_.merge(arrivals, on=['station id', 'day', 'hour'], how='outer', sort=True)
    arrivals['one'] = arrivals['one'].fillna(0.0)
    arrivals = arrivals.groupby(['station id', 'day', 'hour'])['one'].sum().groupby(level=[0, 1]).cumsum()
    arrivals = arrivals.reset_index()
    arrivals.columns = ['station id', 'day', 'hour', 'cumday_arrivals']
    arrivals = arrivals.sort(columns=['station id', 'day', 'hour'])

    saldo = departures.merge(arrivals, on=['station id', 'day', 'hour'])
    saldo['diff'] = saldo.apply(lambda x: x[4]-x[3], axis=1)

    return saldo

def preprocess_simulation(fdata):
    unbalance = get_unbalance(get_fdata())
    mean_diff_index = np.mean((unbalance[['diff']]) ** 2)
    print 'mean_diff_index before incitation:', mean_diff_index

    fdata = get_fdata()
    features = ['start station id', 'start_closest', 'end station id', 'end_closest', 'startday', 'starthour', 'stopday', 'stophour']
    fdata = fdata[features]
    #Computing unbalance
    # start station:
    fdata = fdata.merge(unbalance, left_on=['start station id', 'startday', 'starthour'], right_on=['station id', 'day', 'hour'])
    fdata = fdata[features+['diff']]
    features = features+['diff_start']
    fdata.columns = features
    # start_closest:
    fdata = fdata.merge(unbalance, left_on=['start_closest', 'startday', 'starthour'], right_on=['station id', 'day', 'hour'])
    fdata = fdata[features+['diff']]
    features = features+['diff_start_closest']
    fdata.columns = features
    # end station:
    fdata = fdata.merge(unbalance, left_on=['end station id', 'stopday', 'stophour'], right_on=['station id', 'day', 'hour'])
    fdata = fdata[features+['diff']]
    features = features+['diff_end']
    fdata.columns = features
    # end_closest:
    fdata = fdata.merge(unbalance, left_on=['end_closest', 'stopday', 'stophour'], right_on=['station id', 'day', 'hour'])
    fdata = fdata[features+['diff']]
    features = features+['diff_end_closest']
    fdata.columns = features
    fdata['ind'] = list(range(len(fdata.index)))
    # indices where nearest neighbor for start station is better
    inds_startc_better = fdata[fdata['diff_start'] < fdata['diff_start_closest']][['ind']]
    # indices where nearest neighbor for end station is better
    inds_endc_better = fdata[fdata['diff_end'] > fdata['diff_end_closest']][['ind']]
    return fdata, inds_startc_better, inds_endc_better

#In almost half of the case, picking up or dropping of the bike one more station away would improve the unbalance of the BSS

def simulate_incitation(fdata, rate_start=0.2, rate_stop=0.2):
    """
    Replacing start/stop station by its nearest neighbor if better for unbalance at randam with given acceptance rate
    :param fdata: data with features
    :param rate_start: acceptance rate for incitation to switch start station if better for unbalance
    :param rate_stop: acceptance rate for incitation to switch stop station if better for unbalance
    :return: average unbalance index
    """
    fdata, inds_startc, inds_endc = preprocess_simulation(fdata)
    N_start, N_end = len(inds_startc), len(inds_endc)
    ind_start_tochange = random.sample(range(N_start), int(round(rate_start * N_start)))
    ind_end_tochange = random.sample(range(N_end), int(round(rate_stop * N_end)))

    start_tochange = [inds_startc['ind'].iloc[i]+1 for i in ind_start_tochange # +1 because applies to next hour
                      if inds_startc['ind'].iloc[i] % (24*29) != 0 #24*29: hours a day * #day: #row/station => avoids shift to other station
                      and inds_startc['ind'].iloc[i]+1 < N_start] #avoids out_of_index
    end_tochange = [inds_endc['ind'].iloc[i]+1 for i in ind_end_tochange
                    if inds_endc['ind'].iloc[i] % (24*29) != 0
                    and inds_endc['ind'].iloc[i]+1 < N_end]

    # Replaces start station by its closest station for indice in sample
    fdata.loc[fdata['ind'].isin(start_tochange), 'start station id'] = fdata['start_closest'].iloc[start_tochange]
    # Same for end station
    fdata.loc[fdata['ind'].isin(end_tochange), 'end station id'] = fdata['end_closest'].iloc[end_tochange]
    # Computing average unbalance index
    unbalance = get_unbalance(fdata)
    mean_diff_index = np.mean((unbalance[['diff']]) ** 2)
    return mean_diff_index

# score = simulate_incitation(fdata, inds_startc_better, inds_endc_better)

# [94.952171, 94.675849, 94.799962, 94.952046, 94.616024]
# about 5% improvement of the score

######
# Time series analysis

def get_tseries(new=False):
    if new == False and os.path.exists('tseries.pkl'):
        return pd.read_pickle('tseries.pkl')
    fdata = get_fdata()
    unbalance = get_unbalance(fdata)
    unbalance['date'] = pd.to_datetime(unbalance['day'] + ' ' +unbalance['hour'])
    unbalance = unbalance[['date', 'station id', 'diff']]
    unbalance.to_pickle('tseries.pkl')
    return unbalance

def fourier_extrapolation(x, n_predict):
    n = x.size
    n_harm = 20  # number of harmonics in model
    t = np.arange(0, n)
    p = np.polyfit(t, x, 1)  # find linear trend in x
    x_notrend = x - p[0] * t  # detrended x
    x_freqdom = fft.fft(x_notrend)  # detrended x in frequency domain
    f = fft.fftfreq(n)  # frequencies
    indexes = range(n)
    # sort indexes by frequency, lower -> higher
    indexes.sort(key=lambda i: np.absolute(f[i]))

    t = np.arange(0, n + n_predict)
    restored_sig = np.zeros(t.size)
    for i in indexes[:1 + n_harm * 2]:
        ampli = np.absolute(x_freqdom[i]) / n  # amplitude
        phase = np.angle(x_freqdom[i])  # phase
        restored_sig += ampli * np.cos(2 * np.pi * f[i] * t + phase)
    return restored_sig + p[0] * t


def plot_extrapolation(tscore_by_date, ndays_topredict):
    days = tscore_by_date['date'].dt.strftime('%m.%d').drop_duplicates()
    array = tscore_by_date['score']
    ndays_predicted = 29-ndays_topredict #because 29 days in dataframe
    extrapolation = fourier_extrapolation(array.iloc[range(0, 24*ndays_predicted)], ndays_topredict*24)
    plt.plot(np.arange(0, 24*ndays_predicted), extrapolation[0:24*ndays_predicted], 'r', label='extrapolation')
    plt.plot(np.arange(24*(ndays_topredict), 24*29), extrapolation[24*ndays_topredict:24*29], 'r--', label='prediction extrapolation')
    plt.plot(np.arange(0, array.size), array, 'b', label='real unbalance index', linewidth=3)
    plt.xticks(np.arange(0, 29*24, 24), days, rotation='vertical')
    plt.ylim([0, 540])
    plt.legend()
    plt.title('Extrapolation and Prediction of Unbalance Index \n with Fourier Transformation')
    plt.show()

### Preprocess frequentation difference
freqdiff = get_frequentation_diff()
freqdiff = freqdiff.iloc[range(10)]
stations = freqdiff['start station id']
y_pos = np.arange(len(stations))
diffs = freqdiff['diff_percent']

### Plot frequentation differences between overused and not-overused bikes
plt.bar(y_pos, diffs, align='center', alpha=1)
plt.xticks(y_pos, stations, rotation='vertical')
plt.xlim(min(y_pos)-0.5, max(y_pos)+0.5)
plt.axhline(y=0, linewidth=1, color='k')
plt.ylabel('frequentation difference (in %)')
plt.xlabel('station id')
plt.title('The 10 highest differences in station frequentation \n between overused and not-overused bikes')
plt.show()

### Preprocessing for mapping on CartDB
tseries = get_tseries()
tscores = tseries.copy()
tscores['score'] = tscores['diff'].apply(lambda x: x ** 2)
tscores_by_station = tscores.groupby(['station id'], as_index=False)['score'].mean()
tscores_by_station = tscores_by_station.sort(['score'], ascending=False)
tscores_by_station30 = tscores_by_station.iloc[range(30)]
tscores_by_station30.columns = ['id', 'Unbalance Index']
tscores_by_station30['latitude'] = tscores_by_station30['id'].apply(lambda x: get_station_attr(x, 'latitude'))
tscores_by_station30['longitude'] = tscores_by_station30['id'].apply(lambda x: get_station_attr(x, 'longitude'))
tscores_by_station30['name'] = tscores_by_station30['id'].apply(lambda x: get_station_attr(x, 'name'))
tscores_by_station30.to_csv('unbalance.csv')

### Preprocessing for heatmap
tscore_by_date = tscores.groupby(['date'], as_index=False)['score'].mean()
days = tscore_by_date['date'].dt.strftime('%a %m.%d').drop_duplicates()

# heatmap of unbalance index
data = tscore_by_date['score']
test=data.reshape(29,24)
plt.title('Unbalance Index by Day and Hour')
plt.xlim([0, 24])
plt.xlabel('Hour')
plt.ylabel('Day')
plt.yticks(np.arange(0.5, 29.5, 1), days)
plt.pcolor(test,cmap=plt.cm.Reds)
plt.colorbar()
plt.show()

### Plot Fourier extrapolation
plot_extrapolation(tscore_by_date, 7)