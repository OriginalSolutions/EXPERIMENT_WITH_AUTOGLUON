#!/usr/bin/env python3.8
#
import copy
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from datetime import timedelta
from time import time
from time import sleep
import copy
import requests    
from autogluon.timeseries import TimeSeriesDataFrame, TimeSeriesPredictor
from autogluon.tabular import TabularDataset, TabularPredictor
from functions_autogluon import functions
from functions_autogluon import BASE_URL, CONTEX, HEADERS, URL, LONG


#######################################################################################
#
# LOADING AND CONVERTING DATA TO PANDAS FORMAT
#
#######################################################################################
f = functions(BASE_URL, CONTEX, HEADERS, URL, LONG)
l = f.loading(BASE_URL, CONTEX, HEADERS, URL, LONG)

close_l, index_time_l = f.close_gate(l)
data_long_first          = f.data_frame(close_l, index_time_l, "Long")

data_long = data_long_first
data_long['Date'] = data_long.index
data_long.index   = data_long['Date']
data_long            = data_long.drop('Date', axis=1)
data_long.plot()
plt.show()
data = data_long

data['timestamp'] = data.index
data['item_id']      = 'Long'
data['target']        = data['Long']
print(data)


#######################################################################################
#
#    DIVISION OF DATA INTO TRAINING AND VALIDATION DENE
#
#######################################################################################
limit = int(200)

train = copy.deepcopy(data[ :-limit])
del train['Long']
train = train.reset_index(drop=True)

valid = copy.deepcopy(data[-limit: ])
del valid['Long']
valid = valid.reset_index(drop=True)

print(train)
print(valid)


#######################################################################################
#
#    TRAINING
#
#######################################################################################
data_train = TimeSeriesDataFrame(train)
predictor   = TimeSeriesPredictor(target='target', prediction_length=limit, eval_metric='MAPE', time_limit=12) 
predictor.fit(data_train, presets='high_quality', time_limit=600)  
predictor.fit_summary()


#######################################################################################
#
#    FORECASTING (FROM TRAINING DATA)
#
#######################################################################################
predicted_weigh = predictor.predict(data_train, model='WeightedEnsemble').reset_index()  
predicted_weigh = predicted_weigh[['item_id', 'timestamp', 'mean']]
merged              = pd.merge(predicted_weigh, valid, on=['item_id', 'timestamp'], how='left' )
print(merged)


#######################################################################################
#
#    FORECAST AFTER CORRECTION
#
#######################################################################################
difference_mena = merged['mean'][0] - merged['target'][0]
merged['mean']   = merged['mean'] - int(difference_mena)


#######################################################################################
#
#    DETERMINING STANDARD DEVIATIONS OF THE FORECAST AND PLOTTING THE RESULTS
#
#######################################################################################
inverted   = int(60) 
period      = int(inverted/2)
multiplier  = int(period/3)

forecast_upper_std, forecast_lower_std = f.upper_and_lower_standard_deviation(merged['mean'], period, multiplier)
merged['forecast_upper_std'] = forecast_upper_std 
merged['forecast_lower_std'] = forecast_lower_std  

plt.plot(merged['target'], label = 'BTC', color = 'black', marker = 'o', markerfacecolor = 'b')
plt.plot(merged['mean'], label = 'forecast', color = 'blue')
plt.plot(merged['forecast_upper_std'], label = 'forecast_upper_std', color = 'red')
plt.plot(merged['forecast_lower_std'], label = 'forecast_lower_std', color = 'green')
plt.ylabel('Price')
plt.xlabel("Time slots from the validation period")
plt.legend()
plt.show()

del merged['forecast_upper_std']
del merged['forecast_lower_std']


#######################################################################################
#
#    CALCULATION OF PERCENT FORECAST ERROR
#
#######################################################################################
merged['%_difference_mena'] = ((merged['mean'] / merged['target']) - int(1)) * 100
print(merged)


#######################################################################################
#
#    REVERSATION OF THE FORECAST RESULTS
#
#######################################################################################
merged['reverse_mean'] = merged['mean'][::-1].reset_index(drop=True)
print(merged)


#######################################################################################
#
#    TRAINING (INVERTED FORECAST RESULTS)
#
#######################################################################################
limit_inverted = inverted     #  limit for the inverted forecast
train = copy.deepcopy(merged[ :-limit_inverted])
train.drop(['mean', 'target', '%_difference_mena'], axis=1, inplace=True)
print(train)

valid = copy.deepcopy(merged[-limit_inverted: ])
valid.drop(['mean', 'target', '%_difference_mena'], axis=1, inplace=True)
print(valid)

data = TimeSeriesDataFrame(train)
data['target'] = data['reverse_mean']
del data['reverse_mean']

predictor = TimeSeriesPredictor(target='target', 
                                                prediction_length=limit_inverted, eval_metric='MAPE', 
                                                time_limit=12) 

predictor.fit(data, presets='high_quality', time_limit=300)  
predictor.fit_summary()


#######################################################################################
#
#    FORECASTING (INVERTED FORECAST RESULTS) 
#    FORECAST OF THE PAST BASED ON THE FORECAST OF THE FUTURE
#
#######################################################################################
predicted_reverse_mean = predictor.predict(data, model='WeightedEnsemble').reset_index()  
predicted_reverse_mean = predicted_reverse_mean[['item_id', 'timestamp', 'mean']]
predicted_reverse_mean = predicted_reverse_mean.rename(columns = {'mean':'forecast_reverse_mean'})

merged_2 = pd.merge(merged, predicted_reverse_mean, on=['item_id', 'timestamp'], how='left' )
print(merged_2[-limit_inverted: ])

################################################
tuning_parameter =  3.0  # 0.75

difference_forecasts = merged_2['mean'] - merged_2['forecast_reverse_mean']
merged_2['difference_forecasts'] = difference_forecasts
upper  = merged_2['mean'] + (merged_2['difference_forecasts'] * tuning_parameter)  
lower  = merged_2['forecast_reverse_mean'] - (merged_2['difference_forecasts'] * tuning_parameter)  
merged_2['upper'] = upper
merged_2['lower'] = lower

################################################
period     = int(limit_inverted/1)
multiplier = 3.0

upper_std, lower_std   =  f.upper_and_lower_standard_deviation(data_long_first['Long'], period, multiplier)
upper_std_rest_index  =  upper_std[-limit_inverted:].reset_index()
lower_std_rest_index   =  lower_std[-limit_inverted:].reset_index()

merged_2['upper_std'] = np.nan
merged_2['upper_std'][-limit_inverted:]=upper_std_rest_index['Long']
merged_2['lower_std'] = np.nan
merged_2['lower_std'][-limit_inverted:]=lower_std_rest_index['Long']

print(merged_2[-limit_inverted: ])


#######################################################################################
#
#    PRELIMINARY VERIFICATION OF THE CORRECTNESS OF THE FORECASTS 
#    ACCORDING TO THE INDICATORS: LOWER AND UP
#
#######################################################################################
dif_lower = merged_2['lower'][-limit_inverted : -10] - merged_2['target'][-limit_inverted : -limit_inverted+10]
dif_upp   = merged_2['upper'][-limit_inverted : -10] - merged_2['target'][-limit_inverted : -limit_inverted+10] 

dif_lower_abs  = abs(dif_lower.mean())
dif_upper_abs = abs(dif_upp .mean())


#######################################################################################
#
#    AN ATTEMPT TO DRAW CONCLUSIONS
#
#######################################################################################
if dif_lower_abs < dif_upper_abs:
    clue = str("The  --  lower --  curve should forecast with less error")
    print(clue)


if dif_upper_abs < dif_lower_abs:
    clue = "The  --  upper --  curve should forecast with less error"
    print(clue)


print("During consolidation - sideways trend: After the intersection of the curves, the clue can reverse")


#######################################################################################
#
#    GRAPH OF STANDARD DEVIATIONS AND INDICATORS: LOWER AND UPPER
#
#######################################################################################
plt.plot(merged_2['target'][-limit_inverted:], label = 'BTC', color = 'black', marker = 'o', markerfacecolor = 'b')
plt.plot(merged_2['mean'][-limit_inverted:], label = 'forecast', color = 'blue')
plt.plot(merged_2['forecast_reverse_mean'][-limit_inverted:], label = 'forecast_reverse_mean')
#
plt.plot(merged_2['upper'][-limit_inverted:], label = 'upper', color = 'red')
plt.plot(merged_2['lower'][-limit_inverted:], label = 'lower', color = 'green')
plt.plot(merged_2['upper_std'][-limit_inverted:], label = 'upper_STD')
plt.plot(merged_2['lower_std'][-limit_inverted: ] , label = 'lower_STD')
#
plt.ylabel('Price')
plt.xlabel("Recent time slots from the validation period \n" f'{clue}')
plt.legend()
plt.show()

print("  ****   END   ******  ")
# END