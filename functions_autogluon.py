#!/usr/bin/env python3.8
#
import pandas as pd
from datetime import datetime
import copy
import requests    

BASE_URL = "https://api.gateio.ws"
CONTEX = "/api/v4"
HEADERS = {'Accept': 'application/json', 'Content-Type': 'application/json'}
URL = '/spot/candlesticks'
# SHORT = 'currency_pair=BTC5S_USDT&interval=5m&limit=1000'
LONG = 'currency_pair=BTC_USDT&interval=5m&limit=1000'


class functions:
    def __init__(self, BASE_URL, CONTEX, HEADERS, URL, LONG):
        self.BASE_URL = BASE_URL
        self.CONTEX = CONTEX
        self.HEADERS = HEADERS
        self.URL = URL,
        # self.SHORT = SHORT
        self.LONG = LONG
        
    @staticmethod
    def loading(BASE_URL, CONTEX, HEADERS, URL, LONG):
        # s = requests.request('GET', BASE_URL + CONTEX + URL + "?" + SHORT, headers=HEADERS)
        # s = s.json()
        l = requests.request('GET', BASE_URL + CONTEX + URL + "?" + LONG, headers=HEADERS)
        l = l.json()
        # return [s, l]
        return l
        
    @staticmethod
    def close_gate(r):
        close_long =list()
        index_time_long = list()
        for kline in r:
            close_long.append(float(kline[2])) 
            time_long = datetime.fromtimestamp(int(kline[0])).strftime("%Y-%m-%d %H:%M:%S")   
            index_time_long.append(time_long)   #  time as  int()
        return [close_long,  index_time_long]
        
    @staticmethod
    def data_frame(series, time, columns):
        i=0
        ts = list()
        for t in time:
            ts.append(t)
            ts.append(series[i])
            i += 1
        close = pd.Series(series)
        close.index = pd.to_datetime(time)    
        data_f = pd.DataFrame (close, columns = [columns])
        return data_f
        
    @staticmethod
    def upper_and_lower_standard_deviation(data, period, multiplier):
        upper_std = data.rolling(period).mean()+data.rolling(period).std()*multiplier
        lower_std = data.rolling(period).mean()-data.rolling(period).std()*multiplier
        return [upper_std, lower_std]
# END