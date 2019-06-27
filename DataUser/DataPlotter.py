import os, warnings
import json, pickle
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import datedelta
from datetime import datetime
from mpl_finance import candlestick_ohlc
from poloniex import Poloniex
from DataUser.DataManager import DataManager
from datetime import date
from pathlib import Path

class DataPlotter(DataManager):
    def __init__(self):
        super().__init__()
        self.data = []

    def import_plot_data(self, date, pair, ftype='.pickle', datadir='./data', ReturnData=False): 
        '''
        Import data for the specified currency-pair and date from local data storage. 
        Args: 
            date: string or datetime object. If string, must be in the format 'dd/mm/yyyy 00:00:00'
            pair: string specifying name of currency-pair to load
            type: string specifying data file type. Either '.pickle' or '.json'
        '''
        if type(date) == type(date.today()): 
            date, _ = date.strftime(self.dateformat).split(' ')
            date = date.replace('/','_')
        
        filename = pair + ftype
        path = Path(datadir).joinpath(date, filename).resolve()
        print('Loading data from {}'.format(path))
        new_data = self.load_pickle(path)
        new_data.pop(0) # get rid of label at the start of data. 
        new_data[:] = [d for d in new_data if d.get('date') != 0]
        self.data = self.data + new_data

    def import_plot_data_btwn(self, date1, date2, pair, ftype='.pickle', datadir='./data'):
        '''
        Imports data between the months specified between date1 and date2.
        If dates are strings, should be in format 'dd/mm/yyyy'. 
        Args:
            date1: preceding date. Either string or datetime object.
            date2: later date. Either string or datetime object.
            pair: currency-pair. 
            ftype: '.pickle' or '.json'
            datadir: local data directory
        '''
        warnings.warn('If import data has been used, data will likely have been duplicated.')
        assert date1 < date2, ValueError("date1 has to precede date 2")
        # deal with string type. 
        convert_to_datetime = lambda d : datetime.strptime(d, self.dateformat)

        if isinstance(date1, str): date1 = convert_to_datetime(date1)
        if isinstance(date2, str): date2 = convert_to_datetime(date2)
        
        d = date1; one_month = datedelta.MONTH
        while d < date2: 
            self.import_plot_data(d, pair, ftype=ftype, datadir=datadir)
            d += one_month
        
        if not self.data: 
            raise ValueError("No valid data retrieved for {} between {} to {}".format(
                pair, date1.strftime(self.dateformat), date2.strftime(self.dateformat))
                )
        
    def plot_candlestick(self): 
        '''
        Generates candlestick plot of imported data. 
        '''
        if not self.data: 
            raise ValueError("Need to import data before plotting")

        df = pd.DataFrame.from_dict(self.data, orient="columns")
        df['date'] = pd.to_datetime(df['date'], unit='s')
        df["date"] = df["date"].apply(mdates.date2num)  
        ohlc = df[['date','open','high','low','close']].copy()
        _, ax = plt.subplots(figsize = (10,5))

        # plot the candlesticks
        print('generating candlestick plot')
        candlestick_ohlc(ax, ohlc.values, width=.6, colorup='green', colordown='red')
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))    

        print('plotting...')
        plt.grid()
        plt.show() 
    
    def plot_candlestick_with_ma(self): 
        raise Warning("Not implemented yet")

def test_data_plotter(): 
    plotter = DataPlotter()
    pair = 'BTC_BCN'
    date1 = date(2016,1,1)
    date2 = date(2019,6,1)
    plotter.import_plot_data_btwn(date1, date2, pair, datadir='./data')
    plotter.plot_candlestick()

# if __name__ == "__main__": 
#     test_data_plotter()
#     input('Press enter to continue...')
