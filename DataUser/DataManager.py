import os, warnings
import json, pickle
import datedelta
import pandas as pd
import matplotlib.pyplot as plt
import multiprocessing as mp
from utils import utils
from datetime import date, datetime
from pathlib import Path
from poloniex import Poloniex

class DataManager():
    def __init__(self): 
        self.polo = Poloniex()
        self.currencies = self._get_currencies()
        self.pairs = self._get_pairs()
        self.btc_pairs = self._get_btc_pairs()
        
        # dates
        self.dateformat = "%d/%m/%Y %H:%M:%S"
        self._today  = date.today()  # datetime object
        self._prev_month = self._today - datedelta.MONTH  # datetime object 
        self.today = self._today.strftime(self.dateformat)   # string
        self.prev_month = self._prev_month.strftime(self.dateformat)  # string

    def check_date_format(self):
        return self.dateformat

    def _get_currencies(self):
        '''
        Returns list of currency available in Poloniex
        '''
        currency_dict = self.polo.returnCurrencies()
        currencies = list(currency_dict.keys())
        return currencies
    
    def _get_pairs(self):
        '''
        Returns list of currency-pair available in Poloniex
        '''
        ticker_dict = self.polo.returnTicker()
        pairs = list(ticker_dict.keys())
        return pairs
    
    def _get_btc_pairs(self): 
        btc_pairs = []
        for pair in self.pairs: 
            lhs, _ = pair.split("_")
            if lhs == "BTC": 
                btc_pairs.append(pair)
        return btc_pairs

    def get_chart_data(self, pair, start=None, end=None, period=None):
        '''
        Returns candlestick chart data. 
        Args: 
            pair: Currency Pair
            start: string specifying start time in the format "dd/mm/yyyy hh:mm:ss"
            end: string specifying end time in the format "dd/mm/yyyy hh:mm:ss" 
            period: Candlestick period in seconds; valid values are 300, 900, 1800, 
                    7200, 14400, 86400. 
        Returns: 
            Dictionary containing candlestick chart data 
            date:	The UTC date for this candle in miliseconds since the Unix epoch.
            high:	The highest price for this asset within this candle.
            low:    The lowest price for this asset within this candle.
            open:	The price for this asset at the start of the candle.
            close:	The price for this asset at the end of the candle.
            volume:	The total amount of this asset transacted within this candle.
            quoteVolume: The total amount of base currency transacted for this asset within this candle.
            weightedAverage: The average price paid for this asset within this candle.
        '''
        if period is None: 
            raise ValueError("datamanager: argument 'period' should not be None")

        start = utils.date2unix(start)
        end = utils.date2unix(end)
        data = self.polo.returnChartData(pair, start=start, end=end, period=period)
        return data

    def get_trading_volume(self, pair, start=None, end=None, period=None):
        '''
        Returns volume and quote volume for currency pair in a given time interval for
        the specified time period. 
        Args: 
            pair: Currency pair
            start: string specifying start time in the format "date/month/year"
            end: string specifying end time in the format "date/month/year"
            period: Candlestick period in seconds; valud vlaues are 300, 900, 1800, 
                    7200, 14400, 86400.
        
        Returns: 
            vol: trade volume 
            qvol: quote volume
        '''
        start = utils.date2unix(start)
        end = utils.date2unix(end)
        data = self.polo.returnChartData(pair, start=start, end=end, period=period)
        vol = data[0]['volume']
        qvol = data[0]['quoteVolume']
        return vol, qvol

    def get_monthly_trading_volume(self, pair, date=None): 
        '''
        Get monthly trading volume for specified currency-pair starting from
        date to one month in the past. If date is None, then evaluates assets based on
        trading volume from TODAY to one month in the past. 

        Args: 
            pair: Currency pair
            date: datetime object or string. If string, must be in the form 
            'dd/mm/yyyy hh:mm:ss'.  

        Returns: 
            tvol: Monthly volume for specified currency-pair
            tqvol: Monthly quoted volume for specified currency-pair
        '''
        if isinstance(date, type(self._today)): # check if datetime object
            end = date.strftime(self.dateformat)
            prev_month = (date - datedelta.MONTH).strftime(self.dateformat)
        elif isinstance(date, str): # check if string
            end = date 
            datetimeobj = datetime.strptime(date, self.dateformat) - datedelta.MONTH
            prev_month = datetimeobj.strftime(self.dateformat) # convert to string 
        elif date is None:             
            prev_month = self.prev_month # needs to be string
            end = self.today    # needs to be string
        else: 
            raise ValueError('UNKNOWN ERROR in get_monthly_trading_volume()')

        period24hr = 86400
        data = self.get_chart_data(
                pair, 
                start=prev_month, 
                end=end, 
                period=period24hr
                )
        tvol = sum([item['volume'] for item in data])
        tqvol = sum([item['quoteVolume'] for item in data])

        return tvol, tqvol

    def save2json(self, filename, data):
        '''
        Save data to json file called 'filename'
        
        Args: 
            filename: destination file in json format
            data: dictionary of data
        '''
        with open(filename, 'w') as fp: 
            json.dump(data, fp)
            print("Data saved to {}".format(filename))
    
    def load_json(self, filename): 
        '''
        Loads dictionary object from json file called 'filename'
        Args: 
            filename: 
        Returns: 
            data: Dictionary object
        '''
        with open(filename) as json_file: data = json.load(json_file)
        return data

    def save2pickle(self, filename, data): 
        '''
        Pickles data to file called 'filename'
        Args: 
            filename: destination file
            data: AutoCastDict object created by poloniex. Can be any python object
        '''
        def _convert2dict(data): 
            '''
            convert list of AutoCastDict object to list of dictionaries.
            Args:  
                data: List of AutoCastDicts
            '''
            if isinstance(data[0], str): i = 1
            else: i = 0
            data = [dict(entry) for entry in data[i:]]
            return data
        
        data = _convert2dict(data)
        with open(filename, 'wb') as fp: 
            pickle.dump(data, fp)
    
    def load_pickle(self, filename):
        '''
        Load pickled data object from file called 'filename'. 
        Args:
            filename: target file called 'filename'
        Returns: 
            data: AutoCastDict object created by poloniex. 
        '''
        with open(filename, 'rb') as fp: 
            data = pickle.load(fp)        
        return data

    def import_data(self, date, pair, ftype='.pickle', datadir='./data', ReturnData=False): 
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
        # print('Loading data from {}'.format(path))
        data = self.load_pickle(path)
        data.pop(0) # get rid of label at the start of data. 
        data[:] = [d for d in data if d.get('date') != 0]
        return data

    def import_data_btwn(self, date1, date2, pair, ftype='.pickle', datadir='./data'):
        '''
        Imports data between the months specified between date1 and date2.
        If dates are strings, should be in format 'dd/mm/yyyy hh:mm:ss'. 
        Args:
            date1: preceding date. Either string or datetime object.
            date2: later date. Either string or datetime object.
            pair: currency-pair. 
            ftype: '.pickle' or '.json'
            datadir: local data directory
        '''
        
        # deal with string type. 
        convert_to_datetime = lambda d : datetime.strptime(d, self.dateformat)
        if isinstance(date1, str): date1 = convert_to_datetime(date1)
        if isinstance(date2, str): date2 = convert_to_datetime(date2)
            
        assert date1 < date2, ValueError("date1 has to precede date 2")
        data = []
        d = date1; one_month = datedelta.MONTH
        while d < date2: 
            new_data = self.import_data(d, pair, ftype=ftype, datadir=datadir)
            data = data[0:-1] + new_data
            d += one_month
        
        return data

class DataLoader(DataManager): 
    def __init__(self): 
        super().__init__()
    
    def import_data(self, date, pair, ftype='.pickle', datadir='./data', ReturnData=False): 
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
        # print('Loading data from {}'.format(path))
        data = self.load_pickle(path)
        data.pop(0) # get rid of label at the start of data. 
        data[:] = [d for d in data if d.get('date') != 0]
        return data

    def import_data_btwn(self, date1, date2, pair, ftype='.pickle', datadir='./data'):
        '''
        Imports data between the months specified between date1 and date2.
        If dates are strings, should be in format 'dd/mm/yyyy hh:mm:ss'. 
        Args:
            date1: preceding date. Either string or datetime object.
            date2: later date. Either string or datetime object.
            pair: currency-pair. 
            ftype: '.pickle' or '.json'
            datadir: local data directory
        '''
        assert date1 < date2, ValueError("date1 has to precede date 2")
        # deal with string type. 
        convert_to_datetime = lambda d : datetime.strptime(d, self.dateformat)

        if isinstance(date1, str): date1 = convert_to_datetime(date1)
        if isinstance(date2, str): date2 = convert_to_datetime(date2)
        
        data = []
        d = date1; one_month = datedelta.MONTH
        while d < date2: 
            new_data = self.import_data(d, pair, ftype=ftype, datadir=datadir)
            data = data[0:-1] + new_data
            d += one_month
        
        return data
#-----------------------------------------------------------------------
# Test functions

def test_save_load(): 
    datamanager = DataManager()
    start = "1/2/2018 00:00:00"
    end = "1/3/2018 00:00:00"
    period=86400
    pair = datamanager.pairs[0]
    data = datamanager.get_chart_data(pair, start=start, end=end, period=period)
    filename = "test_data.pickle"
    datamanager.save2pickle(filename, data)
    loaded_data = datamanager.load_pickle(filename)
    
    try: 
        if data == loaded_data: 
            print("Data saving to pickle good")
        else: 
            raise ValueError("Saved data and loaded data not same")
    except ValueError as ve: 
        print(ve)
        os._exit(-1)

def test_get_trading_vol():
    datamanager = DataManager()
    start = "1/2/2018 00:00:00"
    end = "1/2/2018 12:59:59"
    period=86400
    pair = datamanager.pairs[0]
    _ = datamanager.get_chart_data(pair=pair,start=start,end=end,period=period)
    vol, qvol = datamanager.get_trading_volume(pair, start=start, end=end, period=period)
    print("Trading volume for {}:".format(pair))
    print("Volume: {}".format(vol))
    print("Quote Volume: {}".format(qvol))

def test_data_plot(): 
    datamanager = DataManager()
    start = "1/2/2018 00:00:00"
    end = "1/3/2018 00:00:00"
    period=300
    pair = datamanager.pairs[0]
    data = datamanager.get_chart_data(pair, start=start, end=end, period=period)
    df = pd.DataFrame.from_dict(data, orient="columns")
    df.plot("date", "high")
    df.plot("date", "low")
    plt.hold(True)
    plt.show()

def check_volume_data(): 
    '''
    Check that sum of 5-minute volume data is equal to one day worth of 24-hour volume data
    '''
    datamanager = DataManager()
    start = "1/1/2018 00:00:00"
    end = "1/1/2018 23:59:59"
    tol = 0.0001
    pair = datamanager.pairs[0]
    data_300 = datamanager.get_chart_data(pair, start=start, end=end, period=300)
    data_86400 = datamanager.get_chart_data(pair, start=start, end=end, period=86400)

    tvol, tqvol = 0, 0
    for i in range(len(data_300)): 
        tvol += data_300[i]['volume']
        tqvol += data_300[i]['quoteVolume']
    
    if (tvol - data_86400[0]['volume']) < tol:
        print("Test passed")
    else: 
        raise ValueError("5 minute volume data does not match 1-day volume data")
    
    if (tqvol - data_86400[0]['quoteVolume']) < tol:
        print("Test passed")
    else: 
        raise ValueError("5 minute quote volume does not match 1-day quote volume")

def test_monthly_trading_volume():
    dm = DataManager()
    pair = dm.pairs[0]

    v1, qv1 = dm.get_monthly_trading_volume(pair)
    print("Monthly trade volume for {}: {}".format(pair, v1))
    print("Monthly quote trade volume for {}: {}".format(pair, qv1))

    date = dm.today
    v2, qv2 = dm.get_monthly_trading_volume(pair, date)
    print("Monthly trade volume for {}: {}".format(pair, v2))
    print("Monthly quote trade volume for {}: {}".format(pair, qv2))

    date = '1/1/2016 00:00:00'
    v3, qv3 = dm.get_monthly_trading_volume(pair, date)
    print("Monthly trade volume for {}: {}".format(pair, v3))
    print("Monthly quote trade volume for {}: {}".format(pair, qv3))

    date = datetime.strptime(date, dm.dateformat)
    v4, qv4 = dm.get_monthly_trading_volume(pair, date)
    print("Monthly trade volume for {}: {}".format(pair, v4))
    print("Monthly quote trade volume for {}: {}".format(pair, qv4))

    if (v1 != v2) and (qv1 != qv2): raise Exception('Test not passed.')
    if (v3 != v4) and (qv3 != qv4): raise Exception('Test not passed.')
    if (v1 == v3) or (v1==v4): raise Exception('Test not passed.')
    if (qv1 == qv3) or (qv1 == qv4): raise Exception('Test not passed.')

    print('passed')

def test_multiprocessing(): 
    dm = DataManager()
    n_cores = mp.cpu_count()
    pool = mp.Pool(n_cores)
    result = [pool.apply_async(dm.get_chart_data, (pair, "1/1/2018 00:00:00", "1/2/2018 00:00:00")) for pair in dm.pairs]
    output = [x for p in result for x in p.get()]
    print('blah')

def test_import_data_btwn():
    dm = DataLoader()
    date1 = '1/1/2016 00:00:00'
    date2 = '1/1/2018 00:00:00'
    pair = 'BTC_BCN'
    data = dm.import_data_btwn(date1,date2,pair)
    print('Loading data into dataframe')
    df = pd.DataFrame.from_dict(data, orient="columns")
    print('Generating plots...')
    df.plot("date", "high")
    df.plot("date", "low")
    plt.show()


if __name__ == "__main__":
    test_save_load()
    test_get_trading_vol()
    test_monthly_trading_volume()
    test_import_data_btwn()

