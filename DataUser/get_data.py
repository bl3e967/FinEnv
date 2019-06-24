import multiprocessing as mp
import logging
import DataPreprocess
import os, sys
import datetime
import datedelta
from time import time
from utils import RelativeDeltaWrapper
from dateutil.relativedelta import relativedelta
from DataManager import DataManager
from pathlib import Path

DM = DataManager()
logger = mp.log_to_stderr()
logger.setLevel(logging.DEBUG)
lock = mp.Lock()

def get_data(arg_dict):
    '''
    Args: 
        arg_dict: Dictionary of arguments for get_chart_data. 
    '''
    global DM
    pair  = arg_dict['pair'];  period = arg_dict['period']
    start = arg_dict['start']; end = arg_dict['end']

    logger.info('getting data for {}'.format(pair))
    
    data = DM.get_chart_data(pair, start, end, period)

    # insert name of pair for the data. 
    data.insert(0, pair)
    return data

def target(dates, directory):
    '''
    Creates a directory ./data in the current directory filled with directories containing
    data according to the starting month from which the data is taken. 
    Args: 
        dates: List containing dates at which to retrieve data from Poloniex database. Assume
        period is 1800. 
    
    Returns: 
        0:
    '''
    logger.info("Target Running")

    # Create feed dict for get_data() function.
    feed_dict_list = []
    for end in dates: 
        for pair in DM.btc_pairs: 
            start = end - datedelta.MONTH
            logger.debug("Start month: {} || End month: {}".format(start, end))
            feed_dict = {
                'pair'  : pair, 
                'start' : start.strftime(DM.dateformat), 
                'end'   : end.strftime(DM.dateformat), 
                'period': 1800,
            }
            feed_dict_list.append(feed_dict)
    
    # get data and save
    for args in feed_dict_list: 
        data = get_data(args)
        date, _ = args['start'].split(' ')
        p = directory+"/{}/".format(date.replace('/','_'))
        savedir = Path(p)
        # Create target directory & all intermediate directories if don't exists
        with lock:
            logger.debug("Saving to {}".format(savedir))
            if not os.path.exists(savedir):
                os.makedirs(savedir)
                print("Directory " , savedir ,  " Created ")
            filename = savedir.joinpath(args['pair'] + '.pickle')
            DM.save2pickle(filename, data)
            
def test():
    n_cores = mp.cpu_count()
    pool = mp.Pool(n_cores)
    
    feed_dict_list = []
    for pair in DM.btc_pairs:
        feed_dict = {
            'pair': pair, 
            'start': '1/2/2017 00:00:00',
            'end':   '1/2/2017 23:59:59',
            'period': 1800,
        }
        feed_dict_list.append(feed_dict)
    f = time()
    logger.info("Creating processes")
    result = [pool.apply_async(target, (args,)) for args in feed_dict_list]
    output = [p.get() for p in result]  # list of lists
    elapsed = time() - f
    print("Time elapsed: {}".format(elapsed))
    input("Press Enter to continue...")

def test_target(): 
    N_CORES = 1

    #---collect data by month, starting from now to 1st of January 2014---
    timediff = RelativeDeltaWrapper()
    beginning = datetime.date(2014,1,1); end = DM._today.replace(day=1)
    n_months = timediff.diffInMonths(end, beginning)

    N = int(n_months / N_CORES) # number of months per list

    datelist=[]
    for i in range(N_CORES): 
        months = [(end - i*N*datedelta.MONTH - n*datedelta.MONTH) for n in range(N)]
        datelist.append(months)

    savedir = './data'
    logger.info("Creating processes")
    target(datelist[0], savedir)

    input("press enter to continue...")


def run_multiprocess(): 
    # need to get data from serve from now up to when the currency
    # first started trading. 
    N_CORES = mp.cpu_count()

    #---collect data by month, starting from now to 1st of January 2014---
    timediff = RelativeDeltaWrapper()
    beginning = datetime.date(2014,1,1); end = DM._today.replace(day=1)
    n_months = timediff.diffInMonths(end, beginning)

    N = int(n_months / N_CORES) # number of months per list

    datelist=[]
    for i in range(N_CORES): 
        months = [(end - i*N*datedelta.MONTH - n*datedelta.MONTH) for n in range(N)]
        datelist.append(months)

    savedir = './data'
    logger.info("Creating processes")
    pool = mp.Pool(N_CORES)
    result = [pool.apply_async(target, (args, savedir)) for args in datelist]
    output = [p.get() for p in result]

    input("press enter to continue...")

if __name__=="__main__":
    run_multiprocess()