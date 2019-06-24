import multiprocessing as mp
import os, sys
import utils
import datedelta
from copy import copy
from DataManager import DataManager
from datetime import date

dm = DataManager()

def target(pairs_list, DATE=None):
        '''
        Target function used to retrieve monthly trading volume (MTV) for all 
        pairs in a list. 

        Args: 
            pairs_list: list of currency-pairs
        
        Returns:
            vol: list of tuples (pair, MTV) 
        ''' 
        global dm
        mtv_list = []
        for pair in pairs_list: 
            tvol, _ = dm.get_monthly_trading_volume(pair, DATE)
            mtv_list.append(tuple((pair, tvol)))

        return mtv_list

def get_best_pairs(PORTFOLIO_SIZE=12, N_CORES=8, DATE=None): 
    '''
    Selects the currency-pairs with the largest monthly trade volume
    (MTV) according to the portfolio size. We only consider currencies that 
    have BTC as its quote currency. 

    Args: 
        pairs: list containing strings for currency-pairs
        PORTFOLIO_SIZE: Number of assets to manage in portfolio. Defaults
                        to 12. 
        N_CORES: Number of cores to use. Defaults to the number of cores
                 available on the computer. 

    Returns: 
        best_pairs: dictionary with currency-pairs as keys and corresponding
                    volume and quote volume values. 
        mtv_values: corresponding MTV values for currency-pairs.
    '''
    global dm
    pairs = dm.pairs
    if len(pairs) < PORTFOLIO_SIZE: 
        string1 = 'DataPreselector: Portfolio size greater than number of currency-pairs.'
        string2 = "Initialise with smaller portfolio size"
        raise ValueError(string1 + string2)

    btc_pairs = [[] for _ in range(N_CORES)]
    for i, pair in enumerate(pairs):
        lhs, _ = pair.split("_")
        n = i % N_CORES
        if lhs == "BTC":
            btc_pairs[n].append(pair)

    # create pool of processes
    pool = mp.Pool(processes=N_CORES)
    result = [pool.apply_async(target, (pair, DATE)) for pair in btc_pairs]
    output = [x for p in result for x in p.get()]

    # sort in ascending order according to monthly trade volumes
    output.sort(reverse=True, key=lambda x: x[1])
    # extract top pairs that have highest monthly trading volume according
    # to portfolio size
    final_list = list(zip(*output[0:PORTFOLIO_SIZE]))
    best_pairs = list(final_list[0])
    mtv_values = list(final_list[1])
    return best_pairs, mtv_values

def asset_preselection(PORTFOLIO_SIZE, DATE, N_CORES=None):
    if N_CORES is None: 
        N_CORES = mp.cpu_count() 
    best_pairs, mtvlist = get_best_pairs(PORTFOLIO_SIZE, N_CORES, DATE)
    return best_pairs, mtvlist

def deprecated_asset_preselection(PORTFOLIO_SIZE, N_CORES): 
    best_pairs, mtvlist = get_best_pairs(PORTFOLIO_SIZE, N_CORES)
    return best_pairs, mtvlist


def fill_nan(data, timestamps, method = 'pad'):
    '''
    Fill nan values in data with fake data. 
    ''' 
    function = {
        'pad' : _padding, 
        'zero_pad' : _zero_padding, 
    }
    fill_function = function[method]
    return fill_function(data, timestamps)

    
def _padding(data, timestamps): 
    val = data[0].copy()
    ldiff = len(timestamps) - len(data)
    for i in range(1,ldiff+1): 
        val['date'] = timestamps[ldiff-i]
        data = [val] + data
    return data

def _zero_padding(data, timestamps): 
    pass

# ------------------ Tests ----------------------
def test():
    DATE = None
    bp1, mtv1 = asset_preselection(5, 8, None)

    print('Best pairs: {}'.format(bp1))
    print('monthly volum: {}'.format(mtv1))

    DATE = '1/1/2019 00:00:00'
    bp2, mtv2 = asset_preselection(5,8,DATE)

    print('Best pairs: {}'.format(bp2))
    print('monthly volum: {}'.format(mtv2))

if __name__=='__main__':
    test()