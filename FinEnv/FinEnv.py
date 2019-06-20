import utils
from DataPreprocess import asset_preselection, fill_nan
from DataManager import DataLoader
from DataPlotter import DataPlotter

class FinEnv(): 
    def __init__(self, m, n, f, date1, date2, interval = 1800):
        '''
        Args: 
            m: number of preselected non-cash assets
            n: number of input periods before t
            f: feature number
            date1: string start date and time of market 'dd/mm/yyyy hh:mm:ss'
            date2: string end date and time of market 'dd/mm/yyyy hh:mm:ss'
            interval: time interval between periods. 300, 900, 1800, 
                      7200, 14400, 86400

        Refer to: https://arxiv.org/pdf/1706.10059.pdf section 3.2 Price Tensor. 
        m = 11, n = 50, f=3 was used in the paper. 
        '''
        m = int(m); n = int(n); f = int(f)
        
        # time
        self.interval = interval
        self.timestamp = _initialise_timestamp(date1, date2, interval)

        # best pair asset preselection
        self.pairs, self.pairs_mtv = asset_preselection(m, 8, date2)

        # data preprocessing
        self._v, self._vhi, self._vlow = (     
            _initialise_data(self.pairs, self.timestamp, self.interval) # initialises above four variables
        )
        self._state = []
    
    def step(self, action):
        '''
        args: 
            action: The portfolio vector at time period t, w(t)
        Returns:
            next_state: 
            reward: immediate reward for an individual episode, defined as the logarithmic 
            rate of return for period t, where r(t) = ln(p(t)/p(t-1)) = ln(y(t)w(t-1))
            done: 
            info:  
        '''
        pass 

    def reset(self): 
        '''
        Returns: 
            state: State consists of two parts, the external state represented by the
            price tensor, X(t), and the internal state represented by the portfolio 
            vector from the last period w(t-1): S(t) = (X(t), w(t-1))
        '''
        pass 

    def render(self): 
        pass 

def _initialise_timestamp(date1, date2, interval): 
    t1 = int(utils.date2unix(date1)); t2  = int(utils.date2unix(date2))
    timestamps = list([t for t in range(t1,t2+interval,int(interval))])
    return timestamps

def _initialise_data(pairs, timestamps, interval):
        '''
        Args:
            pairs: 
            timestamps: list
            interval: int 

        Returns: 


        The data for each pair has timestamp data which is redundant. 
        Move the timestamp data to a separate variable self.timestamps, 
        such that the env timesteps can be done in intervals from t = 1 to 
        t = tf, where tf = number of timestamps. 
        We can translate from the time interval index to the timestamp by indexing
        the timstamp variable. 
        '''
        dl = DataLoader()
        data_dict = {}

        for pair in pairs: 
            data = dl.import_data_btwn(date1, date2, pair)
            data = fill_nan(data, timestamps)
            _run_timestamp_checks(data, timestamps)
            data_dict[pair] = data 
        # generate price vectors
        
        return 0,0,0

def _run_timestamp_checks(data, timestamps): 
    '''
    Check for timestamp consistency. 
    ''' 
    if timestamps[0] != data[0]['date']: 
        raise Exception("FinEnv Initialisation: First data timestamps do not match")
    if timestamps[-1] != data[-1]['date']: 
        raise Exception("FinEnv Initialisation: Last data timestamps do not match")
    if len(timestamps) != len(data): 
        raise Exception("FinEnv Initialisation: Check timestamp length")

'''
------------------------NOT INCLUDING TRANSACTION COST---------------------------
- price vector for period t, v(t): the closing prices of all assets. 
    - v(t)[i] = vi is the ith asset's closing price from the previous period. 
    - vhi(t): vector containing highest price of the period for each asset. 
    - vlow(t): vector containing lowest price of the period for each asset. 
    - v[0] == vhi[0] == vlow[0] = 1 which is the quote currency, BTC. 

- price relative vector y(t) = v(t) ./ v(t-1) (element-wise)

- w(t): Portfolio vector, weighting of asset allocation for each asset. sums to 1. 
    - w(0) = [1 0 ... 0]

- p(t-1): Portfolio value at the beginning of period t. Then, 
  p(t) = p(t-1)*y(t)*w(t-1)
    - if no transaction cost, then final portfolio value is: 
      pf = p(0)exp(sum(r(t))) from t = 1 to (tf + 1). 

- rate of return rho(t) = p(t)/p(t-1) -1 = y(t)*w(t-1) - 1
- log rate of return r(t) = ln(p(t)/p(t-1)) = ln(y(t)*w(t-1))

-----------------------INCLUDING TRANSACTION COSST----------------------------
At the end of a period, weights evolve into: 
w'(t) = y(t).*w(t-1)/(y(t)*w(t-1))

Now need to reallocate portfolio vector from w'(t) to w(t) by buying and selling. 
Paying all commission fees leads to shrinking portfolio value by factor mu(t). 
If for asset i, p(t)[i]'w(t)[i]' > p(t)[i]w(t)[i] or w(t)[i] > mu(t)w(t)[i], then 
some or all of this asset needs to be sold. 
    - The amount of cash obtained by selling is: 
        (1-cs)p(t)'sum(relu(w(t)[i] - mu(t)w(t)[i])) for i = 1 to i = m. 
        where cs is the commission rate. 
    
    - The money earned and the cash reserve left after the transaction is used to
    buy new assets. 
    (1-cp)[ w(t)[0]' + (1-cs)sum(relu(w(t)[i]'-mu(t)w(t)[0])) - mu(t)w(t)[0] ]

        = sum(mu(t)w(t)[i] - w(t)[i]') for i = 1 to m          (13)

    where cp is the commission rate for purchasing. As relu(a-b) - relu(b-a) = a-b, 
    and w(t)[0]' + sum_i( w(t)[i]' ) = 1 = w(t)[0] + sum_i( w(t)[i] ), (13) becomes

    mu(t) = 1/( 1-cp*w(t)[0] )*[ relu(1 - cp*w(t)[0]' - (cs + cp -cs*cp)sum_i( w(t)[i] - mu(t)w(t)[i])) ]
      (14)
    
    As (14) has mu contained in a relu, cannot analytically calculate mu. Need to
    iteratively solve for mu. Gamma-contraction mapping, so convergence is ensured. 
    Initial guess of mu_init = c*sum_i( |w(t)[i]' - w(t)[i]| ) when cp = cs = c. 

    cp = cs = 0.25% used, which is the maximum commission rate at Poloniex. 

such that the portfolio value becomes: 
    p(t) = mu(t) * p(t)'

rate of return: 
    rho(t) = p(t)/p(t-1) - 1 = mu(t)*y(t)*w(t-1) -1 
    r(t) = log(p(t)/p(t-1)) = ln(mu(t)*y(t)*w(t-1))

final portfolio value: 
    pf = p0*exp(sum(r(t))) from t=1 to t=(tf+1)
       = po * product(mu(t)y(t)*w(t-1)) from t=1 to t=(tf+1)
'''
if __name__=="__main__": 
    date1 = '1/1/2016 00:00:00'
    date2 = '1/1/2019 00:00:00'
    env = FinEnv(10,1,1,date1, date2)
    print(env.pairs)
    print(env.pairs_mtv)