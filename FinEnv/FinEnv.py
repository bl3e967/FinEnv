import utils, warnings, sys
import numpy as np
from DataUser.DataManager import DataManager
from DataUser.DataPreprocess import asset_preselection, fill_nan
from DataUser.DataPlotter import DataPlotter

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
        self.DM = FinDataManager(m, n, f, date1, date2, interval)
        self.DM.prev_a = None
    
    def step(self, action):
        '''
        args: 
            action: The portfolio vector at time period t, w(t)
        Returns:
            next_state: Tuple (price tensor, previous action)
            reward: immediate reward for an individual episode, defined as the logarithmic 
            rate of return for period t, where r(t) = ln(p(t)/p(t-1)) = ln(y(t)w(t-1))
            done: 
            info: 
        '''
        if self.DM.prev_a is None: 
            raise Exception('Cannot take step without reset() being called first.')
        
        X = self.DM._pTensor()
        res = tuple((X, self.DM.prev_a))

        # TODO: Fill in reward, done, info
        reward = None 
        done = None 
        info = None

        self.DM.prev_a = action 
        return res, reward, done, info

    def reset(self): 
        '''
        Returns: 
            state: State consists of two parts, the external state represented by the
            price tensor, X(t), and the internal state represented by the portfolio 
            vector from the last period w(t-1): S(t) = (X(t), w(t-1))
        '''
        # create initial weight vector of [1, 0, ..., 0] of size (M,1)
        self.DM.reset_t()
        X = self.DM._pTensor()
        self.DM.prev_a = np.insert(np.zeros((self.DM.M, 1)), 0, 1)
        return tuple((X, self.DM.prev_a)) 

    def render(self): 
        pass 

    def _get_data(self): 
        return self.DM.data

class FinDataManager(DataManager):
    def __init__(self, m, n, f, date1, date2, interval = 1800):
        super().__init__()
        self.M = int(m); self.N = int(n); self.F = int(f)

        self._cs = 0.25 
        self._cp = 0.25 
        self._c  = 0.25

        # time
        self.interval = interval
        self.timestamp = self._initialise_timestamp(date1, date2, interval) 
        # TODO: Currently starts from the second period to account for the need
        # of the previous period for calculating the price relative vector y(t). 
        # May need to fix. 
        self.T_init_val = self.N
        self.T = self.T_init_val # initial timestep

        # best pair asset preselection
        self.pairs, self.pairs_mtv = asset_preselection(m-1, DATE=date1)
        self.pairs.sort()   # sort pairs into alphabetical order

        # data preprocessing
        self.data = (     
            self._initialise_data(date1, date2, self.pairs, self.timestamp, self.interval) # initialises above four variables
        )

    # property decorator for cp
    @property 
    def cp(self): 
        return self._cp
    
    @property
    def cs(self): 
        return self._cs 

    @property
    def c(self): 
        return self._c 

    @cp.setter
    def cp(self, value): 
        if (value > 1) or (value < 0): 
            raise ValueError("Invalid Commission rate: should be between 0 and 1.")
        else: self._cp = value
    
    @cs.setter
    def cs(self, value): 
        if (value > 1) or (value < 0): 
            raise ValueError("Invalid Commission rate: should be between 0 and 1.")
        else: self._cs = value
    
    @c.setter 
    def c(self, value): 
        '''
        c should only be used when cp and cs is identical. Initialised to be the same. 
        If cp and cs is different and c is set a value, then cp and cs will automatically 
        be set to be the same as this value. 
        '''
        if (value > 1) or (value < 0): 
            raise ValueError("Invalid Commission rate: should be between 0 and 1.")
        if (self.cp != self.cs): 
            warnings.warn('cp and cs should be equal if using c. Setting cp and cs to be equal.')
            self.cp = value; self.cs = value 
        self.c = value        

    def _initialise_timestamp(self, date1, date2, interval): 
        t1 = int(utils.date2unix(date1)); t2  = int(utils.date2unix(date2))
        timestamps = list([t for t in range(t1,t2+interval,int(interval))])
        return timestamps

    def _initialise_data(self, date1, date2, pairs, timestamps, interval):
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
            data_dict = {}

            for pair in pairs: 
                data = self.import_data_btwn(date1, date2, pair)
                data = fill_nan(data, timestamps)
                self._run_timestamp_checks(data, timestamps)
                data_dict[pair] = data
            
            return data_dict

    def _run_timestamp_checks(self, data, timestamps): 
        '''
        Check for timestamp consistency. 
        ''' 
        if timestamps[0] != data[0]['date']: 
            raise Exception("FinEnv Initialisation: First data timestamps do not match")
        if timestamps[-1] != data[-1]['date']: 
            raise Exception("FinEnv Initialisation: Last data timestamps do not match")
        if len(timestamps) != len(data): 
            raise Exception("FinEnv Initialisation: Check timestamp length")

    def increment_t(self): 
        self.T += 1

    def reset_t(self): 
        self.T = self.T_init_val

    def _v(self, n=0): 
        '''
        v(t)[i] = vi is the ith asset's closing price from the previous period. 
        Closing price of previous period is opening price of current period. 
        '''
        T = self.T - n
        if T < 0: raise ValueError("_v: n greater than time index T")
        vector = [self.data[pair][T]['open'] for pair in self.pairs]
        vector.insert(0,1) # quote currency BTC
        res = np.expand_dims(np.array(vector), axis=1)
        return res
    
    def _vhi(self, n=0):
        T = self.T - n
        if T < 0: raise ValueError("_v: n greater than time index T")
        vector = [self.data[pair][T]['high'] for pair in self.pairs]
        vector.insert(0,1) # quote currency BTC
        res = np.expand_dims(np.array(vector), axis=1)
        return res

    def _vlo(self, n=0): 
        T = self.T - n
        if T < 0: raise ValueError("_v: n greater than time index T")
        vector = [self.data[pair][T]['low'] for pair in self.pairs]
        vector.insert(0,1) # quote currency BTC
        res = np.expand_dims(np.array(vector), axis=1)
        return res

    def prv(self): 
        '''
        Price relative vector: y(t) = v(t) ./ v(t-1) (element-wise)
        '''
        return np.multiply(self._v(), self._v(n=1))

    def _normPMatrix(self, func): 
        '''
        Args: 
            func: function handle for price vector, _v, _vhi, _vlow. 
        Returns: 
            res: normalised price matrix. 
                For func = v,  V = [v(t-n+1)./v(t) | ... | v(t-1)./v(t) | 1]
                For func = vhi V = [vhi(t-n+1)./v(t) | ... | vhi(t-1)./v(t) | vhi(t)./v(t)]
                For func = vlo V = [vlo(t-n+1)./v(t) | ... | vlo(t-1)./v(t) | vlo(t)./v(t)]
        '''
        v = self._v()
        res = np.divide(func(), v)
        for i in range(1, self.N): 
            v = np.divide(func(n=i), v)
            res = np.concatenate((v, res), axis=1)
        return res

    def _pTensor(self): 
        '''
        Returns: 
            X: price tensor. The stacking of three normalised price matrices. 
        '''
        V = np.expand_dims(self._normPMatrix(self._v), axis=2)
        Vhi = np.expand_dims(self._normPMatrix(self._vhi), axis=2)
        Vlo = np.expand_dims(self._normPMatrix(self._vlo), axis=2)
        X = np.concatenate((V,Vhi,Vlo), axis=2)
        return X

    def _getEvolvedW(self): 
        '''
        The portfolio vector at the beginning of Period t is w(t-1). Due to 
        price movements in the market, at the end of the same period, the weights
        evolve into:    w'(t) = y(t).*w(t-1)/(y(t)*w(t-1)). 

        The mission of the portfolio manager now at the end of period t is to
        reallocate portfolio vector from w'(t) to w(t) by selling and buying 
        relevant assets. 

        Returns: 
            wdash: evolved weight w'(t) at the end of a period. 
        '''
        num = np.multiply(self.prv, self.prev_a) # vector
        den = np.dot(self.prv, self.prev_a)      # scalar
        return num/den

    def getTRF(self, w, delta = 1e-8): 
        '''
        Calculates transaction remainder factor (TRF) 
        mu(t) = 1/( 1-cp*w(t)[0] )*[ 1 - cp*w(t)[0]' - (cs + cp -cs*cp)sum_i( relu(w(t)[i] - mu(t)w(t)[i]) )) ]
        Args: 
            w: numpy array of size (m+1,1) representing portfolio 
            weight vector w(t)
            delta: error tolerance
        '''
        wdash = self._getEvolvedW()
        error = sys.float_info.max
        K = self.cs + self.cp - self.cs*self.cp
        # quote currency weight values
        q = w[0]; qdash = wdash[0] 
        # rest of weights excluding quote currency
        wrest = np.delete(w,0,0); wdrest = np.delete(wdash,0,0)

        if self.cp == self.cs: 
            trf = self.c * np.sum(np.abs(wdash - w)) # initial estimate
        else: 
            warnings.warn('Commision rates not equal. Will initialise TRF '
            'randomly. Can lead to slower convergence.')
            trf = np.random.random() # random number between [0,1)

        while error < delta: 
            clipped_diff = np.clip(wdrest - trf*wrest, 0, None) # relu
            num = (1 - self.cp*qdash - K*np.sum(clipped_diff))  # numerator
            den = 1 - self.cp*q                                 # denominator
            
            prev = trf 
            trf = num / den # update trf with new value
            
            error = np.abs(trf - prev)
        
        return trf
            


def test_initialise(plot=False): 
    dp = DataPlotter()
    date1 = '1/5/2016 00:00:00'
    date2 = '1/6/2016 00:00:00'
    env = FinEnv(10,1,1,date1, date2)
    data = env._get_data()
    dp.data = data['BTC_DASH']
    if plot: dp.plot_candlestick()

def test_functionality(): 
    date1 = '1/5/2016 00:00:00'
    date2 = '1/6/2016 00:00:00'
    m = 12
    n = 50
    f = 3
    dm = FinDataManager(m,n,f,date1,date2)
    a = dm.prv()
    b = dm._normPMatrix(dm._v)
    c = dm._pTensor()
    print(a)
    print(b)
    print(c)

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
    test_initialise(plot=False)
    test_functionality()