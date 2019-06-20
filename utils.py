import time
import datetime
from dateutil.relativedelta import relativedelta

def date2unix(date): 
    '''
    Convert date string to UNIX timestamp. 
    Args: 
        date: string of format "date/month/year hour/minute/seconds", example: "01/12/2011 24:59:59"
    Returns: 
        timestamp: UNIX timestamp of requested date
    '''
    timestamp = time.mktime(datetime.datetime.strptime(date, "%d/%m/%Y %H:%M:%S").timetuple())
    return timestamp

def unix2date(timestamp): 
    '''
    Convert UNIX timestamp to date string
    '''
    return datetime.datetime.utcfromtimestamp(timestamp).strftime('%Y-%m-%d %H:%M:%S')

class RelativeDeltaWrapper():
    def __init__(self): 
        pass 
    
    def diffInMonths(self, date1, date2): 
        '''
        Args: 
            date1: datetime.date object. Later date
            date2: datetime.date object. Earlier date
        Returns: 
            n_months: int number of months. 
        '''
        assert type(date1) is type(datetime.date.today()), "date1 should be datetime.date object"
        assert type(date2) is type(datetime.date.today()), "date2 should be datetime.date object"

        diff = relativedelta(date1, date2)
        n_months = diff.years*12 + diff.months
        
        return n_months
    
    def diffInDays(self, date1, date2): 
        '''
        Args: 
            date1: datetime.date object. Later date
            date2: datetime.date object. Earlier date
        Returns: 
            diff.days: int number of days between date1 and date2
        '''
        assert type(date1) is type(datetime.date.today()), "date1 should be datetime.date object"
        assert type(date2) is type(datetime.date.today()), "date2 should be datetime.date object"

        diff = date1 - date2 
        return diff.days