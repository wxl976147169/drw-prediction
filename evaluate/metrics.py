import numpy as np
from sklearn.metrics import r2_score,mean_squared_error,mean_absolute_error,mean_squared_log_error

def evaluation(true, predicted):
    mae = mean_absolute_error(true, predicted)
    me = mean_squared_error(true, predicted)
    mse = np.sqrt(me)
    r2 = r2_score(true, predicted)
    r = np.corrcoef(true, predicted)[0,1]

    return r, r2, mae, mse