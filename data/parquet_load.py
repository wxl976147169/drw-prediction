import pandas as pd
import numpy as np

def parquet_load(train_path='/root/Kaggle/drw-crypto-market-prediction/input/train_sub199.parquet', valid_path='/root/Kaggle/drw-crypto-market-prediction/input/test_sub99.parquet'):
    train = pd.read_parquet(train_path) # 525887
    valid = pd.read_parquet(valid_path) # 538150

    train.replace([np.inf, -np.inf], np.nan, inplace=True)
    train.isnull().sum().sort_values(ascending=False)
    null_cols = train.isnull().sum().sort_values(ascending=False)[lambda x: x >0].index
    # 去掉空列
    train.drop(columns=null_cols, inplace=True)
    X = train.drop(columns=['label']) 
    Y = train['label']
    
    valid.drop(columns=null_cols, inplace=True) # 896 -> 875
    valid = valid.drop(columns=['label']) 

    return X, Y, valid

if __name__ == '__main__' : 
    parquet_load()