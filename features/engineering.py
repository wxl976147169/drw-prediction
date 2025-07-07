import numpy as np 
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.compose import ColumnTransformer

def feature_engineering(X, Y, valid, base_feature=['bid_qty','ask_qty','buy_qty','sell_qty','volume']) : 
    # selector = SelectKBest(score_func=f_regression, k=200)
    # X_selected = selector.fit_transform(X, Y)
    # mask = selector.get_support()
    # selected_features = X.columns[mask]
    # print(selected_features)
    # cols_to_keep = base_feature + list(selected_features)
    cols_to_keep = X.columns
    X = X[cols_to_keep]
    cols_to_keep = list(X.columns)

    X_train,X_test,Y_train,Y_test = train_test_split(X, Y, test_size=0.3, random_state=42)
    numeric_cols = [fea for fea in X.columns if X[fea].dtype.kind in 'biufc']

    scaler = MinMaxScaler()
    transformer = ColumnTransformer(transformers =[('standard_scalling' , scaler, numeric_cols),], remainder = 'passthrough')

    X_train_tr = transformer.fit_transform(X_train)
    X_test_tr = transformer.transform(X_test)

    valid = valid[cols_to_keep]
    valid_trf = transformer.transform(valid)

    # 返回归一化后的训练测试验证集合
    return X_train_tr, Y_train, X_test_tr, Y_test, valid_trf



