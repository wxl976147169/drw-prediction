import lightgbm as lgbm
import joblib

def train_lgbm(X, y):
    model = lgbm.LGBMRegressor(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=-1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42
    )
    model.fit(X, y)

    return model

def load_ml_regressor_model(path):
    return joblib.load(path)