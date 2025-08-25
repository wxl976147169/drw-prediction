# 2-2 无base featurexgbregressor调参
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import warnings

from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold

from scipy.stats import rankdata, pearsonr

import shap
def reduce_mem_usage(dataframe, dataset):
    print('Reducing memory usage for:', dataset)
    initial_mem_usage = dataframe.memory_usage().sum() / 1024**2
    
    for col in dataframe.columns:
        col_type = dataframe[col].dtype

        c_min = dataframe[col].min()
        c_max = dataframe[col].max()
        if str(col_type)[:3] == 'int':
            if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                dataframe[col] = dataframe[col].astype(np.int8)
            elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                dataframe[col] = dataframe[col].astype(np.int16)
            elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                dataframe[col] = dataframe[col].astype(np.int32)
            elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                dataframe[col] = dataframe[col].astype(np.int64)
        else:
            if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                dataframe[col] = dataframe[col].astype(np.float16)
            elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                dataframe[col] = dataframe[col].astype(np.float32)
            else:
                dataframe[col] = dataframe[col].astype(np.float64)

    final_mem_usage = dataframe.memory_usage().sum() / 1024**2
    print('--- Memory usage before: {:.2f} MB'.format(initial_mem_usage))
    print('--- Memory usage after: {:.2f} MB'.format(final_mem_usage))
    print('--- Decreased memory usage by {:.1f}%\n'.format(100 * (initial_mem_usage - final_mem_usage) / initial_mem_usage))

    return dataframe
def add_features(df):
    df = df.copy()
    df['bid_ask_spread'] = df['ask_qty'] - df['bid_qty']
    df['bid_ask_ratio'] = df['bid_qty'] / (df['ask_qty'] + 1e-8)
    df['total_liquidity'] = df['bid_qty'] + df['ask_qty']
    df['liquidity_imbalance'] = (df['bid_qty'] - df['ask_qty']) / (df['total_liquidity'] + 1e-8)
    df['normalized_spread'] = df['bid_ask_spread'] / (df['total_liquidity'] + 1e-8)

    df['buy_sell_ratio'] = df['buy_qty'] / (df['sell_qty'] + 1e-8)
    df['net_order_flow'] = df['buy_qty'] - df['sell_qty']
    df['order_flow_imbalance'] = df['net_order_flow'] / (df['buy_qty'] + df['sell_qty'] + 1e-8)
    df['volume_participation'] = df['volume'] / (df['buy_qty'] + df['sell_qty'] + 1e-8)
    df['aggressive_ratio'] = (df['buy_qty'] + df['sell_qty']) / (df['volume'] + 1e-8)

    df['buy_pressure'] = df['buy_qty'] / (df['volume'] + 1e-8)
    df['sell_pressure'] = df['sell_qty'] / (df['volume'] + 1e-8)
    df['net_pressure'] = df['buy_pressure'] - df['sell_pressure']
    df['pressure_ratio'] = df['buy_pressure'] / (df['sell_pressure'] + 1e-8)

    df['depth_ratio'] = df['total_liquidity'] / (df['volume'] + 1e-8)
    df['bid_depth_ratio'] = df['bid_qty'] / (df['volume'] + 1e-8)
    df['ask_depth_ratio'] = df['ask_qty'] / (df['volume'] + 1e-8)
    df['depth_imbalance'] = (df['bid_depth_ratio'] - df['ask_depth_ratio']) / (df['depth_ratio'] + 1e-8)

    df['kyle_lambda'] = np.abs(df['net_order_flow']) / (df['volume'] + 1e-8)
    df['amihud_illiquidity'] = np.abs(df['net_pressure']) / (df['volume'] + 1e-8)
    df['liquidity_consumption'] = df['volume'] / (df['total_liquidity'] + 1e-8)

    df['price_efficiency'] = 1 / (1 + df['amihud_illiquidity'])
    df['execution_quality'] = df['volume'] / (df['bid_ask_spread'] + 1)

    df['pin_proxy'] = np.abs(df['order_flow_imbalance']) * df['amihud_illiquidity']
    df['order_toxicity'] = np.abs(df['order_flow_imbalance']) * df['kyle_lambda']

    df['bid_momentum'] = df['bid_qty'] * df['buy_qty'] / (df['volume'] + 1e-8)
    df['ask_momentum'] = df['ask_qty'] * df['sell_qty'] / (df['volume'] + 1e-8)
    df['liquidity_adjusted_volume'] = df['volume'] / np.sqrt(df['total_liquidity'] + 1)

    df['log_volume'] = np.log1p(df['volume'])
    df['log_liquidity'] = np.log1p(df['total_liquidity'])
    df['log_spread'] = np.log1p(np.abs(df['bid_ask_spread']))
    # Replace any NaNs or Infs
    df = df.replace([np.inf, -np.inf], 0).fillna(0)
    
    return df
def add_statistical_features(df):

    x_cols = [col for col in df.columns if col.startswith('X') and col[1:].isdigit()]
    x_data = df[x_cols]

    # Core stats
    df['x_stat_mean'] = x_data.mean(axis=1)
    df['x_stat_std'] = x_data.std(axis=1)
    df['x_stat_range'] = x_data.max(axis=1) - x_data.min(axis=1)
    df['x_stat_median'] = x_data.median(axis=1)
    df['x_stat_p25'] = x_data.quantile(0.25, axis=1)
    df['x_stat_p75'] = x_data.quantile(0.75, axis=1)

    # Count of values above row mean
    row_means = df['x_stat_mean'].values[:, None]
    df['x_stat_above_mean_count'] = (x_data.values > row_means).sum(axis=1)

    # Index (suffix) of max and min column
    df['x_stat_idx_max'] = x_data.idxmax(axis=1).str.extract(r'(\d+)', expand=False).astype(float)
    df['x_stat_idx_min'] = x_data.idxmin(axis=1).str.extract(r'(\d+)', expand=False).astype(float)

    # Cleanup: handle infs and NaNs
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.fillna(0, inplace=True)

    return df
    
def add_non_linear_x_market_interactions(df):

    market_features = ['volume', 'buy_qty', 'sell_qty','x_stat_mean', 'x_stat_median']
    x_cols = [col for col in df.columns if col.startswith('X') and col[1:].isdigit()]
    
    # Filter only available market features
    market_features = [feat for feat in market_features if feat in df.columns]
    for x_col in x_cols:
        for m_feat in market_features:
            new_col = f'{x_col}_log_x_{m_feat}'
            df[new_col] = df[x_col] * np.log(df[m_feat])
    
    # Cleanup
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.fillna(0, inplace=True)

    return df
def select_top_k_features_by_shap(X_train, y_train, feature_names, k=30):
    """Select top-k features using mean absolute SHAP value importance"""
    print("Calculating SHAP-based feature importance...")

    # Convert categories to int
    X_train = X_train.copy()
    for col in X_train.select_dtypes(include='category').columns:
        X_train[col] = X_train[col].astype(int)

    # Train XGBoost model
    params = {
        'n_estimators': 200,
        'max_depth': 9,
        'learning_rate': 0.02,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'random_state': 37,
        'n_jobs': -1,
        'verbosity': 0
    }

    model = XGBRegressor(**params)
    model.fit(X_train, y_train)

    # Compute SHAP values
    explainer = shap.Explainer(model, X_train)
    shap_values = explainer(X_train)

    # Calculate mean absolute SHAP value per feature
    mean_abs_shap = np.abs(shap_values.values).mean(axis=0)
    shap_importance_df = pd.DataFrame({
        'feature': X_train.columns,
        'importance': mean_abs_shap
    }).sort_values('importance', ascending=False)

    # Select top-k features
    selected_features = shap_importance_df.head(k)['feature'].tolist()

    # Always include critical features
    critical_features = ['order_flow_imbalance', 'kyle_lambda', 'vpin', 'volume',
                         'bid_ask_spread', 'liquidity_imbalance', 'buying_pressure']

    for feat in critical_features:
        if feat in feature_names and feat not in selected_features:
            selected_features.append(feat)

    print(f"Selected top {k} SHAP features (plus critical ones if needed). Total: {len(selected_features)}")

    return selected_features, shap_importance_df
def create_time_decay_weights(n, decay=0.95):

    pos = np.arange(n)
    norm = pos / (n - 1)
    w = decay ** (1.0 - norm)
    return w * n / w.sum()

def adjust_weights_for_outliers(X, y, base_weights, outlier_fraction=0.001):

    # Train quick model to estimate residuals
    # model = RandomForestRegressor(n_estimators=50, max_depth=10, random_state=37, n_jobs=-1)
    # model.fit(X, y, sample_weight=base_weights)
    # preds = model.predict(X)
    params = {
        'n_estimators': 200,
        'max_depth': 9,
        'learning_rate': 0.02,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'random_state': 37,
        'n_jobs': -1,
        'verbosity': 0
    }
    model = XGBRegressor(**params)
    model.fit(X_train, y_train)
    preds = model.predict(X)
    residuals = np.abs(y - preds)

    # Top N residuals = outliers
    n_outliers = max(1, int(outlier_fraction * len(residuals)))
    threshold = np.partition(residuals, -n_outliers)[-n_outliers]
    outlier_mask = residuals >= threshold

    # Downweight outliers (linear scale: 0.2–0.8 of base weight)
    adjusted_weights = base_weights.copy()
    if outlier_mask.any():
        res_out = residuals[outlier_mask]
        # res_norm = (res_out - res_out.min()) / (res_out.ptp() + 1e-8)
        res_norm = (res_out - res_out.min()) / (np.ptp(res_out) + 1e-8)
        weight_factors = 0.8 - 0.6 * res_norm
        adjusted_weights[outlier_mask] *= weight_factors

    return adjusted_weights
warnings.filterwarnings('ignore')

train = pd.read_parquet("/root/autodl-tmp/drw-crypto/train.parquet")
test = pd.read_parquet("/root/autodl-tmp/drw-crypto/test.parquet")
base_features = list(dict.fromkeys([
    "X752", "X287", "X298", "X759", "X302", "X55", "X56", "X52", "X303", "X51",
    "X598", "X385", "X603", "X674", "X415", "X345", "X174", "X178", "X168", "X612",
    "bid_qty", "ask_qty", "buy_qty", "sell_qty", "volume",
    "X758", "X296", "X611", "X780", "X451", "X25", "X591", "X727", "X427", "X288",
    "X721", "X312", "X421", "X471", "X573", "X255", "X144", "X299", "X301", "X563",
    "X737", "X702", "X507", "X306", "X501", "X586", "X43", "X517", "X248", "X137",
    "X757", "X196", "X777", "X280", "X266", "X689", "X294", "X492", "X555", "X731",
    "X262", "X576", "X13", "X518", "X502", "X558", "X6", "X602", "X695", "X703",
    "X413", "X660", "X37", "X15", "X310", "X512", "X362", "X631", "X214", "X562",
    "X488", "X510", "X256", "X35", "X128", "X86", "X170", "X30", "X265", "X323",
    "X559", "X348", "X130", "X529", "X20", "X4", "X90", "X192", "X91", "X582", "X99",
    "X24", "X317", "X707", "X653", "X519", "X557", "X371", "X84", "X83", "X360",
    "X111", "X699", "X187", "X637", "X567", "X577", "X313", "X60", "X671", "X698",
    "X701", "X725", "X292", "X638", "X741", "X379", "X700", "X614", "X676", "X516",
    "X697", "X311", "X615", "X706", "X466", "X571", "X17", "X584", "X436", "X305",
    "X34", "X282", "X681", "X7", "X208", "X41", "X536", "X548", "X776", "X87", "X40",
    "X570", "X539", "X474", "X753", "X425", "X217", "X199", "X18", "X609", "X21",
    "X277", "X279", "X326", "X540", "X688", "X553", "X452", "X738", "X183",
    "label"
]))
train = reduce_mem_usage(train, "train")
test = reduce_mem_usage(test, "test")

# train = train[base_features]
# test = test[base_features]

train = train.dropna().reset_index(drop=True)
test = test.fillna(0)

x_train = add_features(train)
x_train = add_statistical_features(x_train)
x_train = add_non_linear_x_market_interactions(x_train)
x_test = add_features(test)
x_test = add_statistical_features(x_test)
x_test = add_non_linear_x_market_interactions(x_test)
x_train = x_train[np.isfinite(x_train['label'])].reset_index(drop=True)
# Ensure the index is a datetime type
x_train.index = pd.to_datetime(x_train.index)

# Sort the dataset by timestamp (index)
x_train = x_train.sort_index()

# Define the split point (80% train, 20% validation)
split_index = int(len(x_train) * 0.8)

# Slice by index — NO shuffling
train_split = x_train.iloc[:split_index]
val_split = x_train.iloc[split_index:]

# Training features and labels
X_train = train_split.drop(columns=['label'])
y_train = train_split['label']

# Validation features and labels
X_val = val_split.drop(columns=['label'])
y_val = val_split['label']
feature_names = X_train.columns.tolist()
selected_features, importance_df = select_top_k_features_by_shap(X_train, y_train, feature_names, k=80)
X_train = X_train[selected_features]
X_val = X_val[selected_features]

from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

XGB_PARAMS = {
    "tree_method": "hist",
    "device": "gpu",
    "colsample_bylevel": 0.4778,
    "colsample_bynode": 0.3628,
    "colsample_bytree": 0.7107,
    "gamma": 1.7095,
    "learning_rate": 0.005,
    "max_depth": 32,
    "max_leaves": 32,
    "min_child_weight": 32,
    "n_estimators": 2048,
    "subsample": 0.3,
    "reg_alpha": 39.3524,
    "reg_lambda": 75.4484,
    "verbosity": 0,
    "random_state": 37,
    "n_jobs": -1
}

LEARNERS = [
    {"name": "xgb", "Estimator": XGBRegressor, "params": XGB_PARAMS}
]
# Ensure input is numpy and in correct time order
X = X_train.values
y = y_train.values
n_samples = len(X)

# Set up time-aware folds (no shuffling!)
kf = KFold(n_splits=5, shuffle=False)

# Initialize
oof_preds = np.zeros(n_samples)
models = []

# Fold loop
for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
    print(f"\n Fold {fold + 1}")
    
    # Split data
    X_tr, y_tr = X[train_idx], y[train_idx]
    X_val, y_val = X[val_idx], y[val_idx]
    
    # Time-decay + outlier-aware weights for training fold
    time_decay = create_time_decay_weights(len(train_idx), decay=0.95)
    weights = adjust_weights_for_outliers(X_tr, y_tr, time_decay)
    
    # Train model
    model = XGBRegressor(**XGB_PARAMS)
    model.fit(X_tr, y_tr, sample_weight=weights, eval_set=[(X_val, y_val)], verbose=False)
    
    # Predict on validation fold
    oof_preds[val_idx] = model.predict(X_val)
    
    # Store model if needed
    models.append(model)

# Evaluate OOF correlation
from scipy.stats import pearsonr
oof_corr = pearsonr(y, oof_preds)[0]
print(f"\nOOF Pearson Correlation: {oof_corr:.4f}")

X_val_np = X_val  # or use X_val directly if it's already numpy
val_preds = np.mean([model.predict(X_val_np) for model in models], axis=0)
pearson_corr, _ = pearsonr(y_val, val_preds)
print(f"Pearson Correlation on Validation: {pearson_corr:.4f}")

y_val_pred = model.predict(X_val)

# Pearson correlation
pearson_corr, _ = pearsonr(y_val, val_preds)
print(f"Pearson Correlation: {pearson_corr:.4f}")

# Optional: convert to arrays if needed
y_val = np.array(y_val)
y_val_pred = np.array(y_val_pred)

feature_names = [col for col in X_train.columns if col != 'label']
x_test_aligned = x_test[feature_names]
x_test_np = x_test_aligned.values if hasattr(x_test_aligned, "values") else x_test_aligned
y_pred = np.mean([model.predict(x_test_np) for model in models], axis=0)
submission = pd.read_csv("/root/autodl-tmp/drw-crypto/sample_submission.csv")
submission["prediction"] = y_pred
submission.to_csv("submission_shap_feature_ori_2_2_2.csv", index=False)
print("Submission file saved as 'submission_shap_feature_ori_2_2_2.csv'")
