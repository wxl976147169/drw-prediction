# 全部替换为TabNet
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import warnings
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold

from scipy.stats import rankdata, pearsonr
import torch
import torch.nn as nn, torch.optim as optim
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR
from pytorch_tabnet.tab_model import TabNetRegressor
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
    """
    Adds non-linear interaction features between 'X'-prefixed columns and selected market features.

    - For each column starting with 'X' followed by digits (e.g., 'X1', 'X23'), creates new features by 
      multiplying the X-column with the logarithm of each available market feature.
    - Market features considered: volume, buy_qty, sell_qty, x_stat_mean, x_stat_median.
    - Only uses market features that are present in the DataFrame.
    - Resulting feature name format: '{X_col}_log_x_{market_feature}'.
    - Cleans the resulting DataFrame by replacing NaNs and infinities with zero.

    Returns:
        pd.DataFrame: The input DataFrame with new interaction features added.
    """
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
# def select_top_k_features_by_shap(X_train, y_train, feature_names, k=30):
#     print("Calculating SHAP-based feature importance...")
#     # Convert categories to int
#     X_train = X_train.copy()
#     for col in X_train.select_dtypes(include='category').columns:
#         X_train[col] = X_train[col].astype(int)

#     # 1) 定义 TabNet 参数
#     tabnet_params = dict(
#     n_d=16, n_a=16, n_steps=8, gamma=1.5,
#     lambda_sparse=1e-3, mask_type='sparsemax',
#     optimizer_fn=torch.optim.AdamW,
#     optimizer_params=dict(lr=1e-2, weight_decay=1e-5),
#     scheduler_params=dict(step_size=50, gamma=0.9),
#     scheduler_fn=torch.optim.lr_scheduler.StepLR,
#     )
#     model = TabNetRegressor(**tabnet_params)
#     model.fit(
#         X_train.values[-100000:],
#         y_train.values[-100000:].reshape(-1, 1),
#         eval_set=[(X_train.values, y_train.values.reshape(-1, 1))],
#         eval_name=['val'],
#         eval_metric=['rmse'],
#         max_epochs=150,         
#         patience=50,            
#         batch_size=1024,
#         virtual_batch_size=256,
#         num_workers=4,
#         drop_last=False,
#         compute_importance=True
#     )
#     print('importances calculating')
#     # 3) 取出 TabNet 内部学到的特征重要性
#     importances = model.feature_importances_  # shape = (n_features,)

#     # Calculate mean absolute SHAP value per feature
#     shap_importance_df = pd.DataFrame({
#         'feature': X_train.columns,
#         'importance': importances
#     }).sort_values('importance', ascending=False)

#     # Select top-k features
#     selected_features = shap_importance_df.head(k)['feature'].tolist()

#     # Always include critical features
#     critical_features = ['order_flow_imbalance', 'kyle_lambda', 'vpin', 'volume',
#                          'bid_ask_spread', 'liquidity_imbalance', 'buying_pressure']

#     for feat in critical_features:
#         if feat in feature_names and feat not in selected_features:
#             selected_features.append(feat)

#     print(f"Selected top {k} SHAP features (plus critical ones if needed). Total: {len(selected_features)}")
#     return selected_features, shap_importance_df

def select_top_k_features_by_shap(X_train, y_train, feature_names, k=30, plot=False, sample_size=None):
    from pytorch_tabnet.tab_model import TabNetRegressor
    # from pytorch_tabnet.explain import explain_matrix
    import numpy as np
    import pandas as pd
    import torch

    print("Calculating SHAP-based feature importance...")

    # 可选：只采样部分数据来加速解释
    if sample_size is not None and sample_size < len(X_train):
        X_sample = X_train.sample(sample_size, random_state=42)
        y_sample = y_train.loc[X_sample.index]
    else:
        X_sample = X_train
        y_sample = y_train

    # 转换为 numpy 数组（TabNet 要求）
    X_np = X_sample.to_numpy() if isinstance(X_sample, pd.DataFrame) else X_sample
    y_np = y_sample.to_numpy() if isinstance(y_sample, pd.Series) else y_sample

    # 初始化 TabNet 模型（可按需修改）
    model = TabNetRegressor(
        n_d=8,
        n_a=8,
        n_steps=5,
        gamma=1.5,
        lambda_sparse=1e-4,
        optimizer_fn=torch.optim.Adam,
        optimizer_params=dict(lr=2e-2),
        scheduler_params={"step_size":10, "gamma":0.95},
        scheduler_fn=torch.optim.lr_scheduler.StepLR,
        seed=42,
        verbose=1
    )

    # 拟合模型（这里只训练 1 轮，只为解释用）
    model.fit(
        X_train=X_np[-10000:], y_train=y_np[-10000:].reshape(-1, 1),
        eval_set=[(X_np[-10000:], y_np[-10000:].reshape(-1, 1))],
        eval_name=["train"],
        eval_metric=["rmse"],
        max_epochs=100,
        patience=20,
        batch_size=2048,
        virtual_batch_size=256,
        num_workers=8,
        drop_last=False
    )

    # 显式计算解释矩阵（避免卡住）
    print('importances calculating')
    explain_matrix_, masks = model.explain(X_np[-10000:])
    importances = explain_matrix_.sum(axis=0)

    # 排序并选择前 k 个特征
    top_indices = np.argsort(importances)[::-1][:k]
    top_features = [feature_names[i] for i in top_indices]

    print("Top-{} features selected by SHAP importance:".format(k))
    print(top_features)

    # 可选：绘图展示 SHAP 全局重要性
    if plot:
        import matplotlib.pyplot as plt
        import seaborn as sns
        top_importances = importances[top_indices]

        plt.figure(figsize=(10, min(0.4 * k, 10)))
        sns.barplot(x=top_importances, y=top_features, orient='h')
        plt.xlabel("SHAP Importance (Summed over samples)")
        plt.title("Top-{} Features by TabNet SHAP".format(k))
        plt.tight_layout()
        plt.show()

    return top_features


def create_time_decay_weights(n, decay=0.95):
    """
    Generates a sequence of time-decay weights for a series of length `n`.

    Weights are assigned such that more recent observations (towards the end of the series)
    receive higher weight, following an exponential decay pattern. The weights are normalized
    to sum to `n`, preserving the original scale.

    Args:
        n (int): Number of observations.
        decay (float, optional): Decay rate between 0 and 1. Lower values decay faster.
                                 Defaults to 0.95.

    Returns:
        np.ndarray: A 1D array of shape (n,) containing the normalized time-decay weights.
    """
    pos = np.arange(n)
    norm = pos / (n - 1)
    w = decay ** (1.0 - norm)
    return w * n / w.sum()

def adjust_weights_for_outliers(X, y, base_weights, outlier_fraction=0.001):
    """
    Adjusts sample weights to downweight outliers based on model prediction residuals.

    A quick Random Forest model is trained using the provided features and labels
    to estimate residuals. Samples with the largest residuals (as defined by 
    `outlier_fraction`) are considered outliers and receive lower weights.

    The adjustment scales down the base weights for outliers on a linear scale 
    between 0.2 and 0.8 of the original value, depending on residual severity.

    Args:
        X (pd.DataFrame or np.ndarray): Feature matrix.
        y (pd.Series or np.ndarray): Target values.
        base_weights (np.ndarray): Original sample weights.
        outlier_fraction (float, optional): Fraction of samples to treat as outliers.
                                            Defaults to 0.001.

    Returns:
        np.ndarray: Adjusted sample weights with reduced influence for outliers.
    """
    # Train quick model to estimate residuals
    tabnet_params = dict(
    n_d=16, n_a=16, n_steps=8, gamma=1.5,
    lambda_sparse=1e-3, mask_type='sparsemax',
    optimizer_fn=torch.optim.AdamW,
    optimizer_params=dict(lr=1e-2, weight_decay=1e-5),
    scheduler_params=dict(step_size=50, gamma=0.9),
    scheduler_fn=torch.optim.lr_scheduler.StepLR,
    )
    model = TabNetRegressor(**tabnet_params)
    model.fit(
        X_train.values[-100000:],
        y_train.values[-100000:].reshape(-1, 1),
        eval_set=[(X_train.values, y_train.values.reshape(-1, 1))],
        eval_name=['val'],
        eval_metric=['rmse'],
        max_epochs=150,         
        patience=50,            
        batch_size=2048,
        virtual_batch_size=256,
        num_workers=8,
        drop_last=False,
        compute_importance=True
    )
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
# selected_features, importance_df = select_top_k_features_by_shap(X_train, y_train, feature_names, k=80)
# selected_features = select_top_k_features_by_shap(X_train, y_train, feature_names, k=80)
selected_features = ['X464_log_x_x_stat_median', 'X53_log_x_sell_qty', 'X199', 'X406_log_x_sell_qty', 'X84_log_x_sell_qty', 'X755_log_x_x_stat_median', 'X58_log_x_buy_qty', 'X37', 'X27_log_x_volume', 'X645_log_x_volume', 'X472_log_x_sell_qty', 'X548_log_x_volume', 'X749_log_x_x_stat_median', 'X51', 'X607_log_x_buy_qty', 'X267', 'X199_log_x_x_stat_mean', 'X434_log_x_volume', 'X454_log_x_volume', 'X162', 'X67', 'X266_log_x_volume', 'X196_log_x_sell_qty', 'X435_log_x_volume', 'X77_log_x_x_stat_median', 'X325_log_x_x_stat_median', 'X44_log_x_x_stat_median', 'X87_log_x_buy_qty', 'X33_log_x_buy_qty', 'X483_log_x_sell_qty', 'X766_log_x_buy_qty', 'X380_log_x_sell_qty', 'X683', 'X16_log_x_x_stat_mean', 'X721_log_x_buy_qty', 'X505_log_x_x_stat_median', 'X485_log_x_sell_qty', 'X359_log_x_volume', 'X32_log_x_sell_qty', 'X687_log_x_x_stat_median', 'X24_log_x_x_stat_median', 'X517_log_x_x_stat_mean', 'X298_log_x_x_stat_mean', 'X559_log_x_sell_qty', 'X356_log_x_x_stat_mean', 'X624_log_x_buy_qty', 'X250_log_x_x_stat_mean', 'X504_log_x_x_stat_median', 'X551_log_x_x_stat_mean', 'X447_log_x_buy_qty', 'X181_log_x_buy_qty', 'X607', 'X110_log_x_sell_qty', 'X750_log_x_buy_qty', 'X208_log_x_x_stat_mean', 'X56_log_x_x_stat_median', 'X517_log_x_x_stat_median', 'X529_log_x_x_stat_mean', 'X465', 'X500_log_x_volume', 'X577', 'X366_log_x_sell_qty', 'X367_log_x_x_stat_mean', 'X133_log_x_volume', 'X55_log_x_x_stat_mean', 'X162_log_x_sell_qty', 'X223_log_x_x_stat_median', 'X197_log_x_x_stat_median', 'X109_log_x_buy_qty', 'X165_log_x_sell_qty', 'X297', 'X14_log_x_sell_qty', 'X675_log_x_buy_qty', 'X80', 'X755_log_x_sell_qty', 'X538', 'X81_log_x_sell_qty', 'X476', 'X625_log_x_buy_qty', 'X265']

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
    "learning_rate": 0.02213,
    "max_depth": 20,
    "max_leaves": 12,
    "min_child_weight": 16,
    "n_estimators": 1667,
    "subsample": 0.3,
    "reg_alpha": 39.3524,
    "reg_lambda": 75.4484,
    "verbosity": 0,
    "random_state": 42,
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
    # time_decay = create_time_decay_weights(len(train_idx), decay=0.95)
    # weights = adjust_weights_for_outliers(X_tr, y_tr, time_decay)
    
    # Train model
    # model = XGBRegressor(**XGB_PARAMS)
    # model.fit(X_tr, y_tr, sample_weight=weights, eval_set=[(X_val, y_val)], verbose=False)
    tabnet_params = dict(
    n_d=32, n_a=32, n_steps=10, gamma=1.5,
    lambda_sparse=1e-3, mask_type='sparsemax',
    optimizer_fn=torch.optim.AdamW,
    optimizer_params=dict(lr=1e-2, weight_decay=1e-5),
    scheduler_params=dict(step_size=50, gamma=0.9),
    scheduler_fn=torch.optim.lr_scheduler.StepLR,
    )
    model = TabNetRegressor(**tabnet_params)
    model.fit(
        X_train=X_tr, y_train=y_tr.reshape(-1, 1),
        eval_set=[(X_val, y_val.reshape(-1, 1))],
        eval_name=['val'], eval_metric=['rmse'],
        max_epochs=300, patience=100,
        batch_size=2048, virtual_batch_size=256,
        num_workers=8, drop_last=False
    )
    # Predict on validation fold
    oof_preds[val_idx] = model.predict(X_val).ravel()
    # Store model if needed
    models.append(model)

# Evaluate OOF correlation
from scipy.stats import pearsonr
oof_corr = pearsonr(y, oof_preds)[0]
print(f"\nOOF Pearson Correlation: {oof_corr:.4f}")

X_val_np = X_val  # or use X_val directly if it's already numpy
val_preds = np.mean([model.predict(X_val_np).ravel() for model in models], axis=0)

pearson_corr, _ = pearsonr(y_val, val_preds)
print(f"Pearson Correlation on Validation: {pearson_corr:.4f}")

y_val_pred = model.predict(X_val).ravel()

# Pearson correlation
pearson_corr, _ = pearsonr(y_val, val_preds)
print(f"Pearson Correlation: {pearson_corr:.4f}")

# Optional: convert to arrays if needed
y_val = np.array(y_val)
y_val_pred = np.array(y_val_pred)

feature_names = [col for col in X_train.columns if col != 'label']
x_test_aligned = x_test[feature_names]
x_test_np = x_test_aligned.values if hasattr(x_test_aligned, "values") else x_test_aligned
y_pred = np.mean([model.predict(x_test_np).ravel() for model in models], axis=0)
submission = pd.read_csv("/root/autodl-tmp/drw-crypto/sample_submission.csv")
submission["prediction"] = y_pred
submission.to_csv("submission_shap_feature_2_1.csv", index=False)
print("Submission file saved as 'submission_shap_feature_2_1.csv'")
