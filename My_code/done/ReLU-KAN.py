import os, gc, math
import numpy as np, pandas as pd, torch
import torch.nn as nn, torch.optim as optim
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from scipy.stats import pearsonr
from torch.utils.data import DataLoader, TensorDataset
from KAN import KAN, KANLinear
os.environ["PYTORCH_CUDA_ALLOC_CONF"]="expandable_segments:True"
# —— 数据加载 & 清洗 ——————————————————————————
train = pd.read_parquet("/root/autodl-tmp/drw-crypto/train.parquet")
test  = pd.read_parquet("/root/autodl-tmp/drw-crypto/test.parquet")
submission = pd.read_csv("/root/autodl-tmp/drw-crypto/sample_submission.csv")

train = train.loc[:, ~train.isin([float('inf'), -float('inf')]).any()]
train = train.dropna(axis=1, how='all').dropna(axis=0, how='any')
def add_features(df):
    df['bid_ask_interaction'] = df['bid_qty'] * df['ask_qty']
    df['bid_buy_interaction'] = df['bid_qty'] * df['buy_qty']
    df['bid_sell_interaction'] = df['bid_qty'] * df['sell_qty']
    df['ask_buy_interaction'] = df['ask_qty'] * df['buy_qty']
    df['ask_sell_interaction'] = df['ask_qty'] * df['sell_qty']

    df['volume_weighted_sell'] = df['sell_qty'] * df['volume']
    df['buy_sell_ratio'] = df['buy_qty'] / (df['sell_qty'] + 1e-10)
    df['selling_pressure'] = df['sell_qty'] / (df['volume'] + 1e-10)
    df['log_volume'] = np.log1p(df['volume'])

    df['effective_spread_proxy'] = np.abs(df['buy_qty'] - df['sell_qty']) / (df['volume'] + 1e-10)
    df['bid_ask_imbalance'] = (df['bid_qty'] - df['ask_qty']) / (df['bid_qty'] + df['ask_qty'] + 1e-10)
    df['order_flow_imbalance'] = (df['buy_qty'] - df['sell_qty']) / (df['buy_qty'] + df['sell_qty'] + 1e-10)
    df['liquidity_ratio'] = (df['bid_qty'] + df['ask_qty']) / (df['volume'] + 1e-10)
    
    df['ask_buy_interaction_x_X293']=df['X293']*df['ask_buy_interaction']
     # Price Pressure Indicators
    df['net_order_flow'] = df['buy_qty'] - df['sell_qty']
    df['normalized_net_flow'] = df['net_order_flow'] / (df['volume'] + 1e-10)
    df['buying_pressure'] = df['buy_qty'] / (df['volume'] + 1e-10)
    df['volume_weighted_buy'] = df['buy_qty'] * df['volume']
    
    # Liquidity Depth Measures
    df['total_depth'] = df['bid_qty'] + df['ask_qty']
    df['depth_imbalance'] = (df['bid_qty'] - df['ask_qty']) / (df['total_depth'] + 1e-10)
    df['relative_spread'] = np.abs(df['bid_qty'] - df['ask_qty']) / (df['total_depth'] + 1e-10)
    df['log_depth'] = np.log1p(df['total_depth'])
    
    # Order Flow Toxicity Proxies
    df['kyle_lambda'] = np.abs(df['net_order_flow']) / (df['volume'] + 1e-10)
    df['flow_toxicity'] = np.abs(df['order_flow_imbalance']) * df['volume']
    df['aggressive_flow_ratio'] = (df['buy_qty'] + df['sell_qty']) / (df['total_depth'] + 1e-10)
    
    # Market Activity Indicators
    df['volume_depth_ratio'] = df['volume'] / (df['total_depth'] + 1e-10)
    df['activity_intensity'] = (df['buy_qty'] + df['sell_qty']) / (df['volume'] + 1e-10)
    df['log_buy_qty'] = np.log1p(df['buy_qty'])
    df['log_sell_qty'] = np.log1p(df['sell_qty'])
    df['log_bid_qty'] = np.log1p(df['bid_qty'])
    df['log_ask_qty'] = np.log1p(df['ask_qty'])
    
    # Microstructure Volatility Proxies
    df['realized_spread_proxy'] = 2 * np.abs(df['net_order_flow']) / (df['volume'] + 1e-10)
    df['price_impact_proxy'] = df['net_order_flow'] / (df['total_depth'] + 1e-10)
    df['quote_volatility_proxy'] = np.abs(df['depth_imbalance'])
    
    # Complex Interaction Terms
    df['flow_depth_interaction'] = df['net_order_flow'] * df['total_depth']
    df['imbalance_volume_interaction'] = df['order_flow_imbalance'] * df['volume']
    df['depth_volume_interaction'] = df['total_depth'] * df['volume']
    df['buy_sell_spread'] = np.abs(df['buy_qty'] - df['sell_qty'])
    df['bid_ask_spread'] = np.abs(df['bid_qty'] - df['ask_qty'])
    
    # Information Asymmetry Measures
    df['trade_informativeness'] = df['net_order_flow'] / (df['bid_qty'] + df['ask_qty'] + 1e-10)
    df['execution_shortfall_proxy'] = df['buy_sell_spread'] / (df['volume'] + 1e-10)
    df['adverse_selection_proxy'] = df['net_order_flow'] / (df['total_depth'] + 1e-10) * df['volume']
    
    # Market Efficiency Indicators
    df['fill_probability'] = df['volume'] / (df['buy_qty'] + df['sell_qty'] + 1e-10)
    df['execution_rate'] = (df['buy_qty'] + df['sell_qty']) / (df['total_depth'] + 1e-10)
    df['market_efficiency'] = df['volume'] / (df['bid_ask_spread'] + 1e-10)
    
    # Non-linear Transformations
    df['sqrt_volume'] = np.sqrt(df['volume'])
    df['sqrt_depth'] = np.sqrt(df['total_depth'])
    df['volume_squared'] = df['volume'] ** 2
    df['imbalance_squared'] = df['order_flow_imbalance'] ** 2
    
    # Relative Measures
    df['bid_ratio'] = df['bid_qty'] / (df['total_depth'] + 1e-10)
    df['ask_ratio'] = df['ask_qty'] / (df['total_depth'] + 1e-10)
    df['buy_ratio'] = df['buy_qty'] / (df['buy_qty'] + df['sell_qty'] + 1e-10)
    df['sell_ratio'] = df['sell_qty'] / (df['buy_qty'] + df['sell_qty'] + 1e-10)
    
    # Market Stress Indicators
    df['liquidity_consumption'] = (df['buy_qty'] + df['sell_qty']) / (df['total_depth'] + 1e-10)
    df['market_stress'] = df['volume'] / (df['total_depth'] + 1e-10) * np.abs(df['order_flow_imbalance'])
    df['depth_depletion'] = df['volume'] / (df['bid_qty'] + df['ask_qty'] + 1e-10)
    
    # Directional Indicators
    df['net_buying_ratio'] = df['net_order_flow'] / (df['volume'] + 1e-10)
    df['directional_volume'] = df['net_order_flow'] * np.log1p(df['volume'])
    df['signed_volume'] = np.sign(df['net_order_flow']) * df['volume']

    #etc
    df['sqrt_volume_div_log_volume'] = df['sqrt_volume'] / (df['log_volume'] + 1e-6)
    df['sqrt_volume_div_activity_intensity'] = df['sqrt_volume'] / (df['activity_intensity'] + 1e-6)
    df['sqrt_volume_mul_fill_probability'] = df['sqrt_volume'] * df['fill_probability']
    df['volume_div_sqrt_volume'] = df['volume'] / (df['sqrt_volume'] + 1e-6)
    df['sqrt_volume_div_fill_probability'] = df['sqrt_volume'] / (df['fill_probability'] + 1e-6)
    df['sqrt_volume_mul_activity_intensity'] = df['sqrt_volume'] * df['activity_intensity']
    df['sqrt_volume_div_log_sell_qty'] = df['sqrt_volume'] / (df['log_sell_qty'] + 1e-6)
    df['log_buy_qty_mul_sqrt_volume'] = df['log_buy_qty'] * df['sqrt_volume']
    df['sqrt_volume_mul_log_buy_qty'] = df['sqrt_volume'] * df['log_buy_qty']
    df['log_volume_mul_sqrt_volume'] = df['log_volume'] * df['sqrt_volume']
    # Replace infinities and NaNs
    df = df.replace([np.inf, -np.inf], 0).fillna(0)
    return df

train = add_features(train)
test = add_features(test)

feat_cols = train.drop(columns=["label"]).columns
test = test[feat_cols]

X = train[feat_cols].values
y = train["label"].values
X_test_raw = test.values

del train, test
gc.collect()

# —— 标准化输入输出 ————————————————
x_scaler = StandardScaler().fit(X)
X = x_scaler.transform(X)
X_test = x_scaler.transform(X_test_raw)

# y 标准化后反归一化使用
y_scaler = StandardScaler().fit(y.reshape(-1,1))
y = y_scaler.transform(y.reshape(-1,1)).ravel()

# —— 模型定义 ——————————————————————————


# —— 训练配置 ——————————————————————————
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# kf = KFold(n_splits=5, shuffle=True, random_state=37)
kf = KFold(n_splits=5, shuffle=False)
batch_size = 16
max_epoch = 150
fold_results = []

for fold, (tr_idx, val_idx) in enumerate(kf.split(X), 1):
    if fold != 5 :
        continue
    print(f"\n=== Fold {fold}/5 ===")
    # X_tr, y_tr = X[tr_idx], y[tr_idx]
    X_tr, y_tr = X, y
    X_val, y_val = X[val_idx], y[val_idx]

    ds_tr = TensorDataset(torch.tensor(X_tr, dtype=torch.float32),
                          torch.tensor(y_tr, dtype=torch.float32).unsqueeze(1))
    # loader_tr = DataLoader(ds_tr, batch_size=batch_size, shuffle=True)
    loader_tr = DataLoader(ds_tr, batch_size=batch_size, shuffle=False)

    ds_val = TensorDataset(torch.tensor(X_val, dtype=torch.float32), 
                           torch.tensor(y_val, dtype=torch.float32).unsqueeze(1))
    loader_val = DataLoader(ds_val, batch_size=batch_size, shuffle=False)

    model = KAN(layers_hidden=[X.shape[1], X.shape[1]//2, X.shape[1]//4, 1]).to(device)
    opt = optim.AdamW(model.parameters(), lr=5e-3, weight_decay=1e-4)
    total_steps = 50 * math.ceil(len(X_tr)/256)
    warmup_steps = int(0.1 * total_steps)
    scheduler_l = lambda step: min((step+1)/warmup_steps, 1)
    cosine = CosineAnnealingLR(opt, T_max=total_steps-warmup_steps)

    best_corr = -np.inf
    best_epoch = 0 
    best_val_loss = np.inf
    patience = 50
    noimp = 0
    global_step = 0

    for epoch in range(max_epoch):
        model.train()
        train_loss = 0
        for xb, yb in loader_tr:
            xb, yb = xb.to(device), yb.to(device)
            loss = nn.MSELoss()(model(xb), yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=5)
            opt.step()
            opt.zero_grad()
            step_lr_scale = scheduler_l(global_step)
            for pg in opt.param_groups: 
                pg['lr'] = step_lr_scale * pg['initial_lr']
            if global_step >= warmup_steps: 
                cosine.step()
            global_step += 1
            train_loss += loss.item()

        # 验证
        model.eval()
        val_preds = []
        val_loss = 0
        with torch.no_grad():
            for xb, yb in loader_val:
                xb, yb = xb.to(device), yb.to(device)
                out = model(xb)
                val_preds.append(out.cpu().numpy())
                loss = nn.MSELoss()(out, yb)
                val_loss += loss.item()
        val_pred = np.vstack(val_preds)
        val_pred = y_scaler.inverse_transform(val_pred)
        corr = pearsonr(val_pred.ravel(), y_scaler.inverse_transform(y_val.reshape(-1,1)).ravel())[0]

        print(f"Epoch {epoch+1}: loss={train_loss/len(loader_tr):.4f}, \
              val_loss={val_loss/len(loader_val):.4f}, \
                val_corr={corr:.4f}")

        if corr > (best_corr + 1e-3):
        # if (val_loss < (best_val_loss - 1e-3)) or (epoch > (best_epoch + 20) and (val_loss < (best_val_loss + 0.05))):
            best_corr = corr
            best_epoch = epoch
            best_val_loss = val_loss
            noimp = 0
            torch.save(model.state_dict(), f"relu_kan_fold{fold}.pt")
        else:
            noimp += 1
            if noimp >= patience:
                print("Early stopping.") 
                break

    fold_results.append(best_corr)
    print(f"Fold {fold} best Pearson r: {best_corr:.4f}")

# —— 汇总结果 ——————————————
print("\n✅ Cross-Validation Results:")
print("Fold r:", fold_results)
print("Mean r:", np.mean(fold_results), "Std r:", np.std(fold_results))

# —— 最终训练 + 验证提交  ——————————
model_final = KAN(layers_hidden=[X.shape[1], X.shape[1]//2, X.shape[1]//4, 1]).to(device)
model_final.load_state_dict(torch.load("relu_kan_fold5.pt"))
model_final.eval()
# model_final = model

ds_test = TensorDataset(torch.tensor(X_test, dtype=torch.float32))
loader_test = DataLoader(ds_test, batch_size=batch_size, shuffle=False)

test_preds = []
with torch.no_grad():
    for (xb,) in loader_test:
        xb = xb.to(device)
        out = model_final(xb)
        test_preds.append(out.cpu().numpy())
test_pred = np.vstack(test_preds)
test_pred = y_scaler.inverse_transform(test_pred).ravel()
submission["prediction"] = test_pred
submission.to_csv("submission_relu_kan.csv", index=False)
print("Submission generated.")