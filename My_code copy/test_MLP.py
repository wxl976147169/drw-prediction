import os, gc, math
import numpy as np, pandas as pd, torch
import torch.nn as nn, torch.optim as optim
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from scipy.stats import pearsonr
from torch.utils.data import DataLoader, TensorDataset
os.environ["PYTORCH_CUDA_ALLOC_CONF"]="expandable_segments:True"
# —— 数据加载 & 清洗 ——————————————————————————
train = pd.read_parquet("/root/autodl-tmp/drw-crypto/train.parquet")
test  = pd.read_parquet("/root/autodl-tmp/drw-crypto/test.parquet")
submission = pd.read_csv("/root/autodl-tmp/drw-ipynb/result/sample_submission.csv")

train = train.loc[:, ~train.isin([float('inf'), -float('inf')]).any()]
train = train.dropna(axis=1, how='all').dropna(axis=0, how='any')
# def add_features(df):
    
#     df['bid_ask_interaction'] = df['bid_qty'] * df['ask_qty']
#     df['bid_buy_interaction'] = df['bid_qty'] * df['buy_qty']
#     df['bid_sell_interaction'] = df['bid_qty'] * df['sell_qty']
#     df['ask_buy_interaction'] = df['ask_qty'] * df['buy_qty']
#     df['ask_sell_interaction'] = df['ask_qty'] * df['sell_qty']

#     df['volume_weighted_sell'] = df['sell_qty'] * df['volume']
#     df['buy_sell_ratio'] = df['buy_qty'] / (df['sell_qty'] + 1e-10)
#     df['selling_pressure'] = df['sell_qty'] / (df['volume'] + 1e-10)
#     df['log_volume'] = np.log1p(df['volume'])

#     df['effective_spread_proxy'] = np.abs(df['buy_qty'] - df['sell_qty']) / (df['volume'] + 1e-10)
#     df['bid_ask_imbalance'] = (df['bid_qty'] - df['ask_qty']) / (df['bid_qty'] + df['ask_qty'] + 1e-10)
#     df['order_flow_imbalance'] = (df['buy_qty'] - df['sell_qty']) / (df['buy_qty'] + df['sell_qty'] + 1e-10)
#     df['liquidity_ratio'] = (df['bid_qty'] + df['ask_qty']) / (df['volume'] + 1e-10)
    
#     df['ask_buy_interaction_x_X293']=df['X293']*df['ask_buy_interaction']
#      # Price Pressure Indicators
#     df['net_order_flow'] = df['buy_qty'] - df['sell_qty']
#     df['normalized_net_flow'] = df['net_order_flow'] / (df['volume'] + 1e-10)
#     df['buying_pressure'] = df['buy_qty'] / (df['volume'] + 1e-10)
#     df['volume_weighted_buy'] = df['buy_qty'] * df['volume']
    
#     # Liquidity Depth Measures
#     df['total_depth'] = df['bid_qty'] + df['ask_qty']
#     df['depth_imbalance'] = (df['bid_qty'] - df['ask_qty']) / (df['total_depth'] + 1e-10)
#     df['relative_spread'] = np.abs(df['bid_qty'] - df['ask_qty']) / (df['total_depth'] + 1e-10)
#     df['log_depth'] = np.log1p(df['total_depth'])
    
#     # Order Flow Toxicity Proxies
#     df['kyle_lambda'] = np.abs(df['net_order_flow']) / (df['volume'] + 1e-10)
#     df['flow_toxicity'] = np.abs(df['order_flow_imbalance']) * df['volume']
#     df['aggressive_flow_ratio'] = (df['buy_qty'] + df['sell_qty']) / (df['total_depth'] + 1e-10)
    
#     # Market Activity Indicators
#     df['volume_depth_ratio'] = df['volume'] / (df['total_depth'] + 1e-10)
#     df['activity_intensity'] = (df['buy_qty'] + df['sell_qty']) / (df['volume'] + 1e-10)
#     df['log_buy_qty'] = np.log1p(df['buy_qty'])
#     df['log_sell_qty'] = np.log1p(df['sell_qty'])
#     df['log_bid_qty'] = np.log1p(df['bid_qty'])
#     df['log_ask_qty'] = np.log1p(df['ask_qty'])
    
#     # Microstructure Volatility Proxies
#     df['realized_spread_proxy'] = 2 * np.abs(df['net_order_flow']) / (df['volume'] + 1e-10)
#     df['price_impact_proxy'] = df['net_order_flow'] / (df['total_depth'] + 1e-10)
#     df['quote_volatility_proxy'] = np.abs(df['depth_imbalance'])
    
#     # Complex Interaction Terms
#     df['flow_depth_interaction'] = df['net_order_flow'] * df['total_depth']
#     df['imbalance_volume_interaction'] = df['order_flow_imbalance'] * df['volume']
#     df['depth_volume_interaction'] = df['total_depth'] * df['volume']
#     df['buy_sell_spread'] = np.abs(df['buy_qty'] - df['sell_qty'])
#     df['bid_ask_spread'] = np.abs(df['bid_qty'] - df['ask_qty'])
    
#     # Information Asymmetry Measures
#     df['trade_informativeness'] = df['net_order_flow'] / (df['bid_qty'] + df['ask_qty'] + 1e-10)
#     df['execution_shortfall_proxy'] = df['buy_sell_spread'] / (df['volume'] + 1e-10)
#     df['adverse_selection_proxy'] = df['net_order_flow'] / (df['total_depth'] + 1e-10) * df['volume']
    
#     # Market Efficiency Indicators
#     df['fill_probability'] = df['volume'] / (df['buy_qty'] + df['sell_qty'] + 1e-10)
#     df['execution_rate'] = (df['buy_qty'] + df['sell_qty']) / (df['total_depth'] + 1e-10)
#     df['market_efficiency'] = df['volume'] / (df['bid_ask_spread'] + 1e-10)
    
#     # Non-linear Transformations
#     df['sqrt_volume'] = np.sqrt(df['volume'])
#     df['sqrt_depth'] = np.sqrt(df['total_depth'])
#     df['volume_squared'] = df['volume'] ** 2
#     df['imbalance_squared'] = df['order_flow_imbalance'] ** 2
    
#     # Relative Measures
#     df['bid_ratio'] = df['bid_qty'] / (df['total_depth'] + 1e-10)
#     df['ask_ratio'] = df['ask_qty'] / (df['total_depth'] + 1e-10)
#     df['buy_ratio'] = df['buy_qty'] / (df['buy_qty'] + df['sell_qty'] + 1e-10)
#     df['sell_ratio'] = df['sell_qty'] / (df['buy_qty'] + df['sell_qty'] + 1e-10)
    
#     # Market Stress Indicators
#     df['liquidity_consumption'] = (df['buy_qty'] + df['sell_qty']) / (df['total_depth'] + 1e-10)
#     df['market_stress'] = df['volume'] / (df['total_depth'] + 1e-10) * np.abs(df['order_flow_imbalance'])
#     df['depth_depletion'] = df['volume'] / (df['bid_qty'] + df['ask_qty'] + 1e-10)
    
#     # Directional Indicators
#     df['net_buying_ratio'] = df['net_order_flow'] / (df['volume'] + 1e-10)
#     df['directional_volume'] = df['net_order_flow'] * np.log1p(df['volume'])
#     df['signed_volume'] = np.sign(df['net_order_flow']) * df['volume']

#     #etc
#     df['sqrt_volume_div_log_volume'] = df['sqrt_volume'] / (df['log_volume'] + 1e-6)
#     df['sqrt_volume_div_activity_intensity'] = df['sqrt_volume'] / (df['activity_intensity'] + 1e-6)
#     df['sqrt_volume_mul_fill_probability'] = df['sqrt_volume'] * df['fill_probability']
#     df['volume_div_sqrt_volume'] = df['volume'] / (df['sqrt_volume'] + 1e-6)
#     df['sqrt_volume_div_fill_probability'] = df['sqrt_volume'] / (df['fill_probability'] + 1e-6)
#     df['sqrt_volume_mul_activity_intensity'] = df['sqrt_volume'] * df['activity_intensity']
#     df['sqrt_volume_div_log_sell_qty'] = df['sqrt_volume'] / (df['log_sell_qty'] + 1e-6)
#     df['log_buy_qty_mul_sqrt_volume'] = df['log_buy_qty'] * df['sqrt_volume']
#     df['sqrt_volume_mul_log_buy_qty'] = df['sqrt_volume'] * df['log_buy_qty']
#     df['log_volume_mul_sqrt_volume'] = df['log_volume'] * df['sqrt_volume']
#     # Replace infinities and NaNs
#     df = df.replace([np.inf, -np.inf], 0).fillna(0)
#     return df

# SELECTED_FEATURES=[
#         "buy_qty", "sell_qty", "volume", 'bid_ask_interaction', 'bid_buy_interaction', 'bid_sell_interaction', 'ask_buy_interaction',
#         'ask_sell_interaction', "log_volume", 'net_order_flow', 'normalized_net_flow',
#         'buying_pressure', 'volume_weighted_buy', 'total_depth', 'depth_imbalance',
#         'relative_spread', 'log_depth', 'kyle_lambda', 'flow_toxicity', 'aggressive_flow_ratio',
#         'volume_depth_ratio', 'activity_intensity', 'log_buy_qty', 'log_sell_qty',
#         'log_bid_qty', 'log_ask_qty', 'realized_spread_proxy', 'price_impact_proxy',
#         'quote_volatility_proxy', 'flow_depth_interaction', 'imbalance_volume_interaction',
#         'depth_volume_interaction',  'trade_informativeness',
#         'execution_shortfall_proxy', 'adverse_selection_proxy', 'fill_probability',
#         'execution_rate', 'market_efficiency', 'sqrt_volume', 'sqrt_depth', 'volume_squared',
#         'imbalance_squared', 'bid_ratio', 'ask_ratio', 'buy_ratio', 'sell_ratio',
#         'liquidity_consumption', 'market_stress', 'depth_depletion', 'net_buying_ratio',
#         'directional_volume', 'signed_volume',   
#         "sqrt_volume_mul_fill_probability",
#         "volume_div_sqrt_volume",
#         "log_buy_qty_mul_sqrt_volume",
#         "sqrt_volume_mul_log_buy_qty",
#         "log_volume_mul_sqrt_volume"
#     ]

# train = add_features(train)
# test = add_features(test)

feat_cols = train.drop(columns=["label"]).columns
test = test[feat_cols]

X = train[feat_cols].values
y = train["label"].values
X_test_raw = test.values
# X = train[SELECTED_FEATURES].values
# y = train["label"].values
# X_test_raw = test[SELECTED_FEATURES].values

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
# class ReLUKAN(nn.Module):
#     def __init__(self, input_dim, hidden_dim=64, grid_size=256, depth=9):
#         super().__init__()
#         self.depth = depth
#         self.grids = nn.ParameterList([nn.Parameter(torch.linspace(0,1,grid_size),requires_grad=False)
#                                        for _ in range(depth)])
#         self.fs = nn.ParameterList([nn.Parameter(torch.randn(input_dim, grid_size, hidden_dim))
#                                     for _ in range(depth)])
#         self.post = nn.Sequential(
#             nn.Linear(hidden_dim, hidden_dim*2),
#             nn.ReLU(),
#             nn.Linear(hidden_dim*2, hidden_dim),
#             nn.ReLU(),
#             nn.Linear(hidden_dim, 1)
#         )

#     def forward(self, x):
#         z = (x - x.min(0)[0])/(x.max(0)[0] - x.min(0)[0] + 1e-6)
#         h = None
#         for i in range(self.depth):
#             B = torch.relu(z.unsqueeze(-1) - self.grids[i])
#             E = (B.unsqueeze(-1) * self.fs[i].unsqueeze(0)).sum(2)
#             H = E.sum(1)
#             h = H if i == 0 else h + H
#         return self.post(h)
class ReLUKAN(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, grid_size=128, depth=7, 
                 dropout=0.1, use_layernorm=True, residual=True, 
                 learnable_grid=True, use_attention=True, activation='gelu'):
        super().__init__()
        self.depth = depth
        self.use_attention = use_attention
        self.use_layernorm = use_layernorm
        self.residual = residual
        # 可学习 grid
        self.grids = nn.ParameterList([
            nn.Parameter(torch.rand(input_dim, grid_size)) if learnable_grid 
            else nn.Parameter(torch.linspace(0, 1, grid_size).repeat(input_dim, 1), requires_grad=False)
            for _ in range(depth)
        ])
        # Feature selectors（可学习函数）
        self.fs = nn.ParameterList([
            nn.Parameter(torch.randn(input_dim, grid_size, hidden_dim)) 
            for _ in range(depth)
        ])
        # depth-wise attention
        if self.use_attention:
            self.depth_scores = nn.Parameter(torch.ones(depth))
        # MLP 后处理
        mlp = []
        for _ in range(3):
            mlp.append(nn.Linear(hidden_dim, hidden_dim))
            if self.use_layernorm:
                mlp.append(nn.BatchNorm1d(hidden_dim))
            mlp.append(getattr(F, activation) if activation != 'gelu' else nn.GELU())
            mlp.append(nn.Dropout(dropout))
        mlp.append(nn.Linear(hidden_dim, 1))
        self.mlp = nn.Sequential(*mlp)

    def forward(self, x):
        # Normalize input to [0, 1]
        z = (x - x.min(dim=0, keepdim=True)[0]) / (
            x.max(dim=0, keepdim=True)[0] - x.min(dim=0, keepdim=True)[0] + 1e-6
        )
        outputs = []
        for i in range(self.depth):
            grid = self.grids[i].unsqueeze(0)  # (1, input_dim, grid_size)
            B = F.relu(z.unsqueeze(-1) - grid)  # (B, input_dim, grid_size)
            E = (B.unsqueeze(-1) * self.fs[i].unsqueeze(0)).sum(2)  # (B, input_dim, hidden)
            H = E.sum(1)  # (B, hidden)
            if self.residual and outputs:
                H = H + outputs[-1]
            outputs.append(H)
        stacked = torch.stack(outputs, dim=0)  # (depth, B, hidden)
        if self.use_attention:
            weights = F.softmax(self.depth_scores, dim=0)
            h = (weights.view(-1, 1, 1) * stacked).sum(0)  # (B, hidden)
        else:
            h = stacked.sum(0)  # (B, hidden)
        return self.mlp(h)

class MLP(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 200),
            nn.BatchNorm1d(200),
            nn.ReLU(),
            nn.Linear(200, 100),
            nn.BatchNorm1d(100),
            nn.ReLU(),
            nn.Linear(100, 50),
            nn.BatchNorm1d(50),
            nn.ReLU(),
            nn.Linear(50, 1)
        )
        
    def forward(self, x):
        return self.model(x)

# —— 训练配置 ——————————————————————————
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# kf = KFold(n_splits=5, shuffle=True, random_state=37)
kf = KFold(n_splits=5, shuffle=False)
batch_size = 64
max_epoch = 200
fold_results = []
# —— 最终训练 + 验证提交 (这里用 fold1 模型示例) ——————————
model_final = ReLUKAN(input_dim=X.shape[1]).to(device)
model_final.load_state_dict(torch.load("/root/autodl-tmp/My_code/relu_kan_fold5.pt"))
model_final.eval()

ds_test = TensorDataset(torch.tensor(X_test, dtype=torch.float32))
loader_test = DataLoader(ds_test, batch_size=batch_size, shuffle=False)
# with torch.no_grad():
#     test_pred = model_final(torch.tensor(X_test, dtype=torch.float32).to(device)).cpu().numpy()
test_preds = []
with torch.no_grad():
    for (xb,) in loader_test:
        xb = xb.to(device)
        out = model_final(xb)
        test_preds.append(out.cpu().numpy())
test_pred = np.vstack(test_preds)
test_pred = y_scaler.inverse_transform(test_pred).ravel()
submission["prediction"] = test_pred
submission.to_csv("submission_relu_kan_test.csv", index=False)
print("Submission generated.")

