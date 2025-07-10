import numpy as np
from data.parquet_load import parquet_load
from features.engineering import feature_engineering
from models.nn_model import AutoEncoder, AutoEncoder_MLP
from train.train import train_model
from evaluate.metrics import evaluation
from predict.inference import predict, predict_w_ml
import torch
from sklearn.model_selection import train_test_split

def add_features(df):
    # Original features
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
    # === NEW MICROSTRUCTURE FEATURES ===
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
    # Replace infinities and NaNs
    df = df.replace([np.inf, -np.inf], 0).fillna(0)
    return df

def main(mode):
    # train_path = '/root/autodl-tmp/drw-crypto/input_sub199_sub99/train.parquet'
    # test_path = '/root/autodl-tmp/drw-crypto/input_sub199_sub99/test.parquet'
    train_path = '/root/autodl-tmp/drw-crypto/train.parquet'
    test_path = '/root/autodl-tmp/drw-crypto/test.parquet'
    X, Y, valid = parquet_load(train_path, test_path)
    X = add_features(X)
    valid = add_features(valid)
    X_train, Y_train, X_test, Y_test, valid = feature_engineering(X, Y, valid)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = AutoEncoder_MLP(input_size=X_train.shape[1], output_size = X_train.shape[1], encoding_size=128).to(device)

    if mode == 'train':
        train_model(model, X_train, Y_train, X_test, Y_test, device,
                    epochs = 2000,
                    batch_size = 256,
                    lr = 0.0025)

    elif mode == 'predict':
        predict(model, 
                valid,
                checkpoint_path = '/root/autodl-tmp/drw-prediction/checkpoints/model_epoch1668_test_loss0.1709.pt', 
                submission_template_path = '/root/autodl-tmp/drw-prediction/result/sample_submission.csv',  
                output_path = '/root/autodl-tmp/drw-prediction/result/',  
                device = device, 
                )
        # predict_w_ml(model, 
        #         valid,
        #         checkpoint_path = '/root/autodl-tmp/drw-prediction/checkpoints/model_epoch15_dl_test_loss0.4253.pt', 
        #         submission_template_path = '/root/autodl-tmp/drw-prediction/result/sample_submission.csv',  
        #         output_path = '/root/autodl-tmp/drw-prediction/result/',  
        #         device = device, 
        #         ml_regressor_path = '/root/autodl-tmp/drw-prediction/checkpoints/model_epoch15_test_loss1.0197.pkl' )

if __name__ == '__main__':
    main(mode = 'train')