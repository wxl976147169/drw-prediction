import lightgbm as lgbm
import joblib
from sklearn.linear_model import Ridge

def train_lgbm(X, y):
    # model = lgbm.LGBMRegressor(
    #     objective='regression',           # 默认也是'regression'，用于回归任务
    #     n_estimators=1000,                # 提高上限，结合 early_stopping 自动停止
    #     learning_rate=0.01,               # 小学习率更稳定，配合 n_estimators 较大
    #     max_depth=7,                      # 控制过拟合，默认 -1 表示不限制
    #     num_leaves=31,                    # 默认值，实际效果通常较好，可以调大一些如64
    #     subsample=0.8,                    # 行采样比例，防止过拟合
    #     colsample_bytree=0.8,            # 列采样比例，防止过拟合
    #     reg_alpha=1.0,                    # L1 正则化，防止过拟合
    #     reg_lambda=1.0,                   # L2 正则化，防止过拟合
    #     min_child_samples=20,             # 每个叶子最小样本数，防止过拟合
    #     random_state=42,                  # 保证复现性
    #     n_jobs=-1,                        # 并行线程数
    #     verbosity=-1                      # 不显示 warning
    # )
    model = Ridge(alpha=0.1, random_state = 42)
    model.fit(X, y)

    return model

def load_ml_regressor_model(path):
    return joblib.load(path)