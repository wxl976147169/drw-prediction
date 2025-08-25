import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.preprocessing import StandardScaler
from scipy.stats import pearsonr
import warnings
warnings.filterwarnings('ignore')
import gc
from typing import Dict, List, Tuple, Optional
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import time

# ===== Feature Engineering =====
def feature_engineering(df):
    """Original features plus new robust features"""
    # Convert to float32 for memory efficiency
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        df[col] = df[col].astype(np.float32)
    
    # Original features
    df['volume_weighted_sell'] = (df['sell_qty'] * df['volume']).astype(np.float32)
    df['buy_sell_ratio'] = (df['buy_qty'] / (df['sell_qty'] + 1e-8)).astype(np.float32)
    df['selling_pressure'] = (df['sell_qty'] / (df['volume'] + 1e-8)).astype(np.float32)
    df['effective_spread_proxy'] = (np.abs(df['buy_qty'] - df['sell_qty']) / (df['volume'] + 1e-8)).astype(np.float32)
    
    # New robust features
    df['log_volume'] = np.log1p(df['volume']).astype(np.float32)
    df['bid_ask_imbalance'] = ((df['bid_qty'] - df['ask_qty']) / (df['bid_qty'] + df['ask_qty'] + 1e-8)).astype(np.float32)
    df['order_flow_imbalance'] = ((df['buy_qty'] - df['sell_qty']) / (df['buy_qty'] + df['sell_qty'] + 1e-8)).astype(np.float32)
    df['liquidity_ratio'] = ((df['bid_qty'] + df['ask_qty']) / (df['volume'] + 1e-8)).astype(np.float32)
    
    return df

# ===== Configuration =====
class Config:
    TRAIN_PATH = "/root/autodl-tmp/drw-crypto/train.parquet"
    TEST_PATH = "/root/autodl-tmp/drw-crypto/test.parquet"
    SUBMISSION_PATH = "/root/autodl-tmp/drw-crypto/sample_submission.csv"
    
    # Core features
    CORE_FEATURES = [
        "X598", "X385", "X603", "X674",
        "X415", "X345", "X174", "X302", "X178", "X168", "X612",
        "buy_qty", "sell_qty", "volume", "X421", "X333",
        "bid_qty", "ask_qty"
    ]
    
    # Engineered features
    ENGINEERED_FEATURES = [
        'volume_weighted_sell', 'buy_sell_ratio', 'selling_pressure', 'effective_spread_proxy',
        'log_volume', 'bid_ask_imbalance', 'order_flow_imbalance', 'liquidity_ratio'
    ]
    
    LABEL_COLUMN = "label"
    N_FOLDS = 3
    RANDOM_STATE = 42
    
    # GPU availability check
    USE_GPU = False  # Set to False for CPU-only execution

# ===== Model Parameters =====
XGB_PARAMS = {
    "tree_method": "hist",
    "device": "cpu",  # Changed to CPU for compatibility
    "colsample_bylevel": 0.4778,
    "colsample_bynode": 0.3628,
    "colsample_bytree": 0.7107,
    "gamma": 1.7095,
    "learning_rate": 0.02213,
    "max_depth": 20,
    "max_leaves": 12,
    "min_child_weight": 16,
    "n_estimators": 1667,
    "subsample": 0.06567,
    "reg_alpha": 39.3524,
    "reg_lambda": 75.4484,
    "verbosity": 0,
    "random_state": Config.RANDOM_STATE,
    "n_jobs": -1
}

LGBM_PARAMS = {
    "n_estimators": 1000,
    "learning_rate": 0.025,
    "num_leaves": 15,
    "min_child_samples": 30,
    "subsample": 0.1,
    "colsample_bytree": 0.7,
    "reg_alpha": 20,
    "reg_lambda": 50,
    "random_state": Config.RANDOM_STATE,
    "device": "cpu",
    "verbosity": -1,
    "n_jobs": 4
}

# ===== Performance Tracker =====
class PerformanceTracker:
    """Track model performance and generate insights"""
    
    def __init__(self):
        self.baseline_score = None
        self.results = []
        self.feature_importance = {}
        self.insights = []
        
    def record_baseline(self, score: float, model_type: str):
        """Record baseline performance"""
        self.baseline_score = score
        print(f"\n📊 BASELINE ESTABLISHED")
        print(f"   Model: {model_type}")
        print(f"   Score: {score:.6f}")
        print("="*60)
        
    def record_feature_test(self, feature: str, score: float, model_type: str, improvement: float):
        """Record single feature addition test"""
        self.results.append({
            'feature': feature,
            'score': score,
            'model_type': model_type,
            'improvement': improvement
        })
        
        # Update feature importance
        if feature not in self.feature_importance:
            self.feature_importance[feature] = []
        self.feature_importance[feature].append(improvement)
        
        # Generate insight
        if improvement > 0.001:
            self.insights.append(f"✅ {feature} improves {model_type} by {improvement:.4f}")
        elif improvement < -0.001:
            self.insights.append(f"❌ {feature} degrades {model_type} by {improvement:.4f}")
            
    def get_top_features(self, n: int = 5) -> List[str]:
        """Get top performing features based on average improvement"""
        avg_improvements = {}
        for feature, improvements in self.feature_importance.items():
            avg_improvements[feature] = np.mean(improvements)
            
        sorted_features = sorted(avg_improvements.items(), key=lambda x: x[1], reverse=True)
        return [feat for feat, _ in sorted_features[:n]]
    
    def print_summary(self):
        """Print comprehensive performance summary"""
        print("\n" + "="*80)
        print("🔬 PERFORMANCE ANALYSIS SUMMARY")
        print("="*80)
        
        if not self.results:
            print("No results recorded yet.")
            return
            
        # Best overall result
        best_result = max(self.results, key=lambda x: x['score'])
        print(f"\n🏆 BEST RESULT")
        print(f"   Feature: {best_result['feature']}")
        print(f"   Model: {best_result['model_type']}")
        print(f"   Score: {best_result['score']:.6f}")
        print(f"   Improvement: {best_result['improvement']:.6f} ({best_result['improvement']/self.baseline_score*100:.2f}%)")
        
        # Top features by average improvement
        print(f"\n📈 TOP FEATURES BY AVERAGE IMPROVEMENT")
        top_features = self.get_top_features(10)
        for i, feature in enumerate(top_features[:5], 1):
            avg_imp = np.mean(self.feature_importance[feature])
            print(f"   {i}. {feature}: {avg_imp:.6f} avg improvement")
            
        # Model-specific insights
        print(f"\n🎯 MODEL-SPECIFIC INSIGHTS")
        for model_type in ['XGBoost', 'LightGBM', 'Neural Network']:
            model_results = [r for r in self.results if r['model_type'] == model_type]
            if model_results:
                best_for_model = max(model_results, key=lambda x: x['improvement'])
                print(f"   {model_type}: Best with {best_for_model['feature']} (+{best_for_model['improvement']:.6f})")
                
        # Key insights
        if self.insights:
            print(f"\n💡 KEY INSIGHTS")
            for insight in self.insights[-5:]:  # Show last 5 insights
                print(f"   {insight}")

# ===== Bucket Transformer =====
class BucketTransformer:
    """Transform features into 3 buckets based on quantiles"""
    
    def __init__(self):
        self.thresholds = {}
        
    def fit(self, X: pd.DataFrame, features: List[str]):
        """Fit bucket thresholds based on data distribution"""
        for feature in features:
            if feature in X.columns:
                # Calculate tertile thresholds
                q33 = X[feature].quantile(0.33)
                q67 = X[feature].quantile(0.67)
                self.thresholds[feature] = (q33, q67)
                
    def transform(self, X: pd.DataFrame, feature: str) -> pd.Series:
        """Transform a single feature into 3 buckets"""
        if feature not in self.thresholds:
            raise ValueError(f"Feature {feature} not fitted")
            
        q33, q67 = self.thresholds[feature]
        
        # Create buckets: 1, 2, 3
        buckets = pd.Series(2, index=X.index, dtype=np.int8)  # Default to middle bucket
        buckets[X[feature] <= q33] = 1
        buckets[X[feature] > q67] = 3
        
        return buckets

# ===== Neural Network Models =====
def create_balanced_nn(input_dim: int) -> keras.Model:
    """Create a balanced neural network"""
    model = keras.Sequential([
        layers.Input(shape=(input_dim,)),
        layers.Dense(128, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        layers.Dense(64, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.2),
        layers.Dense(32, activation='relu'),
        layers.Dense(1)
    ])
    
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='mse'
    )
    
    return model

def create_wide_nn(input_dim: int) -> keras.Model:
    """Create a wide neural network"""
    model = keras.Sequential([
        layers.Input(shape=(input_dim,)),
        layers.Dense(256, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        layers.Dense(128, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.2),
        layers.Dense(1)
    ])
    
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='mse'
    )
    
    return model

# ===== VAE Implementation =====
class VAE(keras.Model):
    """Variational Autoencoder for feature reduction"""
    
    def __init__(self, input_dim: int, latent_dim: int = 10):
        super(VAE, self).__init__()
        self.latent_dim = latent_dim
        
        # Encoder
        self.encoder = keras.Sequential([
            layers.Input(shape=(input_dim,)),
            layers.Dense(64, activation='relu'),
            layers.Dense(32, activation='relu')
        ])
        
        self.z_mean = layers.Dense(latent_dim)
        self.z_log_var = layers.Dense(latent_dim)
        
        # Decoder
        self.decoder = keras.Sequential([
            layers.Input(shape=(latent_dim,)),
            layers.Dense(32, activation='relu'),
            layers.Dense(64, activation='relu'),
            layers.Dense(input_dim, activation='sigmoid')
        ])
        
    def encode(self, x):
        h = self.encoder(x)
        return self.z_mean(h), self.z_log_var(h)
    
    def reparameterize(self, z_mean, z_log_var):
        eps = tf.random.normal(shape=tf.shape(z_mean))
        return z_mean + tf.exp(0.5 * z_log_var) * eps
    
    def decode(self, z):
        return self.decoder(z)
    
    def call(self, inputs):
        z_mean, z_log_var = self.encode(inputs)
        z = self.reparameterize(z_mean, z_log_var)
        reconstructed = self.decode(z)
        
        # Add KL divergence loss
        kl_loss = -0.5 * tf.reduce_mean(
            1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var)
        )
        self.add_loss(kl_loss)
        
        return reconstructed
    
    def get_latent_features(self, x):
        """Get latent space representation"""
        z_mean, _ = self.encode(x)
        return z_mean

# ===== Model Training Functions =====
def train_model_cv(X: pd.DataFrame, y: pd.Series, model_type: str, features: List[str]) -> Tuple[float, np.ndarray, np.ndarray]:
    """Train model with cross-validation"""
    kf = KFold(n_splits=Config.N_FOLDS, shuffle=False)
    oof_preds = np.zeros(len(X))
    cv_scores = []
    
    X_features = X[features].values
    
    for fold, (train_idx, valid_idx) in enumerate(kf.split(X)):
        X_train, X_valid = X_features[train_idx], X_features[valid_idx]
        y_train, y_valid = y.iloc[train_idx], y.iloc[valid_idx]
        
        if model_type == "XGBoost":
            model = XGBRegressor(**XGB_PARAMS)
            model.fit(X_train, y_train, eval_set=[(X_valid, y_valid)], verbose=False)
            valid_preds = model.predict(X_valid)
            
        elif model_type == "LightGBM":
            model = LGBMRegressor(**LGBM_PARAMS)
            model.fit(X_train, y_train, eval_set=[(X_valid, y_valid)], callbacks=[])
            valid_preds = model.predict(X_valid)
            
        elif model_type in ["Balanced NN", "Wide NN"]:
            # Scale features for neural networks
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_valid_scaled = scaler.transform(X_valid)
            
            if model_type == "Balanced NN":
                model = create_balanced_nn(X_train.shape[1])
            else:
                model = create_wide_nn(X_train.shape[1])
                
            # Train with early stopping
            model.fit(
                X_train_scaled, y_train,
                validation_data=(X_valid_scaled, y_valid),
                epochs=50,
                batch_size=256,
                callbacks=[
                    keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True)
                ],
                verbose=0
            )
            
            valid_preds = model.predict(X_valid_scaled).flatten()
            
        oof_preds[valid_idx] = valid_preds
        fold_score = pearsonr(y_valid, valid_preds)[0]
        cv_scores.append(fold_score)
    
    mean_score = np.mean(cv_scores)
    return mean_score, oof_preds, np.array(cv_scores)

# ===== Feature Exploration System =====
class FeatureExplorer:
    """Systematically explore feature additions"""
    
    def __init__(self, train_df: pd.DataFrame, test_df: pd.DataFrame):
        self.train_df = train_df
        self.test_df = test_df
        self.tracker = PerformanceTracker()
        self.bucket_transformer = BucketTransformer()
        self.baseline_features = Config.CORE_FEATURES + Config.ENGINEERED_FEATURES
        
        # Get all X features not in core features
        all_x_features = [col for col in train_df.columns if col.startswith('X') and col not in Config.CORE_FEATURES]
        self.candidate_features = sorted(all_x_features)
        
        print(f"🔍 FEATURE EXPLORATION SETUP")
        print(f"   Baseline features: {len(self.baseline_features)}")
        print(f"   Candidate features for bucketing: {len(self.candidate_features)}")
        print("="*60)
        
    def establish_baseline(self):
        """Establish baseline performance with core + engineered features"""
        print("\n📊 ESTABLISHING BASELINE PERFORMANCE")
        print("="*60)
        
        y = self.train_df[Config.LABEL_COLUMN]
        
        for model_type in ["XGBoost", "LightGBM", "Balanced NN", "Wide NN"]:
            print(f"\nTraining {model_type} baseline...")
            start_time = time.time()
            
            score, _, _ = train_model_cv(self.train_df, y, model_type, self.baseline_features)
            
            elapsed = time.time() - start_time
            print(f"   Score: {score:.6f} (took {elapsed:.1f}s)")
            
            if model_type == "XGBoost":  # Use XGBoost as primary baseline
                self.tracker.record_baseline(score, model_type)
                self.baseline_score = score
                
    def explore_single_features(self, n_features: int = 5):
        """Test adding single bucketed features"""
        print(f"\n🔬 EXPLORING TOP {n_features} BUCKETED FEATURES")
        print("="*60)
        
        # Fit bucket transformer on all candidate features
        print("Fitting bucket transformer...")
        self.bucket_transformer.fit(self.train_df, self.candidate_features)
        
        y = self.train_df[Config.LABEL_COLUMN]
        
        # Test each feature
        feature_scores = []
        
        for i, feature in enumerate(self.candidate_features[:n_features]):
            print(f"\n[{i+1}/{n_features}] Testing feature: {feature}")
            
            # Create bucketed feature
            bucket_col = f"{feature}_bucket"
            self.train_df[bucket_col] = self.bucket_transformer.transform(self.train_df, feature)
            self.test_df[bucket_col] = self.bucket_transformer.transform(self.test_df, feature)
            
            # Test with each model type
            test_features = self.baseline_features + [bucket_col]
            
            for model_type in ["XGBoost", "LightGBM"]:  # Skip NNs for speed in exploration
                score, _, _ = train_model_cv(self.train_df, y, model_type, test_features)
                improvement = score - self.baseline_score
                
                print(f"   {model_type}: {score:.6f} (improvement: {improvement:+.6f})")
                
                self.tracker.record_feature_test(bucket_col, score, model_type, improvement)
                feature_scores.append((feature, score, improvement))
            
            # Clean up
            self.train_df.drop(columns=[bucket_col], inplace=True)
            self.test_df.drop(columns=[bucket_col], inplace=True)
            
        # Print interim summary
        self.tracker.print_summary()
        
        return feature_scores
    
    def apply_vae_reduction(self, n_top_features: int = 10, latent_dim: int = 5):
        """Apply VAE to top bucketed features"""
        print(f"\n🧬 APPLYING VAE DIMENSIONALITY REDUCTION")
        print(f"   Top features: {n_top_features}")
        print(f"   Latent dimensions: {latent_dim}")
        print("="*60)
        
        # Get top performing features
        top_features = self.tracker.get_top_features(n_top_features)
        
        if not top_features:
            print("No features to reduce. Skipping VAE.")
            return
            
        # Create bucketed versions of top features
        bucketed_features = []
        for feature in top_features:
            base_feature = feature.replace('_bucket', '')
            if base_feature in self.candidate_features:
                bucket_col = f"{base_feature}_bucket"
                self.train_df[bucket_col] = self.bucket_transformer.transform(self.train_df, base_feature)
                self.test_df[bucket_col] = self.bucket_transformer.transform(self.test_df, base_feature)
                bucketed_features.append(bucket_col)
        
        if not bucketed_features:
            print("No valid features for VAE. Skipping.")
            return
            
        print(f"Created {len(bucketed_features)} bucketed features for VAE")
        
        # Prepare data for VAE
        X_vae_train = self.train_df[bucketed_features].values.astype(np.float32)
        X_vae_test = self.test_df[bucketed_features].values.astype(np.float32)
        
        # Normalize for VAE
        scaler = StandardScaler()
        X_vae_train_scaled = scaler.fit_transform(X_vae_train)
        X_vae_test_scaled = scaler.transform(X_vae_test)
        
        # Train VAE
        print("Training VAE...")
        vae = VAE(input_dim=len(bucketed_features), latent_dim=latent_dim)
        vae.compile(optimizer='adam', loss='mse')
        
        vae.fit(
            X_vae_train_scaled, X_vae_train_scaled,
            epochs=50,
            batch_size=256,
            validation_split=0.1,
            verbose=0
        )
        
        # Get latent features
        latent_train = vae.get_latent_features(X_vae_train_scaled).numpy()
        latent_test = vae.get_latent_features(X_vae_test_scaled).numpy()
        
        # Add latent features to dataframes
        for i in range(latent_dim):
            col_name = f'vae_latent_{i}'
            self.train_df[col_name] = latent_train[:, i]
            self.test_df[col_name] = latent_test[:, i]
        
        vae_features = [f'vae_latent_{i}' for i in range(latent_dim)]
        
        # Test performance with VAE features
        print("\nTesting models with VAE features...")
        y = self.train_df[Config.LABEL_COLUMN]
        
        test_features = self.baseline_features + vae_features
        
        for model_type in ["XGBoost", "LightGBM", "Balanced NN", "Wide NN"]:
            score, _, _ = train_model_cv(self.train_df, y, model_type, test_features)
            improvement = score - self.baseline_score
            
            print(f"   {model_type} with VAE: {score:.6f} (improvement: {improvement:+.6f})")
            
            self.tracker.record_feature_test("VAE_features", score, model_type, improvement)
        
        # Clean up bucketed features but keep VAE features for final model
        for col in bucketed_features:
            self.train_df.drop(columns=[col], inplace=True)
            self.test_df.drop(columns=[col], inplace=True)
            
        return vae_features

# ===== Main Pipeline =====
def main():
    """Main execution pipeline"""
    print("🚀 STREAMLINED CRYPTO PREDICTION PIPELINE")
    print("="*80)
    print("📋 PIPELINE OVERVIEW:")
    print("   1. Load data with core + engineered features")
    print("   2. Establish baseline performance")
    print("   3. Explore single bucketed feature additions")
    print("   4. Apply VAE to top features")
    print("   5. Generate final predictions")
    print("="*80)
    
    # Load data
    print("\n📁 LOADING DATA...")
    train_df = pd.read_parquet(Config.TRAIN_PATH)
    test_df = pd.read_parquet(Config.TEST_PATH)
    submission_df = pd.read_csv(Config.SUBMISSION_PATH)
    
    # Apply feature engineering
    train_df = feature_engineering(train_df)
    test_df = feature_engineering(test_df)
    
    print(f"   Train shape: {train_df.shape}")
    print(f"   Test shape: {test_df.shape}")
    
    # Initialize feature explorer
    explorer = FeatureExplorer(train_df, test_df)
    
    # Step 1: Establish baseline
    explorer.establish_baseline()
    
    # Step 2: Explore single features
    feature_scores = explorer.explore_single_features(n_features=5)
    
    # Step 3: Apply VAE to top features
    vae_features = explorer.apply_vae_reduction(n_top_features=10, latent_dim=5)
    
    # Step 4: Final performance summary
    print("\n" + "="*80)
    print("🏁 FINAL PERFORMANCE SUMMARY")
    print("="*80)
    
    explorer.tracker.print_summary()
    
    # Generate recommendations
    print("\n💡 RECOMMENDATIONS")
    print("="*60)
    
    top_features = explorer.tracker.get_top_features(3)
    if top_features:
        print(f"1. Top performing bucketed features:")
        for i, feat in enumerate(top_features, 1):
            print(f"   {i}. {feat}")
    
    if vae_features:
        print(f"\n2. VAE reduced {len(top_features)} features to {len(vae_features)} latent dimensions")
        print("   Consider using VAE features for production if improvement is significant")
    
    print("\n3. Next steps:")
    print("   • Test ensemble methods combining best models")
    print("   • Explore interaction features between top buckets")
    print("   • Fine-tune hyperparameters for best model configuration")
    print("   • Consider feature selection within buckets")
    
    # Create final submission with best configuration
    print("\n📝 CREATING FINAL SUBMISSION...")
    
    # Use best performing configuration
    best_result = max(explorer.tracker.results, key=lambda x: x['score'])
    print(f"Using configuration: {best_result['model_type']} with {best_result['feature']}")
    
    # Train final model on full data
    if vae_features and "VAE" in best_result['feature']:
        final_features = explorer.baseline_features + vae_features
    else:
        final_features = explorer.baseline_features
    
    y = train_df[Config.LABEL_COLUMN]
    
    if best_result['model_type'] == "XGBoost":
        final_model = XGBRegressor(**XGB_PARAMS)
        final_model.fit(train_df[final_features].values, y)
        predictions = final_model.predict(test_df[final_features].values)
    elif best_result['model_type'] == "LightGBM":
        final_model = LGBMRegressor(**LGBM_PARAMS)
        final_model.fit(train_df[final_features].values, y)
        predictions = final_model.predict(test_df[final_features].values)
    
    # Create submission
    submission_df['prediction'] = predictions
    submission_df.to_csv('submission_optimized_bucketing.csv', index=False)
    print("✅ Submission saved to 'submission_optimized.csv'")
    
    print("\n🎯 PIPELINE COMPLETED SUCCESSFULLY!")

if __name__ == "__main__":
    main()