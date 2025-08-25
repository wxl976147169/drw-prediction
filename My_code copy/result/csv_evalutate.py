import pandas as pd
from scipy.stats import pearsonr
import warnings
warnings.filterwarnings("ignore")


GT_label = pd.read_csv("/root/autodl-tmp/drw-ipynb/result/sample_submission.csv")
Preds = pd.read_csv("/root/autodl-tmp/drw-ipynb/result/submission.csv")
corr, _ = pearsonr(GT_label["prediction"], Preds["prediction"])
print(f" Pearson Correlation on Test Set: {corr:.4f}")