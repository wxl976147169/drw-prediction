import os 
import sys
import numpy as np
import pandas as pd
import torch
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from models.regressor import load_ml_regressor_model

def predict(model, valid_array, checkpoint_path,submission_template_path, output_path, device, ml_regressor_path = None):
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()
    with torch.no_grad():
        # preds = model.encode(torch.tensor(valid_array, dtype=torch.float32).to(device))
        # preds = model(torch.tensor(valid_array, dtype=torch.float32).to(device))
        _, _, preds = model(torch.tensor(valid_array, dtype=torch.float32).to(device))

    sub = pd.read_csv(submission_template_path)
    GT_pred = sub.copy()
    sub['prediction'] = preds.cpu()

    piereson_r = np.corrcoef(GT_pred['prediction'], preds.squeeze().cpu())[0, 1]
    print(f"piereson_corr_{piereson_r}")
    filename = f"pred_piereson_cor{piereson_r:0.6f}.csv"
    file_path = os.path.join(os.path.dirname(output_path), filename)
    sub.to_csv(file_path, index=False)

def predict_w_ml(model, valid_array, checkpoint_path,submission_template_path, output_path, device, ml_regressor_path = None):
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()
    with torch.no_grad():
        preds = model.encode(torch.tensor(valid_array, dtype=torch.float32).to(device))
        # preds = model(torch.tensor(valid_array, dtype=torch.float32).to(device))
    if ml_regressor_path :
        ml_regressor = load_ml_regressor_model(ml_regressor_path)
        preds = ml_regressor.predict(preds.cpu().detach().numpy())
    
    sub = pd.read_csv(submission_template_path)
    GT_pred = sub.copy()
    sub['prediction'] = preds

    piereson_r = np.corrcoef(GT_pred['prediction'], preds)[0, 1]
    print(f"piereson_corr_{piereson_r}")
    filename = f"pred_piereson_cor{piereson_r:0.6f}.csv"
    file_path = os.path.join(os.path.dirname(output_path), filename)
    sub.to_csv(file_path, index=False)