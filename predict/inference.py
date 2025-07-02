import os 
import numpy as np
import pandas as pd
import torch

def predict(model, valid_array, checkpoint_path,submission_template_path, output_path, device):
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()
    with torch.no_grad():
        preds = model(torch.tensor(valid_array, dtype=torch.float32).to(device)).squeeze().cpu().numpy()
    sub = pd.read_csv(submission_template_path)
    GT_pred = sub
    sub['prediction'] = preds

    piereson_r = np.corrcoef(GT_pred['prediction'], preds)[0, 1]
    print(f"piereson_corr_{piereson_r}")
    filename = f"dl_pred_piereson_cor{piereson_r:0.6f}.csv"
    file_path = os.path.join(os.path.dirname(output_path), filename)
    sub.to_csv(file_path, index=False)