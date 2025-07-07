import os 
import sys
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
import joblib
from sklearn.metrics import mean_squared_error
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from models.regressor import train_lgbm

def train_model(model, X_train, Y_train, X_test, Y_test, device, epochs=30, batch_size=16, lr=0.0001, checkpoint_path='/root/autodl-tmp/drw_code/checkpoints/'):
    train_ds = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(Y_train.values, dtype=torch.float32).view(-1, 1))
    test_ds = TensorDataset(torch.tensor(X_test, dtype=torch.float32), torch.tensor(Y_test.values, dtype=torch.float32).view(-1, 1))

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=0.0005)
    loss_fn = torch.nn.MSELoss()
    best_test_loss = float('inf')

    list_pred_enc_train = []
    list_y_train = []
    for epoch in range(epochs):
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            yb_AE = torch.cat([xb[:, :5], xb[:, -58:]], dim=1)
            pred_enc, pred_dec = model(xb)
            loss = loss_fn(pred_dec, yb_AE)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            list_pred_enc_train.append(pred_enc.cpu().detach().numpy()) # 保存编码器输出
            list_y_train.append(yb.cpu().detach().numpy()) # 保存编码器输出对应的标签
        
        X_train_ml = np.vstack(list_pred_enc_train)  
        y_train_ml = np.vstack(list_y_train)  
        ml_regressor = train_lgbm(X_train_ml, y_train_ml)
        # y_train_ml_pred = ml_regressor.predict(X_train_ml)
        # 测试
        model.eval()
        test_loss_dl = 0
        test_loss_ml = 0
        with torch.no_grad():
            for xb, yb in test_loader:
                xb, yb = xb.to(device), yb.to(device)
                yb_AE = torch.cat([xb[:, :5], xb[:, -58:]], dim=1)
                pred_enc, pred_dec = model(xb)
                test_loss_dl += loss_fn(pred_dec, yb_AE).item()
                # ML回归器
                y_test_ml = ml_regressor.predict(pred_enc.cpu().detach().numpy())
                test_loss_ml += mean_squared_error(y_test_ml, yb.cpu().detach().numpy().squeeze())

        avg_test_loss_dl = test_loss_dl / len(test_loader)
        avg_test_loss = test_loss_ml / len(test_loader)
        print(f"Epoch {epoch+1}, DL Test Loss: {avg_test_loss_dl:.4f}")
        print(f"Epoch {epoch+1}, Test Loss: {avg_test_loss:.4f}")

        if avg_test_loss < best_test_loss :
            best_test_loss = avg_test_loss
            if checkpoint_path : # and epoch >= 49:
                filename = f"model_epoch{epoch+1:02d}_dl_test_loss{avg_test_loss_dl:.4f}.pt"
                full_path = os.path.join(os.path.dirname(checkpoint_path), filename)
                torch.save(model.state_dict(), full_path)
                
                ml_filename = f"model_epoch{epoch+1:02d}_test_loss{best_test_loss:.4f}.pkl"
                ml_full_path = os.path.join(os.path.dirname(checkpoint_path), ml_filename)
                joblib.dump(ml_regressor, ml_full_path)

                print(f"Saved best model to {full_path}")