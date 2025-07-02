import os 
import torch
from torch.utils.data import DataLoader, TensorDataset

def train_model(model, X_train, Y_train, X_test, Y_test, device, epochs=30, batch_size=16, lr=0.0001, checkpoint_path='/root/Kaggle/dl_prediction_code/checkpoints/'):
    train_ds = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(Y_train.values, dtype=torch.float32).view(-1, 1))
    test_ds = TensorDataset(torch.tensor(X_test, dtype=torch.float32), torch.tensor(Y_test.values, dtype=torch.float32).view(-1, 1))

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=0.0005)
    loss_fn = torch.nn.MSELoss()
    best_test_loss = float('inf')

    for epoch in range(epochs):
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            pred = model(xb)
            loss = loss_fn(pred, yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        model.eval()
        test_loss = 0
        with torch.no_grad():
            for xb, yb in test_loader:
                xb, yb = xb.to(device), yb.to(device)
                test_loss += loss_fn(model(xb), yb).item()

        avg_test_loss = test_loss / len(test_loader)
        print(f"Epoch {epoch+1}, Test Loss: {avg_test_loss:.4f}")


        if avg_test_loss < best_test_loss:
            best_test_loss = avg_test_loss
            if checkpoint_path:
                filename = f"model_epoch{epoch+1:02d}_test_loss{best_test_loss:.4f}.pt"
                full_path = os.path.join(os.path.dirname(checkpoint_path), filename)
                torch.save(model.state_dict(), full_path)
                print(f"Saved best model to {full_path}")