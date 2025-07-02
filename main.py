from data.parquet_load import parquet_load
from features.engineering import feature_engineering
from models.nn_model import NNModel
from train.train import train_model
from evaluate.metrics import evaluation
from predict.inference import predict
import torch
from sklearn.model_selection import train_test_split

def main(mode):
    train_path = '/root/Kaggle/drw-crypto-market-prediction/input/train_sub199.parquet'
    test_path = '/root/Kaggle/drw-crypto-market-prediction/input/test_sub99.parquet'
    X, Y, valid = parquet_load(train_path, test_path)
    X_train, Y_train, X_test, Y_test, valid = feature_engineering(X, Y, valid)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = NNModel(input_size=X_train.shape[1]).to(device)

    if mode == 'train':
        train_model(model, X_train, Y_train, X_test, Y_test, device,
                    epochs=200,
                    batch_size=16,
                    lr=0.0001)

    elif mode == 'predict':
        predict(model, valid, checkpoint_path='/root/Kaggle/dl_prediction_code/checkpoints/model_epoch49_test_loss0.0117.pt', submission_template_path='/root/Kaggle/Result/sample_submission.csv', output_path='/root/Kaggle/Result/', device=device)
        preds = model(torch.tensor(valid, dtype=torch.float32).to(device)).squeeze().cpu().detach().numpy()

if __name__ == '__main__':
    main(mode = 'train')