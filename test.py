import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import random
import warnings
from argparse import ArgumentParser
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from train import FeatureEngineer 

warnings.filterwarnings('ignore')

def set_seeds(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    return seed

# --- LSTM Model Definition ---
class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, layer_dim, output_dim):
        super(LSTMModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.layer_dim = layer_dim
        self.lstm = nn.LSTM(input_dim, hidden_dim, layer_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        h0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).to(x.device)
        c0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        return self.sigmoid(self.fc(out[:, -1, :]))

def create_sequences(data, target, time_steps):
    X, y = [], []
    for i in range(len(data) - time_steps):
        X.append(data[i:(i + time_steps)])
        y.append(target[i + time_steps])
    return np.array(X), np.array(y)

def main():
    parser = ArgumentParser()
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--model", type=str, required=True, choices=['LDA', 'XGB', 'RF', 'LSTM', 'DT', 'GBDT'])
    parser.add_argument("--seed", type=int, default=42)
    # Hyperparameters as strings to be evaluated or individual args
    parser.add_argument("--params", type=str, help="Dict of params, e.g. \"{'n_estimators':100, 'max_depth':5}\"")
    args = parser.parse_args()

    set_seeds(args.seed)
    
    # 1. Load Data
    df = pd.read_csv(args.data_path)
    # X = df.select_dtypes(include=[np.number]).copy()
    X = FeatureEngineer().get_processed_features(df)
    y = np.where(df['close'].shift(-1) > df['close'], 1, 0)
    
    # Align and Clean (Matching your code's double iloc)
    X = X.iloc[:-1].fillna(0)
    y = y[:-1]
    X = X.iloc[:-1].fillna(0)
    y = y[:-1]

    # 2. Split and Scale
    split = int(0.8 * len(X))
    X_tr_raw, X_te_raw = X.iloc[:split], X.iloc[split:]
    y_tr, y_te = y[:split], y[split:]

    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_tr_raw)
    X_te = scaler.transform(X_te_raw)

    # 3. Parse Parameters
    bp = eval(args.params) if args.params else {}

    # 4. Initialize and Train Model
    print(f"Training Final {args.model} with params: {bp}")
    
    if args.model == "LDA":
        model = LinearDiscriminantAnalysis()
    elif args.model == "XGB":
        model = XGBClassifier(**bp, random_state=args.seed, eval_metric='logloss')
    elif args.model == "RF":
        model = RandomForestClassifier(**bp, random_state=args.seed)
    elif args.model == "DT":
        model = DecisionTreeClassifier(**bp, random_state=args.seed)
    elif args.model == "GBDT":
        model = GradientBoostingClassifier(**bp, random_state=args.seed)
    
    if args.model != "LSTM":
        model.fit(X_tr, y_tr)
        te_p = model.predict_proba(X_te)[:, 1]
        y_te_e = y_te
    else:
        # LSTM specific training logic
        X_tr_3d, y_tr_3d = create_sequences(X_tr, y_tr, 10)
        X_te_3d, y_te_3d = create_sequences(X_te, y_te, 10)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = LSTMModel(X_tr.shape[1], bp.get('h', 32), 1, 1).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=bp.get('lr', 0.001))
        
        for epoch in range(20):
            model.train()
            optimizer.zero_grad()
            out = model(torch.FloatTensor(X_tr_3d).to(device))
            loss = nn.BCELoss()(out, torch.FloatTensor(y_tr_3d).view(-1, 1).to(device))
            loss.backward()
            optimizer.step()
        
        model.eval()
        with torch.no_grad():
            te_p = model(torch.FloatTensor(X_te_3d).to(device)).cpu().numpy().flatten()
        y_te_e = y_te_3d

    # 5. Final Metrics Table
    print("\n" + "="*75)
    print(f"{'Threshold':<10} | {'Filt Acc':<10} | {'F1 Score':<10} | {'P1 Score':<10} | {'Coverage':<10}")
    print("-" * 75)
    
    for th in [0.50, 0.52, 0.55, 0.58, 0.60, 0.62, 0.65]:
        mask = (te_p > th) | (te_p < (1 - th))
        if any(mask):
            f_preds = (te_p[mask] > 0.5).astype(int)
            acc = accuracy_score(y_te_e[mask], f_preds)
            f1 = f1_score(y_te_e[mask], f_preds)
            p1 = np.mean(f_preds)
            cov = np.sum(mask) / len(y_te_e)
            print(f"{th:<10.2f} | {acc:<10.4f} | {f1:<10.4f} | {p1:<10.2%} | {cov:<10.2%}")
        else:
            print(f"{th:<10.2f} | {'N/A':<10} | {'N/A':<10} | {'N/A':<10} | {'0.00%':<10}")
    print("="*75)

if __name__ == "__main__":
    main()