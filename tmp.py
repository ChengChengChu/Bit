import pandas as pd
import numpy as np
import optuna
import warnings
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import RobustScaler
from xgboost import XGBClassifier
from argparse import ArgumentParser

warnings.filterwarnings('ignore')

# 設定中文字體 (若環境支援) 或使用預設
plt.rcParams['font.sans-serif'] = ['Arial'] 
plt.rcParams['axes.unicode_minus'] = False

def get_pro_features(df):
    c = pd.to_numeric(df['close'], errors='coerce').ffill()
    h = pd.to_numeric(df['high'], errors='coerce').ffill()
    l = pd.to_numeric(df['low'], errors='coerce').ffill()
    v = pd.to_numeric(df['volume'], errors='coerce').ffill()
    
    new_df = pd.DataFrame()
    for w in [5, 20, 60]:
        ret = c.pct_change(w)
        new_df[f'z_ret_{w}'] = (ret - ret.rolling(200).mean()) / (ret.rolling(200).std() + 1e-9)
    
    new_df['vol_z'] = (v - v.rolling(50).mean()) / (v.rolling(50).std() + 1e-9)
    new_df['price_vol_corr'] = c.rolling(20).corr(v)
    
    delta = c.diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    new_df['rsi_14'] = gain / (gain + loss + 1e-9)
    new_df['willr_14'] = (h.rolling(14).max() - c) / (h.rolling(14).max() - l.rolling(14).min() + 1e-9)
    new_df['bias_10'] = (c - c.rolling(10).mean()) / c.rolling(10).mean()
    new_df['volatility'] = c.pct_change().rolling(20).std()
    
    return new_df.ffill().fillna(0)

def save_plots(train_res, val_res, test_res, bp):
    """
    將分析圖表拆分為 Training, Validating, Testing 三份
    """
    if not os.path.exists('outcome'):
        os.makedirs('outcome')

    datasets = [
        ('Training', train_res),
        ('Validating', val_res),
        ('Testing', test_res)
    ]

    # --- 1. 漲跌計數圖 (分成三份) ---
    fig, axes = plt.subplots(1, 3, figsize=(20, 6), sharey=False)
    for i, (name, res) in enumerate(datasets):
        target_val = 1 if bp['mode'] == 'long' else 0
        plot_df = pd.DataFrame({
            'prob': res['probs'], 
            'actual': (res['is_correct'] == True) if bp['mode'] == 'long' else (res['is_correct'] == False)
        })
        # 這裡 actual 的邏輯：如果是 SHORT 模式，is_correct=True 代表實際下跌，is_correct=False 代表實際上漲
        # 為了統一視覺：我們畫出 實際漲(UP) 與 實際跌(DOWN)
        # 假設 y=1 是漲，y=0 是跌
        
        # 重新校正實際漲跌邏輯
        # 注意：res['is_correct'] 是指「是否預測正確」
        actual_up = res['is_correct'] if bp['mode'] == 'long' else ~res['is_correct']
        
        temp_df = pd.DataFrame({'prob': res['probs'], 'is_up': actual_up})
        temp_df['prob_bin'] = pd.cut(temp_df['prob'], bins=np.linspace(0, 1, 51))
        
        bin_stats = temp_df.groupby('prob_bin').agg(
            up_count=('is_up', 'sum'),
            total_count=('is_up', 'count')
        )
        bin_stats['down_count'] = bin_stats['total_count'] - bin_stats['up_count']
        bin_stats.index = [interval.mid for interval in bin_stats.index]

        axes[i].plot(bin_stats.index, bin_stats['up_count'], label='Actual UP', color='red', alpha=0.7)
        axes[i].plot(bin_stats.index, bin_stats['down_count'], label='Actual DOWN', color='green', alpha=0.7)
        axes[i].axvspan(bp['l'], bp['u'], color='yellow', alpha=0.2, label='Trade Zone')
        axes[i].set_title(f"{name} Set: Prob vs Actual")
        axes[i].set_xlabel("Probability")
        axes[i].legend()
        axes[i].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('outcome/probability_vs_actual_split.png')
    plt.close()

    # --- 2. 時段勝率圖 (分成三份) ---
    fig, axes = plt.subplots(1, 3, figsize=(20, 6), sharey=True)
    for i, (name, res) in enumerate(datasets):
        df_hour = pd.DataFrame({
            'hour': res['times'].dt.hour,
            'correct': res['is_correct'],
            'is_trade': res['is_trade']
        })
        # 只統計有交易的時段勝率
        hourly_stats = df_hour[df_hour['is_trade']].groupby('hour')['correct'].mean()
        
        if not hourly_stats.empty:
            hourly_stats.plot(kind='bar', ax=axes[i], color='skyblue', edgecolor='black')
            axes[i].axhline(y=0.5, color='red', linestyle='--')
        
        axes[i].set_title(f"{name} Set: Hourly Win Rate")
        axes[i].set_ylim(0, 1)
        axes[i].set_xlabel("Hour")

    plt.tight_layout()
    plt.savefig('outcome/hourly_performance_split.png')
    plt.close()

    # --- 3. 原有的機率分佈圖 (維持單張疊加以便對比) ---
    plt.figure(figsize=(10, 6))
    for name, res in datasets:
        sns.kdeplot(res['probs'], label=name, fill=True, alpha=0.2)
    plt.axvline(bp['l'], color='red', linestyle='--')
    plt.axvline(bp['u'], color='green', linestyle='--')
    plt.title("Probability Distribution Consistency Check")
    plt.legend()
    plt.savefig('outcome/probability_distribution_compare.png')
    plt.close()

def main():
    parser = ArgumentParser()
    parser.add_argument("--data_path", type=str, required=True)
    args = parser.parse_args()

    df = pd.read_csv(args.data_path)
    date_col = 'open_time' if 'open_time' in df.columns else 'datetime'
    df['datetime'] = pd.to_datetime(df[date_col])
    df = df.sort_values('datetime').reset_index(drop=True)

    X_raw = get_pro_features(df)
    y = ((df['close'].shift(-1) - df['close']) > 0).astype(int).values
    
    data = X_raw.copy()
    data['target'] = y
    data['datetime'] = df['datetime']
    data = data.iloc[200:-1] 

    split_date = pd.Timestamp("2025-09-01")
    train_val_data = data[data['datetime'] < split_date]
    test_data = data[data['datetime'] >= split_date]

    n_tv = len(train_val_data)
    t_idx = int(n_tv * 0.8)
    
    # 保留時間資訊以便後續繪圖
    train_dates = train_val_data['datetime'].iloc[:t_idx]
    val_dates = train_val_data['datetime'].iloc[t_idx:]
    test_dates = test_data['datetime']

    X_train = train_val_data.drop(columns=['target', 'datetime']).values[:t_idx]
    y_train = train_val_data['target'].values[:t_idx]
    X_val = train_val_data.drop(columns=['target', 'datetime']).values[t_idx:]
    y_val = train_val_data['target'].values[t_idx:]
    X_test = test_data.drop(columns=['target', 'datetime']).values
    y_test = test_data['target'].values

    scaler = RobustScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_val_s = scaler.transform(X_val)
    X_test_s = scaler.transform(X_test)

    def objective(trial):
        params = {
            'n_estimators': trial.suggest_int('n', 80, 200),
            'max_depth': 3,
            'learning_rate': trial.suggest_float('lr', 0.005, 0.05, log=True),
            'gamma': trial.suggest_float('g', 0.1, 1.0),
            'reg_lambda': trial.suggest_float('l2', 10.0, 30.0),
            'subsample': 0.8,
            'random_state': 42,
            'n_jobs': -1
        }
        m = XGBClassifier(**params).fit(X_train_s, y_train)
        probs_v = m.predict_proba(X_val_s)[:, 1]
        
        mode = trial.suggest_categorical('mode', ['long', 'short'])
        if mode == 'long':
            l, u = trial.suggest_float('l', 0.51, 0.55), trial.suggest_float('u', 0.58, 0.85)
            target = 1
        else:
            l, u = trial.suggest_float('l', 0.20, 0.44), trial.suggest_float('u', 0.45, 0.49)
            target = 0
            
        mask_v = (probs_v >= l) & (probs_v <= u)
        coverage = np.sum(mask_v) / len(y_val)
        if not (0.15 < coverage < 0.25): return 0
        
        val_wr = np.mean(y_val[mask_v] == target)
        probs_t = m.predict_proba(X_train_s)[:, 1]
        train_wr = np.mean(y_train[(probs_t >= l) & (probs_t <= u)] == target)
        return val_wr - abs(train_wr - val_wr) * 0.6

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=60)
    
    bp = study.best_params
    final_m = XGBClassifier(
        n_estimators=bp['n'], max_depth=3, learning_rate=bp['lr'], 
        gamma=bp['g'], reg_lambda=bp['l2'], random_state=42
    ).fit(X_train_s, y_train)

    def get_eval_details(X_s, y_true, times):
        probs = final_m.predict_proba(X_s)[:, 1]
        target_val = 1 if bp['mode'] == 'long' else 0
        mask = (probs >= bp['l']) & (probs <= bp['u'])
        return {
            'probs': probs,
            'is_trade': mask,
            'is_correct': (y_true == target_val),
            'times': times
        }

    train_res = get_eval_details(X_train_s, y_train, train_dates)
    val_res = get_eval_details(X_val_s, y_val, val_dates)
    test_res = get_eval_details(X_test_s, y_test, test_dates)

    # 繪製並儲存圖表
    save_plots(train_res, val_res, test_res, bp)

    # --- 輸出報告 ---
    print(f"\n" + "="*65)
    print(f"📊 提頻優化報告 & 圖表已生成至 /outcome")
    print(f"模式: {bp['mode'].upper()} | 信心區間: {bp['l']:.3f}-{bp['u']:.3f}")
    
    tr_acc = np.mean(train_res['is_correct'][train_res['is_trade']])
    va_acc = np.mean(val_res['is_correct'][val_res['is_trade']])
    print(f"訓練集準確率: {tr_acc:.2%} | 預測比例: {np.mean(train_res['is_trade']):.2%}")
    print(f"驗證集準確率: {va_acc:.2%} | 預測比例: {np.mean(val_res['is_trade']):.2%}")
    print(f"="*65)

    test_months = test_dates.dt.month
    for m in [9, 10, 11, 12]:
        m_mask = (test_months == m).values
        if np.any(m_mask):
            m_probs = test_res['probs'][m_mask]
            m_y = y_test[m_mask]
            trade_mask = (m_probs >= bp['l']) & (m_probs <= bp['u'])
            if np.sum(trade_mask) > 0:
                m_acc = np.mean(m_y[trade_mask] == (1 if bp['mode']=='long' else 0))
                print(f"  - 2025/{m:02d} 月: 準確率 {m_acc:.2%} | 預測比例: {np.mean(trade_mask):.2%}")

if __name__ == "__main__":
    main()