import numpy as np
import joblib
import ccxt
import time
import warnings
# from datetime import datetime, timezone

warnings.filterwarnings('ignore')

class BitcoinPredictor:
    def __init__(self, model_path="model.joblib"):
        """
        初始化：載入模型並建立持久化連線
        """
        # loading model, scaler
        self.ckpt = joblib.load(model_path)
        self.model = self.ckpt['model']
        self.scaler = self.ckpt['scaler']
        
        # construct connection to binance
        self.exchange = ccxt.binance({
            'enableRateLimit': True,
            'options': {'defaultType': 'spot'}
        })

    def get_prediction(self, symbol='BTC/USDT', since_ms=None):
        """
        Input: symbol: str 
               since_ms: default None -> fetch the last 25 klines start from now
                         specified -> fetch the last 25 klines from that moment
        Output: prediciton of model (bool)
                True: Predict price will go up
                False: Giveup predicition
        """
        # start_time = time.perf_counter()
        
        try:
            ohlcv = self.exchange.fetch_ohlcv(symbol, timeframe='1h', limit=25, since=since_ms,)
            if len(ohlcv) < 24:
                raise ValueError(f"Not enough history (24 needed): {len(ohlcv)}")

            data = np.array(ohlcv)
            closes = data[:, 4]
            volumes = data[:, 5]

            # calculate input features
            c_last_24 = closes[-24:]
            ma24 = c_last_24.mean()
            std24 = c_last_24.std()
            
            cur_c = closes[-1]
            z = (cur_c - ma24) / (std24 + 1e-9)
            r1 = (cur_c / closes[-2]) - 1
            r4 = (cur_c / closes[-5]) - 1
            vs = volumes[-1] / (volumes[-12:].mean() + 1e-9)
            
            # hour is a feature to model 
            h = int((ohlcv[-1][0] // 3600000) % 24)

            # 3. model prediciton
            x = [[z, r1, r4, vs, h]]
            x_scaled = self.scaler.transform(x)
            prob = self.model.predict_proba(x_scaled)[0, 1]
            
            # duration = time.perf_counter() - start_time
            # return {
            #     "probability": float(prob),
            #     "signal": bool(prob > 0.555),
            #     "latency_sec": duration,
            #     "timestamp": ohlcv[-1][0]
            # }
            return bool(prob > 0.555)

        except Exception as e:
            return {"error": str(e)}

# --- 使用範例 ---
if __name__ == "__main__":
    # 建立一個實例（這步會花約 0.5-1 秒）
    predictor = BitcoinPredictor("model.joblib")

    # 之後重複呼叫 get_prediction (這步通常只需 0.1-0.2 秒)
    for i in range(3):
        result = predictor.get_prediction()
        print(result)
        # if "error" in result:
        #     print(f"錯誤: {result['error']}")
        # else:
        #     print(f"第 {i+1} 次預測 - 機率: {result['probability']:.4f}, 耗時: {result['latency_sec']:.4f}s")
        time.sleep(1) # 模擬間隔
