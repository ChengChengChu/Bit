import requests
import pandas as pd
import time

def get_historical_klines(symbol, interval, start_str, end_str=None):
    # Force UTC for consistent timestamps
    start_ts = int(pd.to_datetime(start_str, utc=True).timestamp() * 1000)
    end_ts = int(pd.to_datetime(end_str, utc=True).timestamp() * 1000) if end_str else None

    url = "https://api.binance.com/api/v3/klines"
    all_data = []

    while True:
        params = {"symbol": symbol, "interval": interval, "startTime": start_ts, "limit": 1000}
        if end_ts: params["endTime"] = end_ts
        
        response = requests.get(url, params=params).json()
        if not response or len(response) == 0: break
            
        all_data.extend(response)
        
        # LOGIC FIX: Move to the next candle's open time accurately
        # Open time is response[-1][0]. We want the next one.
        last_open_time = response[-1][0]
        
        # Dynamically calculate interval in ms (e.g., 15m = 900,000ms)
        # Or simply jump to the millisecond after the last close time:
        start_ts = response[-1][6] + 1 
        
        print(f"[INFO]: Downloaded up to {pd.to_datetime(last_open_time, unit='ms')}")

        if len(response) < 1000 or (end_ts and start_ts >= end_ts):
            break
        
        time.sleep(0.1) 

    columns = ['open_time', 'open', 'high', 'low', 'close', 'volume', 
               'close_time', 'qav', 'num_trades', 'taker_base', 'taker_quote', 'ignore']
    df = pd.DataFrame(all_data, columns=columns)
    
    # Clean up types immediately
    df['open_time'] = pd.to_datetime(df['open_time'], unit='ms')
    for col in ['open', 'high', 'low', 'close', 'volume']:
        df[col] = pd.to_numeric(df[col])
        
    return df

df = get_historical_klines("BTCUSDT", "1h", "2025-01-01", "2025-07-01")
df.to_csv("btc_2025_01_2025_07_01_1h.csv", index=False)