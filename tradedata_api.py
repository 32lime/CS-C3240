import pandas as pd
import numpy as np
import requests
import math
import time

api_url = "https://data-api.polymarket.com/trades"



def trade_api_call(df, cid, max_retries, max_trades):
    trades = []
    offset = 0
    limit = 500

    while len(trades) < max_trades:
        params = {
                "market": cid,
                "limit": 500,  #Fetch 500 most recent trades for the market. This is the page limit for this API 
                "offset": offset
            }
        success = False
        attempt = 0

        while attempt < max_retries:
            try:
                response = requests.get(api_url, params=params, timeout=10)

                if response.status_code == 200:
                    fetched_trades = response.json()
                    trades.extend(fetched_trades)
                    success = True

                    if len(fetched_trades) < limit:
                        break

                    offset += limit
                    time.sleep(0.1)

                elif response.status_code == 429: #too many requests, wait for a while and try again
                    wait_time = (attempt + 1) * 2
                    time.sleep(wait_time)
                    attempt += 1

                else:
                    break        

            except requests.exceptions.RequestException as e:  
                time.sleep(2)
                attempt +=1

        if not success or (len(trades) > 0 and len(fetched_trades) < limit):
            break        

    calculate_features(df, cid, trades)




def calculate_features(df, cid, trades_list):
    n = len(trades_list)

    if n < 2:
        return

    newest = int(trades_list[0]['timestamp'])
    oldest = int(trades_list[-1]['timestamp'])

    duration_seconds = newest - oldest
    duration_hours = duration_seconds / 3600.0

    if duration_hours > 0:
        trades_per_hour = len(trades_list) / duration_hours
    else:
        trades_per_hour = len(trades_list)    

    prices = []
    trade_usd_volumes = []

    for trade in trades_list:
        raw_price = float(trade['price'])
        size = float(trade['size'])

        outcome = trade.get('outcome', '').upper()

        if outcome == 'NO':
            true_price = 1.0 - raw_price
        else:
            true_price = raw_price

        prices.append(true_price)
        trade_usd_volumes.append(raw_price*size)

    
    
    avg_trade_size = np.mean(trade_usd_volumes)
    trade_size_sd = np.std(trade_usd_volumes)

    prices_sd = np.std(prices)


    df.loc[df['conditionId'] == cid, 'trades_per_hour_last_2500_trades'] = trades_per_hour
    df.loc[df['conditionId'] == cid, 'trade_size_avg_usd_last_2500'] = avg_trade_size
    df.loc[df['conditionId'] == cid, 'trade_size_std_usd_last_2500'] = trade_size_sd
    df.loc[df['conditionId'] == cid, 'price_volatility_last_2500_trades'] = prices_sd

  


def main():
    df = pd.read_csv("selected_markets.csv", low_memory=False)

    df['trade_size_avg_usd_last_2500'] = np.nan
    df['trade_size_std_usd_last_2500'] = np.nan
    df['price_volatility_last_2500_trades'] = np.nan
    df['trades_per_hour_last_2500_trades'] = np.nan
 

    for market in df.itertuples():
        cid = market.conditionId
        trade_api_call(df, cid, 3, 2500)
        time.sleep(0.15)

    # clean up column names
    df = df.rename(columns={
    "market_volume": "market_total_volume",
    "spread": "market_spread"
    })


    #this drops all columns except for the ones listed
    final_column_order = [
        "category",
        "lifespan_days",
        "market_total_volume",
        "market_spread",
        "trade_size_avg_usd_last_2500",
        "trade_size_std_usd_last_2500",
        "trades_per_hour_last_2500_trades",
        "price_volatility_last_2500_trades",
    ]

    df = df[final_column_order]

    
    output_filename = "polymarket_ml.csv"
    df.to_csv(output_filename, index=False)

        
if __name__ == "__main__":
    main()