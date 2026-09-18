import pandas as pd
import numpy as np

LIFESPAN_REQUIREMENT = 5.0


def initial_filtering(filename):
    df_events = pd.read_csv(filename, low_memory=False) #import the events csv file

    start_dt = pd.to_datetime(df_events["startDate"], utc=True, format="ISO8601", errors="coerce")
    end_dt = pd.to_datetime(df_events["endDate"], utc=True, format="ISO8601", errors="coerce")

    df_events["lifespan_days"] = (end_dt - start_dt).dt.total_seconds() / 86400.0 #add new column lifespan
    df_events["end_year"] = end_dt.dt.year #add new column end_year

    time_mask = (df_events["lifespan_days"] >= LIFESPAN_REQUIREMENT) & (df_events["end_year"] == 2025) #we want to filter out events that had a lifetime of less than 5 days and that didn't end in 2025
    df_filtered = df_events[time_mask].copy()

    crypto_mask = df_filtered["tags"].str.contains("Crypto|Bitcoin|BTC|Ethereum|ETH|Solana|SOL|Dogecoin|DOGE|XRP|Binance|Coinbase|DeFi|NFT|Airdrop|Blockchain|Stablecoin|ETF", case=False, na=False) #Select crypto rows
    politics_mask = df_filtered["tags"].str.contains("Politics|Election|Elections|US Politics|President|Senate|Congress|GOP|Democrat|Republican|Trump|Biden|Harris|Debate|Primary|Cabinet|Supreme Court|SCOTUS|Parliament|Geopolitics", case=False, na=False) #Select politics rows
    sports_mask = df_filtered["tags"].str.contains("Sports|NFL|NBA|NHL|MLB|Soccer|Football|Basketball|Baseball|Tennis|Golf|UFC|MMA|Formula 1|F1|Olympics|FIFA|UEFA|Premier League|NCAA|Super Bowl|World Cup", case=False, na=False) #Select sports rows

    df_filtered["category"] = ""
    df_filtered.loc[crypto_mask, "category"] = "Crypto"
    df_filtered.loc[politics_mask, "category"] = "Politics"
    df_filtered.loc[sports_mask, "category"] = "Sports"

    # 7. Keep only the essential columns required to join with the Markets file
    columns_to_keep = ["id", "lifespan_days", "category"]
    df_clean_events = df_filtered[columns_to_keep]

    crypto_sample = df_clean_events[df_clean_events["category"] == "Crypto"].sample(n=200, random_state=42)
    politics_sample = df_clean_events[df_clean_events["category"] == "Politics"].sample(n=200, random_state=42)
    
    # We need to select more Sports rows than Crypto or Politics because a lot of them get dropped in the market filtering phase. 
    sports_sample = df_clean_events[df_clean_events["category"] == "Sports"].sample(n=253, random_state=42) #with n=253 and random_state=42 we get exactly 200 sports rows.

    # Combine them back together
    df_even = pd.concat([crypto_sample, politics_sample, sports_sample]).reset_index(drop=True)

    return df_even

def market_id_merge(filename, filtered_events):
    market_columns = [
        "conditionId", #this is needed to fetch the trade data from polymarkets API
        "event_id", #this is used to connect events from polymarket_events.csv to markets in polymarket_markets.csv
        "question",
        "spread",
        "volume",
   ]

    df_markets = pd.read_csv(filename, usecols=market_columns, low_memory=False) 

    df_markets["market_volume"] = pd.to_numeric(
        df_markets["volume"], errors="coerce"
    ).fillna(0.0)

    filtered_events["id"] = filtered_events["id"].astype(str)
    df_markets["event_id"] = df_markets["event_id"].astype(str)

    matched_markets = df_markets[ 
        #we select from polymarket_markets.csv, only the markets, that have the same event id:s as the ones we chose in earlier step
        df_markets["event_id"].isin(filtered_events["id"])
    ].copy()

    valid_markets = matched_markets[ 
        #then we drop all markets with market volume = 0 (no trades)
        #we also drop those markets that don't have a conditionId value, since we wont be able to fetch its trade data.
        (matched_markets["conditionId"].notna())
        & (matched_markets["market_volume"] > 0)
    ]

    top_markets = ( #we select the market with the highest volume, because we want to captute the "main event" (high signal)
        valid_markets.sort_values(by="market_volume", ascending=False)
        .groupby("event_id")
        .first()
        .reset_index()
    )

    final_merged = filtered_events.merge(
        top_markets, left_on="id", right_on="event_id", how="inner"
    )

    final_merged = final_merged.drop(columns=["event_id"])
    final_merged = final_merged.drop(columns=["volume"])

    return final_merged


def main():
    filtered_events = initial_filtering("polymarket_events.csv")
    selected_markets = market_id_merge("polymarket_markets.csv", filtered_events)

    output_filename = "selected_markets.csv"
    selected_markets.to_csv(output_filename, index=False)
    print(selected_markets["category"].value_counts())

if __name__ == "__main__":
    main()