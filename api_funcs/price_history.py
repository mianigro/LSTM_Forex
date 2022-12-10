# Import Python modules
import json
import requests
import time
from datetime import datetime, timedelta
from dotenv import load_dotenv
import os

# Third party imports
import numpy as np
import pandas as pd

# This generates price data history based on arguments and returns the dataframe

# API Key and details
load_dotenv()
API_KEY = os.getenv("API_KEY")
ACCOUNT_ID = os.getenv("ACCOUNT_ID")


# Generate dataframe of prices
def make_price_csv_many(gran_candles, days, instrument, use_local):
    if use_local == False:
        # Set API path and build the request
        CANDLES_PATH = f"https://api-fxpractice.oanda.com/v3/accounts/{ACCOUNT_ID}/instruments/{instrument}/candles"
        header = {"Authorization": "Bearer " + API_KEY}

        # Set timeframes based on initial argument
        from_time = time.mktime(pd.to_datetime(datetime.now() - timedelta(days=days)).timetuple())
        to_time = time.mktime(pd.to_datetime(datetime.now()).timetuple())
        
        # Builds and executes the query
        query = {"from": str(from_time), "to": str(to_time), "granularity": gran_candles}
        response = requests.get(CANDLES_PATH, headers=header, params=query)

        # Processes the response data
        data = response.json()["candles"]
        
        # Process the request and extract relevant data into lists
        best_ave = []
        high_list = []
        low_list = []
        close_list = []
        open_list = []
        volatile_list = []

        for candle in data:
            dec_test = str(candle["mid"]["o"]).split(".")[1]
            dec_pl = len(dec_test)

            open_in = float(candle["mid"]["o"])
            close_in = float(candle["mid"]["c"])
            low_in = float(candle["mid"]["l"])
            high_in = float(candle["mid"]["h"])

            best_ave.append(round(((low_in + high_in) / 2), dec_pl))
            high_list.append(round(high_in, dec_pl))
            low_list.append(round(low_in, dec_pl))
            close_list.append(round(close_in, dec_pl))
            open_list.append(round(open_in, dec_pl))
            volatile_list.append(round((high_in-low_in), dec_pl))

        # Turn the lists into a dataframe and format it
        df = pd.DataFrame({"high": best_ave, "high_high": high_list, "low": low_list, "close": close_list, "open": open_list, "volatile": volatile_list})
        df.index.name = "time"

        df.to_csv(f"{gran_candles}_{instrument}.csv")
    
    else:
        df = pd.read_csv(f"{gran_candles}_{instrument}.csv", index_col="time")

    return df


def make_price_csv_high(gran_candles, days, instrument, use_local):
    if use_local == False:
        # Set API path and build the request
        CANDLES_PATH = f"https://api-fxpractice.oanda.com/v3/accounts/{ACCOUNT_ID}/instruments/{instrument}/candles"
        header = {"Authorization": "Bearer " + API_KEY}

        # Set timeframes based on initial argument
        from_time = time.mktime(pd.to_datetime(datetime.now() - timedelta(days=days)).timetuple())
        to_time = time.mktime(pd.to_datetime(datetime.now()).timetuple())
        
        # Builds and executes the query
        query = {"from": str(from_time), "to": str(to_time), "granularity": gran_candles}
        response = requests.get(CANDLES_PATH, headers=header, params=query)

        # Processes the response data
        data = response.json()["candles"]
        
        # Process the request and extract relevant data into lists
        best_ave = []

        for candle in data:
            dec_test = str(candle["mid"]["o"]).split(".")[1]
            dec_pl = len(dec_test)

            low_in = float(candle["mid"]["l"])
            high_in = float(candle["mid"]["h"])

            best_ave.append(round(((low_in + high_in) / 2), dec_pl))

        # Turn the lists into a dataframe and format it
        df = pd.DataFrame({"high": best_ave})
        df.index.name = "time"

        df.to_csv(f"{gran_candles}_{instrument}.csv")
    
    else:
        df = pd.read_csv(f"{gran_candles}_{instrument}.csv", index_col="time")

    return df
