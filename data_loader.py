import yfinance as yf
import requests
import os
from dotenv import load_dotenv
import pandas as pd
from pathlib import Path

load_dotenv(".env")
API_KEY = os.getenv("API_KEY1")

class Data_Loader():
    def __init__(self,data_path):
        self.data_path = Path(data_path)
        self.alpha_vantage_base_url ='https://www.alphavantage.co/query?apikey={API_KEY}' 
        self.alpha_vantage_function_map = {
            "min":"&function=TIME_SERIES_INTRADAY&outputsize=full",
            "d":"&function=TIME_SERIES_DAILY",
            "wk":"&function=TIME_SERIES_WEEKLY_ADJUSTED",
            "mo":"&function=TIME_SERIES_MONTHLY_ADJUSTED"
        }

    def load_yahoo(self,symbol,intervals="1m",period="max"):
        (self.data_path/"yf").mkdir(parents=True,exist_ok=True)
        try:
            stock = yf.Ticker(symbol) 
            df = stock.history(period=period,intervals=intervals)
            if df.empty():
                print(f"data not found for {symbol}")

            if not isinstance(df.columns,pd.MultiIndex):
                #normal
                df.columns = pd.Index([col.split(" ")[-1].lower() for col in df.columns])
            else:
                #forex
                df.columns = pd.Index([col[0].split(" ")[-1].lower() for col in df.columns])
            df.to_csv(self.data_path /"yf"/f"{symbol}.csv")
            print(f"Data saved: {self.data_path/"yf"/f'{symbol}.csv'}")
        except Exception as e:
            print(f"Error fetching {symbol}: {e}")
            return

    def load_alphavantage(self,symbol,period,intervals="1min"):
        base_url = self.alpha_vantage_base_url
        additive = ""
        for inter in self.alpha_vantage_function_map.keys():
            if intervals.endswith(inter):
                additive=self.alpha_vantage_function_map[inter]
                break
        if not additive:
            print("given interval not found should be with ending of: min, d, wk, mo")
        url = base_url + f"&datatype=csv" + f"&symbol={symbol}"+additive
        try:
            data = requests.get(url)
            if data.status_code == 200:
                path = self.data_path / "alph"/f"{symbol}.csv"
                with open(path,"w") as f:
                    f.write(data.text)
                print(f"CSV data saved to {path}")
            else:
                print(f"Error fetching data: {data.status_code} - {data.text}")
        except Exception as e:
            print(f"Error fetching {symbol}: {e}")
