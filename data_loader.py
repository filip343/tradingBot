import yfinance as yf
import requests
import os
from dotenv import load_dotenv
import pandas as pd
from pathlib import Path
import io

load_dotenv(".env")
API_KEY = os.getenv("API_KEY1")

class Data_Loader():
    def __init__(self,data_path:str)->None:
        self.service_map={
            "yfinance":self.load_yahoo,
            "alphavantage":self.load_alphavantage
        }
        self.data_path = Path(data_path)
        self.alpha_vantage_base_url ='https://www.alphavantage.co/query?apikey={API_KEY}' 
        self.alpha_vantage_function_map = {
            "min":"&function=TIME_SERIES_INTRADAY&outputsize=full",
            "d":"&function=TIME_SERIES_DAILY",
            "wk":"&function=TIME_SERIES_WEEKLY_ADJUSTED",
            "mo":"&function=TIME_SERIES_MONTHLY_ADJUSTED"
        }

    def load_yahoo(self,symbol:str,intervals:str="1m",period:str="max"):
        (self.data_path/"yf").mkdir(parents=True,exist_ok=True)
        try:
            stock = yf.Ticker(symbol) 
            df = stock.history(period=period,interval=intervals)
            if df.empty:
                print(f"data not found for {symbol}")

            if not isinstance(df.columns,pd.MultiIndex):
                #normal
                df.columns = pd.Index([col.split(" ")[-1].lower() for col in df.columns])
            else:
                #forex
                df.columns = pd.Index([col[0].split(" ")[-1].lower() for col in df.columns])
            df.index = pd.to_datetime(df.index)
            df.to_csv(self.data_path /"yf"/f"{symbol}.csv")
            print(f"Data saved: {self.data_path/"yf"/f'{symbol}.csv'}")
        except Exception as e:
            print(f"Error fetching {symbol}: {e}")
        return df

    def load_alphavantage(self,symbol:str,intervals:str="1min",period:str="max"):
        base_url = self.alpha_vantage_base_url
        additive = ""
        for inter in self.alpha_vantage_function_map.keys():
            if intervals.endswith(inter):
                additive=self.alpha_vantage_function_map[inter]
                if inter=="min":
                    additive+=f"&interval={intervals}"
                break
        if not additive:
            raise ValueError("given interval not found should be with ending of: min, d, wk, mo")
        url = base_url + f"&datatype=csv" + f"&symbol={symbol}"+additive
        try:
            data = requests.get(url)
            if data.status_code == 200:
                path = self.data_path / "alph"/f"{symbol}.csv"
                df = pd.read_csv(io.StringIO(data.text),index_col=0,parse_dates=True)
                df.to_csv(path,index=False)
                print(f"CSV data saved to {path}")
            else:
                print(f"Error fetching data: {data.status_code} - {data.text}")
        except Exception as e:
            print(f"Error fetching {symbol}: {e}")
        return df
    def get_symbols(self):
        url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
        df = pd.read_html(url, header=0)[0]

        sp500_tickers = df["Symbol"].tolist()
        return sp500_tickers