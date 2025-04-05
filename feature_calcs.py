class FeaturesCalc():
    def __init__(self,data,price_col="close",vol_col="volume",low_col="low",high_col="high"):
        self.feature_func_map={
            "SMA":self.calc_moving_average,
            "EMA":self.calc_exp_average,
            "MACD":self.calculate_macd,
            "vol":self.calculate_volatility,
            "rsi":self.calculate_rsi,
            "VPR":self.calculate_VPR,
            "VWAP":self.calculate_VWAP,
            "boll_band":self.calculate_bollinger_bands,
            "lags":self.create_lags
        }
        self.data=data
        self.price_col = price_col
        self.vol_col=vol_col
        self.low_col=low_col
        self.high_col=high_col
        data_col=data.columns
        for col in [price_col,vol_col,low_col,high_col]:
            if col not in data_col:
                raise ValueError(f"column: {self.price_col} is not in data")


    def calc_moving_average(self,prevCandles=50):
        column = self.price_col
        data = self.data
        if column not in data.columns:
            raise ValueError(f"column: {column} is not in data")
        data[f'SMA_{prevCandles}'] = data[column].rolling(prevCandles,min_periods=1).mean()
        return data[f'SMA_{prevCandles}']

    def calc_exp_average(self,smoothing=10):
        column = self.price_col
        data = self.data
        data[f'EMA_{smoothing}'] = data[column].ewm(span=smoothing,adjust=True).mean()
        return data[f'EMA_{smoothing}'] 

    def calculate_rsi(self,period=14):
        column = self.price_col
        data = self.data

        delta = data[column].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period,min_periods=1).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period,min_periods=1).mean()
        gain = gain.replace(0,1e-6)
        loss = loss.replace(0, 1e-6)
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        data[f'rsi_{period}'] = rsi
        return rsi

    def calculate_macd(self,short_window=12, long_window=26, signal_window=9):
        column = self.price_col
        data = self.data
        if f'EMA_{short_window}' not in data.columns:
            data[f'EMA_{short_window}'] = self.calc_exp_average(short_window)
        if f'EMA_{long_window}' not in data.columns:
            data[f'EMA_{long_window}'] = self.calc_exp_average(long_window)
        data['MACD'] = data[f'EMA_{short_window}'] - data[f'EMA_{long_window}']
        data['Signal'] = data["MACD"].ewm(span=signal_window,adjust=True).mean()
        data['Histogram'] = data['MACD'] - data['Signal']
        return data[['MACD', 'Signal', 'Histogram']]

    def calculate_volatility(self,window=14):
        column = self.price_col
        data = self.data
        data[f"vol_{window}"] = data[column].rolling(window=window,min_periods=1).std()
        data[f"vol_{window}"]=data[f"vol_{window}"].fillna(1e-6)
        return data[f"vol_{window}"]

    def calculate_bollinger_bands(self,window=14):
        column = self.price_col
        data = self.data
        if f"SMA_{window}" not in data.columns:
            data[f"SMA_{window}"] = self.calc_moving_average(prevCandles=window)
        if f"vol_{window}" not in data.columns:
            data[f"vol_{window}"] = self.calculate_volatility(window=window)
        data["upr_band"] =data[f"SMA_{window}"] +2*data[f"vol_{window}"] 
        data["lwr_band"] =data[f"SMA_{window}"] -2*data[f"vol_{window}"] 
        data['band_width'] = (data['upr_band'] - data['lwr_band']) / (data[f'SMA_{window}']).replace(0,1e-6)
        data['%B'] = (data['close'] - data['lwr_band']) / (data['upr_band'] - data['lwr_band']).replace(0,1e-6)
        return data[["upr_band","lwr_band","band_width","%B"]]

    def calculate_VWAP(self):
        columns=[self.price_col,self.vol_col]
        data=self.data
        data['VWAP'] = (data[columns[0]] * data[columns[1]]).cumsum() / (data[columns[1]].cumsum()).replace(0,1e-6)
        return data["VWAP"]

    def calculate_VPR(self):
        columns=[self.high_col,self.low_col,self.vol_col]
        data = self.data
        data['VPR'] = data[columns[2]] / (data[columns[0]] - data[columns[1]]).replace(0,1e-6)
        return data["VPR"]

    def create_lags(self,lag_features):
        data=self.data
        if len(lag_features) ==0:
            raise ValueError(f"provided zero columns for lagging")
        if any(feat not in data.columns for feat in lag_features):
            raise ValueError(f"columns: {lag_features} are not in data")
        for feat in lag_features:
            data[feat+"_lag"] = data[feat].shift(1)
        return data[[feat+"_lag" for feat in lag_features]]
