def calc_moving_average(data,column="close",prevCandles=50):
    if column not in data.columns:
        raise ValueError(f"column: {column} is not in data")
    data[f'SMA_{prevCandles}'] = data[column].rolling(prevCandles,min_periods=1).mean()
    return data[f'SMA_{prevCandles}']
    
def calc_exp_average(data,column="close",smoothing=10):
    if column not in data.columns:
        raise ValueError(f"column: {column} is not in data")
    data[f'EMA_{smoothing}'] = data[column].ewm(span=smoothing,adjust=True).mean()
    return data[f'EMA_{smoothing}'] 

def calculate_rsi(data,column="close", period=14):
    if column not in data.columns:
        raise ValueError(f"column: {column} is not in data")
    delta = data[column].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period,min_periods=1).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period,min_periods=1).mean()
    gain = gain.replace(0,1e-6)
    loss = loss.replace(0, 1e-6)
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    data[f'rsi_{period}'] = rsi
    return rsi

def calculate_macd(data,column="close", short_window=12, long_window=26, signal_window=9):
    if column not in data.columns:
        raise ValueError(f"column: {column} is not in data")
    if f'EMA_{short_window}' not in data.columns:
        data[f'EMA_{short_window}'] = calc_exp_average(data,column,short_window)
    if f'EMA_{long_window}' not in data.columns:
        data[f'EMA_{long_window}'] = calc_exp_average(data,column,long_window)
    data['MACD'] = data[f'EMA_{short_window}'] - data[f'EMA_{long_window}']
    data['Signal'] = calc_exp_average(data,"MACD",signal_window)
    data['Histogram'] = data['MACD'] - data['Signal']
    return data[['MACD', 'Signal', 'Histogram']]

def calculate_volatility(data,column="close",window=14):
    if column not in data.columns:
        raise ValueError(f"column: {column} is not in data")
    data[f"vol_{window}"] = data[column].rolling(window=window,min_periods=1).std()
    data[f"vol_{window}"]=data[f"vol_{window}"].fillna(1e-6)
    return data[f"vol_{window}"]

def calculate_bollinger_bands(data,column="close",window=14):
    if column not in data.columns:
        raise ValueError(f"column: {column} is not in data")
    if f"SMA_{window}" not in data.columns:
        data[f"SMA_{window}"] = calc_moving_average(data,column=column,prevCandles=window)
    if f"vol_{window}" not in data.columns:
        data[f"vol_{window}"] = calculate_volatility(data,column=column,window=window)
    data["upr_band"] =data[f"SMA_{window}"] -2*data[f"vol_{window}"] 
    data["lwr_band"] =data[f"SMA_{window}"] +2*data[f"vol_{window}"] 
    data['band_width'] = (data['upr_band'] - data['lwr_band']) / (data[f'SMA_{window}']).replace(0,1e-6)
    data['%B'] = (data['close'] - data['lwr_band']) / (data['upr_band'] - data['lwr_band']).replace(0,1e-6)
    return data[["upr_band","lwr_band","band_width","%B"]]

def calculate_VWAP(data,columns=["close","volume"]):
    if len(columns) !=2:
        raise ValueError(f"wrong number of columns provided, expected 2")
    if any(col not in data.columns for col in columns):
        raise ValueError(f"columns: {columns} are not in data")
    
    data['VWAP'] = (data[columns[0]] * data[columns[1]]).cumsum() / (data[columns[1]].cumsum()).replace(0,1e-6)
    return data["VWAP"]

def calculate_VPR(data,columns=["high","low","volume"]):
    if len(columns) !=3:
        raise ValueError(f"wrong number of columns provided, expected 3")
    if any(col not in data.columns for col in columns):
        raise ValueError(f"columns: {columns} are not in data")
    data['VPR'] = data[columns[2]] / (data[columns[0]] - data[columns[1]]).replace(0,1e-6)
    return data["VPR"]

def create_lags(data,lag_features):
    if len(lag_features) ==0:
        raise ValueError(f"provided zero columns for lagging")
    if any(feat not in data.columns for feat in lag_features):
        raise ValueError(f"columns: {lag_features} are not in data")
    for feat in lag_features:
        data[feat+"_lag"] = data[feat].shift(1)
    return data[[feat+"_lag" for feat in lag_features]]

feature_func_map={
    "SMA":calc_moving_average,
    "EMA":calc_exp_average,
    "MACD":calculate_macd,
    "vol":calculate_volatility,
    "rsi":calculate_rsi,
    "VPR":calculate_VPR,
    "VWAP":calculate_VWAP,
    "boll_band":calculate_bollinger_bands,
    "lags":create_lags
}