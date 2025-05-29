from app import App
import numpy as np


def main():
    features=[("target",["close"]),("SMA",[20]),("SMA",[50]),("EMA",[15]),("MACD",[]),("vol",[]),("rsi",[]),("VPR",[]),("VWAP",[]),("boll_band",[]),("lags",[["close"]])] 
    app = App()
    app.load_config("./config.json")
    data,symbols = app.get_data(features)
    columns = list(data.columns)
    print(data.head())
    data_processed = data.drop(columns=["target"]).values
    labels = data["target"].values.reshape(-1,1)
    train_loader,val_loader = app.get_data_loaders(data_processed,labels,0.995,"")
    #app.initModel("lgbm")
    app.initModel("torch","Transformer",input_size=data_processed.shape[-1]-2,output_size=1,hidden_size=128,num_layers=3)
    app.fit(train_loader,val_loader)

if __name__=="__main__":
    main()