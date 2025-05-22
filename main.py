from app import App

def main():
    features=[("target",["close"]),("SMA",[20]),("SMA",[50]),("lags",[["close"]])]
    app = App()
    app.load_config("./config.json")
    data,symbols = app.get_data(features)
    print(data.head())
    features = data.drop(columns=["target"]).values
    labels = data["target"].values.reshape(-1,1)
    train_loader = app.get_data_loader(features,labels,"")
    #app.initModel("lgbm")
    app.initModel("torch","Transformer",input_size=features.shape[-1],output_size=1)
    app.fit(train_loader)

if __name__=="__main__":
    main()