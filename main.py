from app import App

def main():
    features=[("SMA",[20]),("SMA",[50])]
    app = App()
    app.load_config("./config.json")
    data,symbols = app.get_data(features)
    print(data[0].head())
    app.initModel("torch","Transformer",input_size=5,output_size=1)

if __name__=="__main__":
    main()