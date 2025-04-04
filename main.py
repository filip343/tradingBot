from app import App

def main():
    features=[("SMA",20),("SMA",50)]
    app = App()
    app.load_config("./config.json")
    data = app.load_data()

if __name__=="__main__":
    main()