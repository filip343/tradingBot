from app import App

def main():
    app = App()
    app.load_config("./config.json")
    data = app.load_data()

if __name__=="__main__":
    main()