from data_loader import Data_Loader
from feature_calcs import feature_func_map
from pathlib import Path
import json

class App():
    def __init__(self):
        self.config = {
            "data_path":"./data",
            "service":"yfinance",
            "stock_intervals":"1d"
        }
    def preprocess_data(self,data,features=[]):
        unavailable=[]
        for feat in features:
            if feat not in feature_func_map:
                unavailable.append(feat)
                continue
            feature_calc = feature_func_map[feat]
            
        pass
    def load_data(self):
        loader = Data_Loader(self.config["data_path"])
        service = self.config["service"]
        if service not in loader.service_map.keys():
            service="yfinance"
        load_func = loader.service_map[service]
        symbols = loader.get_symbols()
        interval = self.config["stock_intervals"]
        data = []
        for symbol in symbols:
            data.append(load_func(symbol,intervals=interval,period="max"))
        return data
    def load_config(self,config_path:str)->None:
        config_path = Path(config_path)
        if config_path.suffix !=".json":
            raise ValueError("Expected config path to point to a json file")
        config_path.parent.mkdir(parents=True,exist_ok=True)
        if not config_path.exists():
            config_path.touch(exist_ok=True)
            with open(config_path,"r") as file:
                json.dump(file,self.config)
            print(f"file not found default config written in directory: {config_path}") 
        
        with open(config_path,"r") as file:
            self.config.update(json.load(file))
        print(f"config loaded from directory: {config_path}") 
    