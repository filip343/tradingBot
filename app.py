from data_loader import Data_Loader
from feature_calcs import Features_Calc
from pathlib import Path
import json
import time

class App():
    def __init__(self):
        self.config = {
            "data_path":"./data",
            "service":"yfinance",
            "stock_intervals":"1d"
        }
        
    def add_features(self,data,features=[]):
        now = time.time()
        unavailable=[]
        feature_calc_obj = Features_Calc(data)
        feature_func_map= feature_calc_obj.feature_func_map
        print("Feature extracting ...")
        for feat in features:
            if isinstance(feat,tuple):
               args=feat[1] 
               feat=feat[0]
            if feat not in feature_func_map:
                unavailable.append(feat)
                continue
            feature_calc = feature_func_map[feat]
            feature_calc(*args)
        if len(unavailable)!=0:
            print(f"Some of the features not available: {unavailable}")
        print(f"Done in {time.time()-now}s")
        return data
    
    def load_data(self):
        now = time.time()
        print("Data Loading ...")
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
        print(f"Done in {time.time()-now}s")
        return data
    
    def load_config(self,config_path:str)->None:
        config_path = Path(config_path)
        if config_path.suffix !=".json":
            raise ValueError("Expected config path to point to a json file")
        config_path.parent.mkdir(parents=True,exist_ok=True)
        if not config_path.exists():
            config_path.touch(exist_ok=True)
            with open(config_path,"w") as file:
                json.dump(self.config,file)
            print(f"file not found default config written in directory: {config_path}") 
            return
        
        with open(config_path,"r") as file:
            try: 
                self.config.update(json.load(file))
                print(f"config loaded from directory: {config_path}") 
                return
            except json.JSONDecodeError as e:
                print(f"Error while loading a config file: {e}")
        with open(config_path,"w") as file:
            json.dump(self.config,file)
            print(f"default config written to directory: {config_path}")
            return

    