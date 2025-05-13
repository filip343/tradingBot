from data_loader import DataLoader
from feature_calcs import FeaturesCalc
from pathlib import Path
import json
import time
from models import ModelHandler

class App():
    def __init__(self):
        self.config = {
            "data_path":"./data",
            "service":"yfinance",
            "stock_intervals":"1d"
        }
        
    def add_features(self,data,features=[]):
        now = time.time()
        feature_calc_obj = FeaturesCalc(data)
        feature_func_map= feature_calc_obj.feature_func_map
        for feat in features:
            args=[]
            if isinstance(feat,tuple):
               feat,args = feat
            if feat in feature_func_map:
                feature_calc = feature_func_map[feat]
                feature_calc(*args)
        return data
    
    def get_data(self,features):
        now = time.time()
        print("Data Loading ...")
        loader = DataLoader(self.config["data_path"])
        service = self.config["service"]
        if service not in loader.service_map.keys():
            service="yfinance"
        load_func = loader.service_map[service]
        symbols = loader.get_symbols()
        interval = self.config["stock_intervals"]
        data = []

        for symbol in symbols:
            loaded = load_func(symbol,intervals=interval,period="max")
            if loaded is not None and not loaded.empty:
                data.append(loaded)
        for i,df in enumerate(data):
            df = self.add_features(df,features)
            data[i] = df

        print(f"Done in {time.time()-now}s")
        print(f"Loaded: {len(data)} symbols")
        return (data,symbols)
    
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
    def initModel(self,modelType:str,modelName:str="",**kwargs):
        if(modelType=="lgbm"):
            self.model = ModelHandler()
            self.model.initLgbmModel(**kwargs)
        elif(modelType=="torch"):
            self.model = ModelHandler()
            self.model.initTorchModel(modelName,**kwargs)
    def fit(self,data_loader,val_loader=None):
        if hasattr(self.model,"fit"):
            self.model.fit(data_loader,val_loader)
        else:
            raise ValueError("Model does not have a fit method")
    def predict(self,X):
        if hasattr(self.model,"predict"):
            return self.model.predict(X)
        else:
            raise ValueError("Model does not have a predict method")
    