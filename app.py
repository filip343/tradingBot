from data_loader import DataLoader
from feature_calcs import FeaturesCalc
from pathlib import Path
import json
import time
from models import ModelHandler
import torch
import numpy as np
import pandas as pd
from dataset import Dataset
class App():
    def __init__(self):
        self.config = {
            "data_path":"./data",
            "service":"yfinance",
            "stock_intervals":"1d",
            "batch_size":64,
            "num_workers":4,
            "lr":1e-3,
            "max_epochs":4,
            "time_size":10,
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
        data=data.dropna()
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
        num_loaded_symbols = 0

        for symbol in symbols:
            loaded = load_func(symbol,intervals=interval,period="max")
            if loaded is not None and not loaded.empty:
                symbol_col = pd.Series(np.ones(len(loaded))*symbols.index(symbol),name="symbol")
                loaded = pd.concat([loaded.reset_index(), symbol_col], axis=1)
                data.append(loaded)
                num_loaded_symbols+=1
        for i,df in enumerate(data):
            df = self.add_features(df,features)
            data[i] = df
        data = pd.concat(data)
        data["id"] = np.arange(len(data))
        columns = list(data.columns)
        columns.remove("symbol")
        columns.remove("id")
        data = data.reindex(columns=["id","symbol"]+columns)
        data.drop(columns=["Date","open"],inplace=True)
        print(f"Done in {time.time()-now}s")
        print(f"Loaded: {num_loaded_symbols} symbols")
        return (data,symbols)

    def get_data_loader(self,data,labels,dataset_type=None):
        data = torch.tensor(data,dtype=torch.float32)
        labels = torch.tensor(labels,dtype=torch.float32)
        if dataset_type=="single_time" :
            dataset = torch.utils.data.TensorDataset(data,labels)
        else:
            dataset = Dataset(data,labels,time_size=self.config["time_size"])
        data_loader = torch.utils.data.DataLoader(
            dataset=dataset,
            batch_size=self.config["batch_size"],
            shuffle=True,
            persistent_workers=True,
            num_workers=self.config["num_workers"]
        )
        return data_loader
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
            self.model.initLgbmModel(lr=self.config["lr"],**kwargs)
        elif(modelType=="torch"):
            self.model = ModelHandler()
            self.model.initTorchModel(modelName,lr=self.config["lr"],max_epochs=self.config["max_epochs"],**kwargs)
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
    