import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning as L
import lightgbm as lgb
from pytorch_lightning.loggers import CSVLogger
import pandas as pd

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class ModelHandler():
    def __init__(self):
        self.torchModels={
            "Transformer":Transformer
        }
    def initLgbmModel(self,lr,**kwargs):
        self.model = LightGBMWrapper(lr=lr,**kwargs)
        pass
    def initTorchModel(self,model_name:str,lr:float,max_epochs:int,**kwargs):
        torch_model = self.torchModels[model_name](**kwargs)
        self.model = TorchModelWrapper(torch_model,lr,max_epochs)
        pass
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
class LightGBMWrapper():
    def __init__(self,num_leaves=31,max_depth=-1,n_estimators=100,lr=1e-2):
        self.lgbc=lgb.LGBMClassifier(num_leaves=num_leaves,max_depth=max_depth,n_estimators=n_estimators,learning_rate=lr,objective="binary")
    def fit(self,data_loader,_):
        (X,Y) = data_loader.dataset.tensors
        self.lgbc.fit(X,Y)
    def predict(self,X):
        return self.lgbc.predict(X)
class TorchModelWrapper():
    def __init__(self,predictor:nn.Module,lr=1e-3,max_epochs=4):
        self.predictor = predictor
        self.model = Model(self.predictor,lr)
        logger = CSVLogger("./",name="lightning_logs")
        self.trainer = L.Trainer(
            logger=logger,
            min_epochs=1,
            max_epochs=max_epochs,
            devices="auto",
            accelerator=DEVICE.type,
            log_every_n_steps=100,
            #deterministic=True,
            benchmark=True,
            val_check_interval=0.05
            )
    def fit(self,data_loader,val_loader=None):
        self.trainer.fit(self.model,data_loader,val_loader)
    def predict(self,X):
        self.model.eval()
        with torch.no_grad():
            return self.predictor(X)
class Model(L.LightningModule):
    def __init__(self,predictor,lr):
        super().__init__()
        self.predictor = predictor
        self.lr =lr
    def training_step(self,batch,batch_idx):
        x,y = batch
        y_hat = self.predictor(x)
        loss = F.binary_cross_entropy_with_logits(y_hat.flatten(),y.flatten().float()) 
        self.log("train_loss",loss)
        return loss
    def validation_step(self,batch,batch_idx):
        with torch.no_grad():
            x,y=batch
            logits = self.predictor(x)
            probs = torch.sigmoid(logits)
            preds = (probs>0.5).float().squeeze(-1)
            loss = F.binary_cross_entropy_with_logits(logits,y.float())
            acc = (preds.flatten()==y.flatten().float()).float().mean()

            self.log("val_loss",loss,prog_bar=True,on_epoch=True)
            self.log("val_acc",acc,prog_bar=True,on_epoch=True)

        return loss
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(),self.lr)
        return optimizer
class Transformer(nn.Module):
    def __init__(self,input_size,output_size,hidden_size=512,nhead=4,num_layers=4):
        super(Transformer,self).__init__()
        self.cuda(DEVICE)
        self.lin1 = nn.Linear(input_size,hidden_size)
        self.enc_layer =nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=nhead,
            dim_feedforward=2048,
            dropout=0.1,
            activation=F.gelu,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer=self.enc_layer,
            num_layers=num_layers
        )
        self.lin2 = nn.Linear(hidden_size,output_size)
    def forward(self,x):
        time_size = x.shape[1]
        mask = torch.triu(torch.ones(time_size,time_size), diagonal=1).bool()
        x = F.gelu(self.lin1(x))
        x = self.enc_layer(x,mask)
        x=self.lin2(x)
        return x
