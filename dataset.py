import torch
class Dataset(torch.utils.data.Dataset):
    def __init__(self,X,Y,time_size):
        super().__init__()
        self.X=X
        self.Y=Y
        self.time_size = time_size
        self.to_timestamps()
    def to_timestamps(self):
        uniq = torch.unique(self.X[:,1])
        idx=list()

        for symbol in uniq:
            ids= torch.where(self.X[:,1]==symbol)[0]
            first = torch.min(ids).item()
            last = torch.max(ids).item()-self.time_size+1
            if last<first:continue
            idx.extend(list(range(first,last)))

        self.idx = idx
    def __getitem__(self, index):
        index = self.idx[index]
        return (self.X[index:index+self.time_size,2:],self.Y[index:index+self.time_size])
    def __len__(self):
        return len(self.idx)