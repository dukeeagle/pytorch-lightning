import ray
from torch.utils.data import TensorDataset, DataLoader

from pytorch_lightning import LightningModule
import torch.nn.functional as F

import torch as pt
class Foo(LightningModule):
    def __init__(self,batch_size=5):
        super().__init__()
        self.batch_size=10
        self.linear=pt.nn.Linear(3,10)
        self.linear2=pt.nn.Linear(10,3)
        self.loss=pt.nn.CrossEntropyLoss()
    def prepare_data(self) -> None:
        self.ts=TensorDataset(pt.randn(100,3),pt.randint(3,[100]))
        self.vs = TensorDataset(pt.randn(100, 3), pt.randint(3,[100]))
        self.tts = TensorDataset(pt.randn(100, 3), pt.randint(3,[100]))
    def train_dataloader(self) -> DataLoader:
        return DataLoader(self.ts,batch_size=self.batch_size,shuffle=True)
    def configure_optimizers(self):
        return pt.optim.Adam(self.parameters())
    def forward(self, x):
        x=self.linear(x)
        x=F.relu(x)
        x=self.linear2(x)
        return x
    def training_step(self, batch,batch_idx):
        x,y=batch
        yhat=self.forward(x)
        print(yhat.shape)
        loss=self.loss(yhat,y)
        ret={"loss":loss}
        print(ret)
        return ret
from pytorch_lightning.utilities.ray import RayTrainer,RayRemoteTrainer

@ray.remote(num_gpus=None)
class RayRemote(RayRemoteTrainer):
    pass

trainer=RayTrainer(RayRemote,num_nodes=1,distributed_backend="ddp",_ray_random_seed=5,gpus=None)
trainer.make_model(lambda: Foo())
sd=trainer.fit()
print(sd)

