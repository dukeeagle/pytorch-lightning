# trainer.py
from collections import Counter
import os
import sys
import time
import ray
from .ray_test import Foo

if __name__=="__main__":
    redis_password = sys.argv[1]
    num_cpus = int(sys.argv[2])
    from pytorch_lightning.utilities.ray import RayTrainer,RayRemoteTrainer

    @ray.remote(num_gpus=None)
    class RayRemote(RayRemoteTrainer):
        pass

    ray_args=dict(address=os.environ["ip_head"], redis_password=redis_password)
    trainer=RayTrainer(RayRemote,num_nodes=1,distributed_backend="ddp",_ray_random_seed=5,_ray_args=ray_args,gpus=None)
    print("Nodes in the Ray cluster:")
    print(ray.nodes())
    trainer.make_model(lambda: Foo())
    sd=trainer.fit()
    print(sd)
