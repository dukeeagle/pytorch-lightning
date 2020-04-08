import ray
from typing import Dict,List

from pytorch_lightning import Trainer
import os
import random
import torch
import numpy as np
import sys
import uuid

def set_seed(seed,deterministic=True,benchmark=False):
    """
    Call this function *before* instantiating model if creating the model in separate processes
    :param seed:
    :param deterministic:
    :param benchmark:
    :return:
    """
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cudnn.benchmark=benchmark

@ray.remote
class RayRemoteTrainer(Trainer):
    def __init__(self,random_seed,node_rank,local_rank,*args,**kwargs):
        self._node_rank=node_rank
        self._local_rank=local_rank
        self._seed=random_seed
        print(args)
        print(kwargs)
        # set the seed again...
        set_seed(random_seed)

        self._this_node_id=ray.state.current_node_id()
        self._all_nodes=sorted(ray.state.node_ids())
        self._job_id=ray.worker.global_worker.current_task_id
        # set environment list with list of ips
        os.environ["LIGHTNING_NODELIST"]=",".join([n.replace(ray.resource_spec.NODE_ID_PREFIX,"") for n in self._all_nodes])
        os.environ["LIGHTNING_NODEID"]=f"{self._node_rank}"
        os.environ["LIGHTNING_LOCALID"]=f"{self._local_rank}"
        # not gonn set MASTER etc. from https://pytorch.org/tutorials/intermediate/dist_tuto.html#advanced-topics since they are set inside Trainer
        kwargs["ray"]=True
        kwargs["slurm"]=False
        self.args=args
        self.kwargs=kwargs

    def state_dict(self):
        return self._ray_model.state_dict()
    def make_model(self,model_callback):
        self._ray_model=model_callback()

    def set_state_dict(self,dict):
        self._ray_model.load_state_dict(dict)

    def init_inner(self):
        super().__init__(*self.args, **self.kwargs)

    def fit(self):
        super().fit(self._ray_model)


def state_dict(self):
    return self.model.state_dict()


class RayTrainer:
    #TODO: probably want this or RayRemoteTrainer as a wrapper arround the original trainer or as a mixin?
    def __init__(self,*trainer_args:List,_ray_address="auto",_ray_redis_pw=None,_ray_random_seed=None,**trainer_kwargs:Dict):
        ray.init(address=_ray_address,redis_password=_ray_redis_pw)
        if _ray_random_seed is None:
            _ray_random_seed = random.randint(0,sys.maxsize)
        dpbackend=trainer_kwargs.get("distributed_backend")
        num_nodes = trainer_kwargs["num_nodes"]
        any_gpus=trainer_kwargs.get("gpus") is not None
        if dpbackend=="ddp2":
            # ddp2: use multiple nodes, but use all gpus on each process => 1 actor per gpu batch
            actor_gpus=trainer_kwargs["gpus"] if any_gpus else None
            local_actors=1
        elif dpbackend=="ddp":
            # ddp2: use one or multiple nodes, each process has it's own GPU => 1 actor per gpu
            actor_gpus = 1 if any_gpus else None
            local_actors=trainer_kwargs["gpus"] if any_gpus else 1
        else:
           raise ValueError("Only ddp and ddp2 implemented for now, other stuff doesn't really make sense? ")
        self.remote_trainers = []
        RemoteTrainer = RayRemoteTrainer.options(num_gpus=actor_gpus)
        trainer_kwargs["_data_lock"]="ray_lock_"+str(uuid.uuid4())
        for node_rank in range(num_nodes):
            for local_rank in range(local_actors):
                rt=RemoteTrainer.remote(_ray_random_seed,node_rank,local_rank,*trainer_args, **trainer_kwargs)
                self.remote_trainers.append(rt)

        handles = [x.init_inner.remote() for x in self.remote_trainers]
        ray.get(handles)

    def shutdown(self):
        ray.shutdown()
    def __del__(self):
        self.shutdown()
    def make_model(self,model_callback):
        handles = [x.make_model.remote(model_callback) for x in self.remote_trainers]
        ray.get(handles)

    def fit(self):
        handles=[x.fit.remote() for x in self.remote_trainers]
        ray.get(handles)
        state_dict=ray.get(self.remote_trainers[0].state_dict.remote())
        return state_dict
