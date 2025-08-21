
from typing import Union, Tuple, Any, Callable, Iterator, Set, Optional, overload, TypeVar, Mapping, Dict
from collections import OrderedDict
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor, device, dtype

from torch_geometric.nn.conv import GCNConv, SAGEConv, GCN2Conv
from .layers import ApproxLinear, ApproxGCNConv, ApproxSAGEConv, ApproxGCN2Conv
from .conf import config
from .scheme import Scheme


class ApproxModule(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        ApproxModule.convert_layers(model)


    @staticmethod
    def convert_layers(module):
        for name, child in module.named_children():
            # Do not convert layers that are already quantized
            if isinstance(child, (ApproxLinear, ApproxGCNConv)):
                continue
            # if isinstance(child, nn.Linear):
            #     setattr(module, name, ApproxLinear(child.in_features, child.out_features,
            #         child.bias is not None))

            elif isinstance(child, GCNConv):
                setattr(module, name, ApproxGCNConv(child.in_channels, child.out_channels, child.improved, child.cached,
                                               child.add_self_loops, child.normalize, child.bias is not None,
                                               aggr=child.aggr))
            elif isinstance(child, GCN2Conv):
                beta = child.beta
                shared_weights = child.weight2 is None
                setattr(module, name, ApproxGCN2Conv(child.channels, alpha=child.alpha, theta=None, layer=None, shared_weights=shared_weights,
                                                     cached=child.cached, add_self_loops=child.add_self_loops, normalize=child.normalize))
                curconv = getattr(module, name)
                curconv.beta = child.beta
            elif isinstance(child, SAGEConv):
                setattr(module, name, ApproxSAGEConv(child.in_channels, child.out_channels, 
                                                     child.aggr,child.normalize, 
                                                     child.root_weight, child.project,
                                                     child.lin_l.bias is not None))
            # elif isinstance(child, nn.ReLU):
            #     setattr(module, name, QReLU())
            # elif isinstance(child, nn.Dropout):
            #     setattr(module, name, QDropout(child.p))
            else:
                ApproxModule.convert_layers(child)

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    def load_state_dict(self, state_dict: Union[Dict[str, Tensor], Dict[str, Tensor]],
                        strict: bool = True):
        # remove the prefix "model." added by this wrapper
        new_state_dict = OrderedDict([("model." + k,  v) for k, v in state_dict.items()])
        return super().load_state_dict(new_state_dict, strict)

    def state_dict(self, destination=None, prefix='', keep_vars=False):
        ret = super().state_dict(destination, prefix, keep_vars)

        # remove the prefix "model." added by this wrapper
        ret = OrderedDict([(k[6:], v) for k, v in ret.items()])
        return ret

    def reset_parameters(self):
        self.model.reset_parameters()

    @torch.no_grad()
    def mini_inference(self, x_all, loader):
        return self.model.mini_inference(x_all, loader)