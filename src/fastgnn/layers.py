import imp
from typing import Optional, Tuple

import torch
from torch import Tensor
import torch.nn.functional as F
from torch_geometric.nn.conv.gcn_conv import gcn_norm
from torch_geometric.typing import Adj, OptTensor, Union, OptPairTensor, Size
from torch_geometric.nn.conv import GCNConv, SAGEConv, GCN2Conv, MessagePassing
from torch_sparse import SparseTensor, matmul, fill_diag, sum as sparsesum, mul
from torch.nn import Parameter
from torch_geometric.nn.inits import zeros
from fastgnn.op import approxlinear
from fastgnn.spmm import ApproxMatmul
from torch_geometric.nn.dense.linear import reset_weight_, reset_bias_


class ApproxLinear(torch.nn.Linear):
    def __init__(self, input_features, output_features, bias=True):
        super(ApproxLinear, self).__init__(input_features, output_features, bias)

    def forward(self, input):
        if self.training:
            return approxlinear.apply(input, self.weight, self.bias)
        else:
            return super(ApproxLinear, self).forward(input)

class ApproxGCNConv(GCNConv):
    _deg_inv: Optional[SparseTensor]

    def __init__(self, *args, **kwargs):
        super(ApproxGCNConv, self).__init__(*args, **kwargs)
        self.weight = Parameter(torch.Tensor(self.out_channels, self.in_channels))
        self.msg_and_aggr_func = ApproxMatmul(self.aggr, None)
        self.reset_parameters()
        reset_weight_(self.weight, self.in_channels, self.lin.weight_initializer)


    def forward(self, x: Tensor, edge_index: Adj,
                edge_weight: OptTensor = None) -> Tensor:
        """"""
        if self.normalize:
            if isinstance(edge_index, Tensor):
                cache = self._cached_edge_index
                if cache is None:
                    edge_index, edge_weight = gcn_norm(  # yapf: disable
                        edge_index, edge_weight, x.size(self.node_dim),
                        self.improved, self.add_self_loops, dtype=x.dtype)
                    if self.cached:
                        self._cached_edge_index = (edge_index, edge_weight)
                else:
                    edge_index, edge_weight = cache[0], cache[1]

            elif isinstance(edge_index, SparseTensor):
                cache = self._cached_adj_t
                if cache is None:
                    edge_index = gcn_norm(  # yapf: disable
                        edge_index, edge_weight, x.size(self.node_dim),
                        self.improved, self.add_self_loops, dtype=x.dtype)
                    if self.cached:
                        self._cached_adj_t = edge_index
                else:
                    edge_index = cache
        x = F.linear(x, self.weight, None)
        out = self.propagate(edge_index, x=x, edge_weight=edge_weight,
                             size=None) + x * self._deg_inv

        if self.bias is not None:
            out += self.bias
        return out

    def message_and_aggregate(self, adj_t, x):
        return self.msg_and_aggr_func(adj_t, x)


class ApproxSAGEConv(SAGEConv):
    def __init__(self, *args, **kwargs):
        super(ApproxSAGEConv, self).__init__(*args, **kwargs)
        in_channels = self.in_channels
        if isinstance(in_channels, int):
            in_channels = (in_channels, in_channels)
        bias = self.lin_l.bias is not None
        self.lin_l = torch.nn.Linear(in_channels[0], self.out_channels, bias=bias)
        if self.root_weight:
            self.lin_r = torch.nn.Linear(in_channels[1], self.out_channels, bias=False)
        self.msg_and_aggr_func = ApproxMatmul(self.aggr, None)
        self.reset_parameters()


    def forward(self, x: Union[Tensor, OptPairTensor], edge_index: Adj,
                size: Size = None) -> Tensor:
        """"""
        if isinstance(x, Tensor):
            x: OptPairTensor = (x, x)

        # propagate_type: (x: OptPairTensor)
        out = self.propagate(edge_index, x=x, size=size)
        out = self.lin_l(out)
        x_r = x[1]
        if self.root_weight and x_r is not None:
            out += self.lin_r(x_r)

        if self.normalize:
            out = F.normalize(out, p=2., dim=-1)
        return out

    def message_and_aggregate(self, adj_t, x) -> Tensor:
        # adj_t = adj_t.set_value(None, layout=None)
        return self.msg_and_aggr_func(adj_t, x[0])

class ApproxGCN2Conv(GCN2Conv):
    _deg_inv: Optional[SparseTensor]
    def __init__(self, *args, **kwargs):
        super(ApproxGCN2Conv, self).__init__(*args, **kwargs)
        self.msg_and_aggr_func = ApproxMatmul(self.aggr, None)

    def forward(self, x: Tensor, x_0: Tensor, edge_index: Adj,
                edge_weight: OptTensor = None) -> Tensor:
        """"""

        if self.normalize:
            if isinstance(edge_index, Tensor):
                cache = self._cached_edge_index
                if cache is None:
                    edge_index, edge_weight = gcn_norm(  # yapf: disable
                        edge_index, edge_weight, x.size(self.node_dim), False,
                        self.add_self_loops, dtype=x.dtype)
                    if self.cached:
                        self._cached_edge_index = (edge_index, edge_weight)
                else:
                    edge_index, edge_weight = cache[0], cache[1]

            elif isinstance(edge_index, SparseTensor):
                cache = self._cached_adj_t
                if cache is None:
                    edge_index = gcn_norm(  # yapf: disable
                        edge_index, edge_weight, x.size(self.node_dim), False,
                        self.add_self_loops, dtype=x.dtype)
                    if self.cached:
                        self._cached_adj_t = edge_index
                else:
                    edge_index = cache

        # propagate_type: (x: Tensor, edge_weight: OptTensor)
        x = self.propagate(edge_index, x=x, edge_weight=edge_weight, size=None) + x * self._deg_inv

        x.mul_(1 - self.alpha)
        x_0 = self.alpha * x_0[:x.size(0)]

        if self.weight2 is None:
            out = x.add_(x_0)
            out = torch.addmm(out, out, self.weight1, beta=1. - self.beta,
                               alpha=self.beta)
        else:
            out = torch.addmm(x, x, self.weight1, beta=1. - self.beta,
                               alpha=self.beta)
            out += torch.addmm(x_0, x_0, self.weight2, beta=1. - self.beta,
                                alpha=self.beta)

        return out

    def message(self, x_j: Tensor, edge_weight: Tensor) -> Tensor:
            return edge_weight.view(-1, 1) * x_j

    def message_and_aggregate(self, adj_t: SparseTensor, x: Tensor) -> Tensor:
        return self.msg_and_aggr_func(adj_t, x)

    def reset_parameters(self):
        super().reset_parameters()
