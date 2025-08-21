import torch
import numpy as np
import pdb
import json
import os
PROJECT_ABSOLUTE_PATH=os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
print(f'PROJECT_ABSOLUTE_PATH: {PROJECT_ABSOLUTE_PATH}')

from fastgnn.conf import config
from fastgnn import spmm_cuda as spmm

from torch import topk
from torch_scatter import gather_csr

class Scheme(object):
    num_samples = 0
    num_layers = 0
    layers = []
    A_row_norms = None
    # flags for saving the gradients and degrees
    save_grads = False
    model = 'gcn2'
    dataset = 'reddit2'
    k = 1
    minibatch = False

    def __init__(self, layer):
        self.scales = torch.zeros(Scheme.num_samples)
        Scheme.layers.append(self)
        self.layer = layer
        # debug
        self.name = f'{layer.__class__.__name__}_{Scheme.num_layers}'
        self.iteration = 0
        Scheme.num_layers += 1
        self._row = None
        self._col = None
        self._rowptr = None
        self._colptr = None
        self._value = None
        self._rowcount = None
        self._csr2csc = None
        self._idx = None
        self.activate = True
        self.sample_ratio = None
        self.filled = False  # serves as a flag to indicate whether the sample ratio is set.
        self.norm_mult = None
        self.grad_norm = None
        self.F_norm = None
        # for saving the gradients and degrees
        self.deg_saved = False
        self.path_for_saving_grads_and_degree = f'{PROJECT_ABSOLUTE_PATH}/visualization/saved_grads_and_degrees'

    def slice_tensors(self, tensor):
        return tensor[self._idx]

    def subsample_A_rows(self, row, col, value, rowptr, rowcount, grad_outputs):
        if self.minibatch:
            self._idx = self._build_indices(grad_outputs)
            self._row, self._col, self._rowptr, self._colptr, self._value, self._csr2csc, self._rowcount = self._subsample_rows(row, col, value, rowptr, rowcount, self._idx)
        elif self.iteration % self.k == 0:
            self._idx = self._build_indices(grad_outputs)
            self._row, self._col,  self._rowptr, self._colptr, self._value, self._csr2csc, self._rowcount = self._subsample_rows(row, col, value, rowptr, rowcount, self._idx)
        self.iteration += 1
        return self._row, self._col, self._rowptr, self._colptr, self._value, self._csr2csc, self._rowcount

    def subsample_A_cols(self, row, col, value, colptr, colcount, grad_outputs):
        if self.minibatch:
            self._idx = self._build_indices(grad_outputs)
            self._row, self._col, self._rowptr, self._colptr, self._value, self._csr2csc, self._rowcount = self._subsample_rows(row, col, value, colptr, colcount, self._idx)
        elif self.iteration % self.k == 0:
            self._idx = self._build_indices(grad_outputs)
            self._row, self._col,  self._rowptr, self._colptr, self._value, self._csr2csc, self._rowcount = self._subsample_rows(row, col, value, colptr, colcount, self._idx)
        self.iteration += 1
        return self._row, self._col, self._rowptr, self._colptr, self._value, self._csr2csc, self._rowcount

    # _subsample_rows is time-consuming
    def _subsample_rows(self, row, col, value, old_rowptr, rowcount, idx):
        # caz A is a square matrix
        if old_rowptr is None:
            M = int(row.max()) + 1
            old_rowptr = torch.ops.torch_sparse.ind2ptr(row, M)
        if rowcount is None:
            rowcount = old_rowptr[1:] - old_rowptr[:-1]

        num_cols = rowcount.shape[0]
        rowcount = rowcount[idx]
        rowptr = col.new_zeros(idx.size(0) + 1)
        torch.cumsum(rowcount, dim=0, out=rowptr[1:])
        row = torch.arange(idx.size(0),
                           device=col.device, dtype=col.dtype).repeat_interleave(rowcount)
        perm = torch.arange(row.size(0), device=row.device)
        if rowptr.dtype != torch.long:
            perm += spmm.gather_csr_fw(old_rowptr[idx] - rowptr[:-1], rowptr, None)
        else:
            perm += gather_csr(old_rowptr[idx] - rowptr[:-1], rowptr)
        col = col[perm]
        if value is not None:
            value = value[perm]
        del perm
        sparse_sizes = (idx.size(0), num_cols)
        aux_idx = sparse_sizes[0] * col + row
        csr2csc = aux_idx.argsort()
        if col.dtype != torch.long:
            colptr = spmm.ind2ptr_fw(col[csr2csc],sparse_sizes[1])
        else:
            colptr = torch.ops.torch_sparse.ind2ptr(col[csr2csc],sparse_sizes[1])
        return row, col, rowptr, colptr, value, csr2csc, rowcount


    def _subsample_cols(self, row, col, value, old_colptr, colcount, idx):
        colcount = colcount[idx]
        colptr = row.new_zeros(idx.size(0) + 1)
        torch.cumsum(colcount, dim=0, out=colptr[1:])

        col = torch.arange(idx.size(0),
                           device=row.device).repeat_interleave(colcount)

        perm = torch.arange(col.size(0), device=col.device)
        if colptr.dtype != torch.long:
            perm += spmm.gather_csr_fw(old_colptr[idx] - colptr[:-1], colptr, None)
        else:
            perm += gather_csr(old_colptr[idx] - colptr[:-1], colptr)

        row = row[perm]
        csc2csr = (idx.size(0) * row + col).argsort()
        row, col = row[csc2csr], col[csc2csr]

        if value is not None:
            value = value[perm][csc2csr]
        
        num_rows = colcount.shape[0]

        sparse_sizes = (num_rows, idx.size(0))
        aux_idx = sparse_sizes[0] * col + row
        csr2csc = aux_idx.argsort()
        if row.dtype != torch.long:
            rowptr = spmm.ind2ptr_fw(row, sparse_sizes[0])
        else:
            rowptr = torch.ops.torch_sparse.ind2ptr(row, sparse_sizes[0])
        rowcount = rowptr[1:] - rowptr[:-1]
        return row, col, rowptr, colptr, value, csr2csc, rowcount
    
    def _build_indices(self, grad_outputs):
        minimal_k = config.minimal_k
        if config.tune_layer_ratio:
            sample_ratio = self.sample_ratio
        else:
            sample_ratio = config.sample_ratio
        sample_ratio = sample_ratio if sample_ratio else 1.0
        in_features = grad_outputs.shape[0]
        k_candidate = int(in_features * sample_ratio) 
        k = min(max(k_candidate, minimal_k), in_features)
        b_row_norms = torch.norm(grad_outputs, dim=1)
        if self.layer.reduce == 'mean':
            norm_mult = b_row_norms / self.A_row_norms
            norm_mult = torch.nan_to_num(norm_mult, nan=0.0, posinf=0.0, neginf=0.0)
        else:
            norm_mult = self.A_row_norms * b_row_norms
        # norm_mult = b_row_norms
        if config.tune_layer_ratio and not self.filled:
            if self.norm_mult is None or self.minibatch:
                self.norm_mult = norm_mult
            else:
                self.norm_mult = 0.5 * norm_mult + 0.5 * self.norm_mult
                
            if self.grad_norm is None or self.minibatch:
                self.grad_norm = b_row_norms
            else:
                self.grad_norm = 0.5 * b_row_norms + 0.5 * self.grad_norm          
        top_k_indices = topk(norm_mult, k, largest=True).indices
        # top_k_indices = torch.randperm(in_features, device='cuda')[:k]
        top_k_indices, _ = torch.sort(top_k_indices)
        self.save_indices(top_k_indices)
        return top_k_indices
    
    def save_indices(self, indices):
        save_path = f'{self.path_for_saving_grads_and_degree}/{self.model}_{self.dataset}/'
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        file_name = os.path.join(save_path, f'{self.name}_it{self.iteration}_indices.pt')
        torch.save(indices, file_name)


    def save_deg_gradient(self, grad_outputs, rowcount):
        save_path = f'{self.path_for_saving_grads_and_degree}/{self.model}_{self.dataset}/'
        if os.path.exists(save_path) == False:
            os.makedirs(save_path)
        # save deg once
        if self.deg_saved == False:
            self._save_deg(rowcount, save_path)
            self.deg_saved = True
        # save grad
        self._save_grad(grad_outputs, save_path)
        
    def _save_deg(self, rowcount, path):
        path = path+'deg.pt'
        torch.save(rowcount, path)

    def _save_grad(self, grad_outputs, path):
        scales = grad_outputs.norm(dim=1).cpu()
        path = path + f'{self.name}_it{self.iteration}_scale.pt'
        torch.save(scales, path)