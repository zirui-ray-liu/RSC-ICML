import pdb
import os
from numpy import dtype
os.environ['CUDA_VISIBLE_DEVICES'] = "3"
from torch_sparse import SparseTensor
from torch_sparse.matmul import spmm_sum
import fastgnn.spmm_cuda as our_spmm
import torch
import pdb
from torch_sparse.storage import SparseStorage
from fastgnn.scheme import Scheme
from fastgnn import spmm_cuda as spmm



def get_sparse_tensor_row_sampling_argument(sp_tensor):
    return sp_tensor.storage.row(), sp_tensor.storage.col(), sp_tensor.storage.value(), sp_tensor.storage.rowptr(), sp_tensor.storage.rowcount(), sp_tensor.storage.colptr(), sp_tensor.storage.csr2csc()

def test_sample_spmm_correctness():
    scheme = Scheme(None)
    row = [0, 0, 0, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 4, 4, 5, 6, 6, 6, 7, 7, 7, 7]
    col = [2, 4, 7, 3, 6, 0, 4, 7, 1, 4, 6, 0, 2, 3, 6, 7, 7, 1, 3, 4, 0, 2, 4, 5]
    cuda0 = torch.device('cuda:0')
    row, col = torch.tensor(row, dtype=torch.long, device=cuda0), torch.tensor(col,  dtype=torch.long, device=cuda0)
    # value = torch.rand_like(row, dtype=torch.float)
    # value = torch.ones_like(row)
    edge_index = torch.cat([row.view(1, -1), col.view(1, -1)], dim=0)
    sp_tensor = SparseTensor.from_edge_index(edge_index)
    dense = sp_tensor.to_dense()
    grad_output = torch.rand_like(dense)
    row, col, value, rowptr, rowcount, colptr, csr2csc = get_sparse_tensor_row_sampling_argument(sp_tensor)
    idx = torch.tensor([0, 3, 5, 7], device=cuda0)
    sampled_row, sampled_colptr, sampled_value, sampled_csr2csc, sampled_rowcount = scheme._subsample_rows(row, col, value, rowptr, rowcount, idx)
    dense_result = torch.matmul(dense[idx].T, grad_output[idx])
    # dense_result = torch.matmul(dense.T, grad_output)
    has_value = value is not None
    value = col if value is None else value 
    sampled_value = col if sampled_value is None else sampled_value
    _, grad_mat = spmm.spmm_sum_bw(sampled_row, rowptr, col, sampled_value, sampled_colptr, sampled_csr2csc, grad_output, grad_output[idx], 
                                            has_value, False, True)
    # grad_value, grad_mat = spmm.spmm_sum_bw(row, rowptr, col, value, colptr, csr2csc, grad_output, grad_output, 
    #                                         has_value, False, True)

    assert torch.max(torch.abs(grad_mat - dense_result)) == 0.

if __name__ == "__main__":
    test_sample_spmm_correctness()