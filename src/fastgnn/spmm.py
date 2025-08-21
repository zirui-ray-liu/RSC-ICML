import torch
from .conf import config
from .op import qspmm_sum, qspmm_mean, qspmm_max, qspmm_min
from .scheme import Scheme

class ApproxMatmul(torch.nn.Module):
    def __init__(self, reduce, scheme=None):
        super().__init__()
        if reduce in ['sum', 'add']:
            reduce = 'sum'
        self.reduce = reduce
        if scheme is None:
            self.scheme = Scheme(self)
        else:
            self.scheme = scheme

    def forward(self, src, other): 
        rowptr, col, value = src.csr()
        has_value = value is not None
        if self.reduce == 'sum':
            row = src.storage._row
            rowcount = src.storage._rowcount
            csr2csc = src.storage._csr2csc
            colptr = src.storage._colptr
            if has_value and value.requires_grad:
                row = src.storage.row()
            if other.requires_grad:
                row = src.storage.row()
                rowcount = src.storage.rowcount()
                csr2csc = src.storage.csr2csc()
                colptr = src.storage.colptr()
            return qspmm_sum.apply(row, rowptr, col, value, rowcount, colptr, csr2csc, has_value, other, self.scheme)

        elif self.reduce == 'mean':
            row = src.storage._row
            rowcount = src.storage._rowcount
            csr2csc = src.storage._csr2csc
            colptr = src.storage._colptr
            if has_value and value.requires_grad:
                row = src.storage.row()
            if other.requires_grad:
                row = src.storage.row()
                rowcount = src.storage.rowcount()
                csr2csc = src.storage.csr2csc()
                colptr = src.storage.colptr()
            return qspmm_mean.apply(row, rowptr, col, value, rowcount, colptr, csr2csc, has_value, other, self.scheme)

        elif self.reduce == 'min':
            return qspmm_min.apply(rowptr, col, value, has_value, other, self.scheme)

        elif self.reduce == 'max':
            return qspmm_max.apply(rowptr, col, value, has_value, other, self.scheme)

        else:
            raise ValueError


def approx_matmul(src, other, reduce="sum"):
    rowptr, col, value = src.csr()
    has_value = value is not None
    if reduce in ['sum', 'add']:
        reduce = 'sum'
        row = src.storage._row
        csr2csc = src.storage._csr2csc
        colptr = src.storage._colptr
        colcount = src.storage._colcount
        if has_value and value.requires_grad:
            row = src.storage.row()
        if other.requires_grad:
            row = src.storage.row()
            csr2csc = src.storage.csr2csc()
            colptr = src.storage.colptr()
        return qspmm_sum.apply(row, rowptr, col, value, colptr, csr2csc, has_value, colcount, other)

    elif reduce == 'mean':
        row = src.storage._row
        rowcount = src.storage._rowcount
        csr2csc = src.storage._csr2csc
        colptr = src.storage._colptr
        colcount = src.storage._colcount
        if has_value and value.requires_grad:
            row = src.storage.row()
        if other.requires_grad:
            row = src.storage.row()
            rowcount = src.storage.rowcount()
            csr2csc = src.storage.csr2csc()
            colptr = src.storage.colptr()
        return qspmm_mean.apply(row, rowptr, col, value, rowcount, colptr, csr2csc, has_value, other)

    elif reduce == 'min':
        return qspmm_min.apply(rowptr, col, value, has_value, other)

    elif reduce == 'max':
        return qspmm_max.apply(rowptr, col, value, has_value, other)

    else:
        raise ValueError