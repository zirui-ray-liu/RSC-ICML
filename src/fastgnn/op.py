
import pdb
from random import sample
import numpy as np

import torch
import torch.nn.functional as F
from torch import topk
from torch.autograd.function import Function
from torch.cuda.amp import custom_fwd, custom_bwd
from fastgnn.utils import empty_cache
from fastgnn import config
from fastgnn import spmm_cuda as spmm


class approxlinear(Function):
    @staticmethod
    @custom_fwd(cast_inputs=torch.float16)
    def forward(ctx, input, weight, bias):
        ctx.saved = input, weight, bias, config.sample_ratio, config.minimal_k
        res = F.linear(input, weight, bias)
        # res = approx_linear.topk(input, weight.t(), bias, sample_ratio, minimal_k)
        # res = rla_topk(input, weight, bias, sample_ratio, minimal_k)
        return res

    @staticmethod
    @custom_bwd
    def backward(ctx, grad_output):
        input, weight, bias, sample_ratio, minimal_k = ctx.saved
        if sample_ratio == 1.0:
            grad_input = grad_output.mm(weight)
            grad_weight = grad_output.t().mm(input)
        # grad_input = approx_linear.topk(grad_output, weight, None, sample_ratio, minimal_k)
        # grad_weight = approx_linear.topk(grad_output.t(), input, None, sample_ratio, minimal_k)
        else:
            grad_input = rla_topk(grad_output, weight, None, sample_ratio, minimal_k)
            grad_weight = rla_topk(grad_output.t(), input, None, sample_ratio, minimal_k)
        if bias is not None:
            grad_bias = grad_output.sum(0)
        else:
            grad_bias = None
        del input
        return grad_input, grad_weight, grad_bias


@torch.no_grad()
def rla_topk(input, weight, bias, sample_ratio, minimal_k):
    # input: n x #input
    # weight: #input x #output
    in_features = weight.shape[0]
    k_candidate = int(in_features * sample_ratio) 
    k = min(max(k_candidate, minimal_k), in_features)
    if k == in_features:
        return F.linear(input, weight.t(), bias)
    #a_col_norms = torch.norm(input, dim=0)
    #b_row_norms = torch.norm(weight, dim=1)
    #norm_mult = a_col_norms * b_row_norms
    # top_k_indices = topk(norm_mult, k, largest=True, dim=0).indices
    top_k_indices = torch.randint(in_features, size=(k,), device='cuda')
    A_top_k_cols = torch.index_select(input, -1, top_k_indices)
    B_top_k_rows = weight[top_k_indices]
    if bias is not None:
        return torch.addmm(bias, A_top_k_cols, B_top_k_rows)
    else:
        return A_top_k_cols.mm(B_top_k_rows)

class qspmm_sum(Function):
    @staticmethod
    @custom_fwd(cast_inputs=torch.float16)
    def forward(ctx, row, rowptr, col, value, rowcount, colptr, csr2csc, has_value, other, scheme):
        result = spmm.spmm_sum_fw(row, rowptr, col, value, colptr, csr2csc, other)
        ctx.saved = row, rowptr, col, value, rowcount, colptr, csr2csc, other, scheme
        ctx.other_args = has_value, value.requires_grad if has_value else False, other.requires_grad
        return result

        # if scheme.activate and config.sample_ratio < 1.0:
        #     row, col, rowptr, colptr, value, csr2csc, rowcount = scheme.subsample_A(row, col, value, rowptr, rowcount, other)
        # N = other.shape[0]
        # # result = spmm.spmm_sum_fw(col, colptr, row, value, rowptr, csr2csc, scheme.slice_tensors(other))
        # # ctx.saved = row, rowptr, col, value, rowcount, colptr, csr2csc, scheme.slice_tensors(other), scheme
        # result = spmm.spmm_sum_fw(row, rowptr, col, value, colptr, csr2csc, other)
        # ctx.saved = row, rowptr, col, value, rowcount, colptr, csr2csc, other, scheme
        # ctx.other_args = has_value, value.requires_grad if has_value else False, other.requires_grad, N
        # return result


    @staticmethod
    @custom_bwd
    def backward(ctx, grad_outputs):
        row, rowptr, col, value, rowcount, colptr, csr2csc, other, scheme = ctx.saved
        has_value, value_requires_grad, mat_requires_grad = ctx.other_args
        if scheme.save_grads ==True:
            scheme.save_deg_gradient(grad_outputs, rowcount)
        if scheme.activate and config.sample_ratio < 1.0:
            row, col, rowptr, colptr, value, csr2csc, rowcount = scheme.subsample_A_rows(row, col, value, rowptr, rowcount, grad_outputs)
            grad_outputs = scheme.slice_tensors(grad_outputs)
        row = col if row is None else row
        value = col if value is None else value
        colptr = col if colptr is None else colptr
        csr2csc = col if csr2csc is None else csr2csc
        grad_value, grad_mat = spmm.spmm_sum_bw(row, rowptr, col, value, colptr, csr2csc, other, grad_outputs, 
                                                has_value, value_requires_grad, mat_requires_grad)
        if config.tune_layer_ratio and not scheme.filled:
            if scheme.F_norm is None:
                scheme.F_norm = torch.norm(grad_mat)
            else:
                scheme.F_norm = 0.5 * torch.norm(grad_mat) + 0.5 * scheme.F_norm
        del other
        return None, None, None, grad_value, None, None, None, None, grad_mat, None

        # row, rowptr, col, value, rowcount, colptr, csr2csc, other, scheme = ctx.saved
        # has_value, value_requires_grad, mat_requires_grad, N = ctx.other_args
        # row = col if row is None else row
        # value = col if value is None else value
        # colptr = col if colptr is None else colptr
        # csr2csc = col if csr2csc is None else csr2csc
        # grad_mat = torch.zeros_like(grad_outputs)
        # grad_mat_ = spmm.spmm_sum_fw(row, rowptr, col, value, colptr, csr2csc, grad_outputs)
        # grad_value = None
        # # grad_mat[scheme._idx] = grad_mat_
        # grad_mat = grad_mat_
        # if config.tune_layer_ratio and not scheme.filled:
        #     if scheme.F_norm is None:
        #         scheme.F_norm = torch.norm(grad_mat)
        #     else:
        #         scheme.F_norm = 0.5 * torch.norm(grad_mat) + 0.5 * scheme.F_norm
        # del other
        # return None, None, None, grad_value, None, None, None, None, grad_mat, None


class qspmm_mean(Function):

    @staticmethod
    @custom_fwd(cast_inputs=torch.float16)
    def forward(ctx, row, rowptr, col, value, rowcount, colptr, csr2csc, has_value, other, scheme):
        result = spmm.spmm_mean_fw(row, rowptr, col, value, rowcount, colptr, csr2csc, other)
        # if scheme.activate and config.sample_ratio < 1.0:
            # row, col, rowptr, colptr, value, csr2csc, rowcount = scheme.subsample_A_rows(row, col, value, rowptr, rowcount, other)
        # ctx.saved = row, rowptr, col, value, rowcount, colptr, csr2csc, scheme.slice_tensors(other), scheme
        ctx.other_args = has_value, value.requires_grad if has_value else False, other.requires_grad
        ctx.saved = row, rowptr, col, value, rowcount, colptr, csr2csc, other, scheme
        return result

    @staticmethod
    @custom_bwd
    def backward(ctx, grad_outputs):
        row, rowptr, col, value, rowcount, colptr, csr2csc, other, scheme = ctx.saved
        if scheme.activate and config.sample_ratio < 1.0:
            row, col, rowptr, colptr, value, csr2csc, rowcount = scheme.subsample_A_rows(row, col, value, rowptr, rowcount, grad_outputs)
            grad_outputs = scheme.slice_tensors(grad_outputs)
        row = col if row is None else row
        value = col if value is None else value
        rowcount = col if rowcount is None else rowcount
        colptr = col if colptr is None else colptr
        csr2csc = col if csr2csc is None else csr2csc
        has_value, value_requires_grad, mat_requires_grad = ctx.other_args
        grad_value, grad_mat = spmm.spmm_mean_bw(row, rowptr, col, value, rowcount, colptr, csr2csc, other, grad_outputs, 
                                                 has_value, value_requires_grad, mat_requires_grad)
        if config.tune_layer_ratio and not scheme.filled:
            if scheme.F_norm is None:
                scheme.F_norm = torch.norm(grad_mat)
            else:
                scheme.F_norm = 0.5 * torch.norm(grad_mat) + 0.5 * scheme.F_norm
        del other
        return None, None, None, grad_value, None, None, None, None, grad_mat, None


class qspmm_max(Function):

    @staticmethod
    @custom_fwd(cast_inputs=torch.float16)
    def forward(ctx, rowptr, col, value, has_value, other, scheme):
        output, arg_out = spmm.spmm_max_fw(rowptr, col, value, other)
        ctx.saved = col, value, other, arg_out
        ctx.other_args = has_value, value.requires_grad if has_value else False, other.requires_grad
        ctx.mark_non_differentiable(arg_out)
        return output

    @staticmethod
    @custom_bwd
    def backward(ctx, grad_outputs):
        col, value, other, arg_out = ctx.saved
        value = col if value is None else value
        has_value, value_requires_grad, mat_requires_grad = ctx.other_args
        grad_value, grad_mat = spmm.spmm_max_bw(col, value, other, arg_out, grad_outputs, 
                                                has_value, value_requires_grad, mat_requires_grad)
        return None, None, grad_value, None, grad_mat, None


class qspmm_min(Function):
    @staticmethod
    @custom_fwd(cast_inputs=torch.float16)
    def forward(ctx, rowptr, col, value, has_value, other, scheme):
        output, arg_out =  spmm.spmm_min_fw(rowptr, col, value, other)
        ctx.saved = col, value, other, arg_out
        ctx.other_args = has_value, value.requires_grad if has_value else False, other.requires_grad
        ctx.mark_non_differentiable(arg_out)
        return output
   
    @staticmethod
    @custom_bwd
    def backward(ctx, grad_outputs):
        col, value, other, arg_out = ctx.saved
        value = col if value is None else value
        has_value, value_requires_grad, mat_requires_grad = ctx.other_args
        grad_value, grad_mat = spmm.spmm_min_bw(col, value, other, arg_out, grad_outputs, 
                                                has_value, value_requires_grad, mat_requires_grad)
        return None, None, grad_value, None, grad_mat, None