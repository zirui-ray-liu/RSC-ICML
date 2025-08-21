#pragma once

#include <torch/extension.h>

std::tuple<torch::Tensor, torch::optional<torch::Tensor>>
spmm_cuda(torch::Tensor rowptr, torch::Tensor col,
          torch::optional<torch::Tensor> optional_value, torch::Tensor mat,
          std::string reduce);

torch::Tensor spmm_value_bw_cuda(torch::Tensor row, torch::Tensor rowptr,
                                 torch::Tensor col, torch::Tensor mat,
                                 torch::Tensor grad, std::string reduce);

torch::Tensor gather_csr_cuda(torch::Tensor src, torch::Tensor indptr,
                              torch::optional<torch::Tensor> optional_out);

torch::Tensor ind2ptr_cuda(torch::Tensor ind, int64_t M);