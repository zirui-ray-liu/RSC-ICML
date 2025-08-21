// This code is based off of torch_sparse's
// https://github.com/rusty1s/pytorch_sparse/blob/master/csrc/cuda/spmm_cuda.cu

#include "spmm_cuda.h"

#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/detail/IndexUtils.cuh>
#include <ATen/cuda/detail/TensorInfo.cuh>

#include "reducer.cuh"
#include "index_info.cuh"
#include "ext_common.h"

#define THREADS 256
#define FULL_MASK 0xffffffff

__device__ __inline__ at::Half
__shfl_sync(const unsigned mask, const at::Half var, const int srcLane) {
  return __shfl_sync(mask, (__half)var, srcLane);
}

__device__ __inline__ at::Half __shfl_down_sync(const unsigned mask,
                                                const at::Half var,
                                                const unsigned int delta) {
  return __shfl_down_sync(mask, (__half)var, delta);
}

// Paper: Design Principles for Sparse Matrix Multiplication on the GPU
// Code:  https://github.com/owensgroup/merge-spmm
template <typename scalar_t, typename index_t, ReductionType REDUCE, bool HAS_VALUE>
__global__ void spmm_kernel(const index_t *rowptr_data, const index_t *col_data,
                            const scalar_t *value_data,
                            const scalar_t *mat_data, scalar_t *out_data,
                            index_t *arg_out_data, int B, int M, int N, int K) {

  // We ignore blockIdx.y here, because threads
  // across `blockIdx.y` are treated equally.
  int thread_idx = blockDim.x * blockIdx.x + threadIdx.x;

  int row = thread_idx >> 5;            // thread_idx / 32
  int lane_idx = thread_idx & (32 - 1); // thread_idx % 32
  int batch_idx = row / M;

  // Compute the column index of `mat` in which the thread is operating.
  int mat_col_idx = lane_idx + (blockIdx.y << 5);

  // Compute the output index (row-major order).
  int out_idx = row * K + mat_col_idx;

  // Helper arrays for warp communication.
  int mat_row, mat_rows[32];
  scalar_t val, vals[HAS_VALUE ? 32 : 1];

  // Do not aggregate/write across the Y-axis (lane_idx < leftover).
  int leftover = K - (blockIdx.y << 5);

  if (batch_idx < B) {
    int row_start = __ldg(rowptr_data + (row % M));
    int row_end = __ldg(rowptr_data + (row % M) + 1);
    int col_idx = row_start + lane_idx;

    scalar_t result = Reducer<scalar_t, index_t, REDUCE>::init();
    index_t arg;

    // Iterate over all `col` indices in parallel within a warp.
    for (int c = row_start; c < row_end; c += 32) {

      if (col_idx < row_end) {
        // Coalesced memory access into `col` and `val`.
        mat_row = __ldg(col_data + col_idx) * K;
        if (HAS_VALUE)
          val = __ldg(value_data + col_idx);
      } else {
        mat_row = -1;
        if (HAS_VALUE)
          val = (scalar_t)0;
      }
      col_idx += 32;

#pragma unroll
      for (int i = 0; i < 32; i++) {
        // Communication between all threads in a warp.
        mat_rows[i] = __shfl_sync(FULL_MASK, mat_row, i);
        if (HAS_VALUE)
          vals[i] = __shfl_sync(FULL_MASK, val, i);
      }

#pragma unroll
      for (int i = 0; i < 32; i++) {
        if (lane_idx < leftover && mat_rows[i] != -1) {
          // Coalesced memory access into `mat`.
          val = __ldg(mat_data + batch_idx * N * K + mat_rows[i] + mat_col_idx);
          if (HAS_VALUE)
            val = vals[i] * val;
          Reducer<scalar_t, index_t, REDUCE>::update(&result, val, &arg, c + i);
        }
      }
    }

    if (lane_idx < leftover) {
      // Coalesced write into `out`.
      Reducer<scalar_t, index_t, REDUCE>::write(out_data + out_idx, result,
                                       arg_out_data + out_idx, arg,
                                       row_end - row_start);
    }
  }
}

std::tuple<torch::Tensor, torch::optional<torch::Tensor>>
spmm_cuda(torch::Tensor rowptr, torch::Tensor col,
          torch::optional<torch::Tensor> optional_value, torch::Tensor mat,
          std::string reduce) {

  CHECK_CUDA(rowptr);
  CHECK_CUDA(col);
  if (optional_value.has_value())
    CHECK_CUDA(optional_value.value());
  CHECK_CUDA(mat);
  cudaSetDevice(rowptr.get_device());

  CHECK_INPUT(rowptr.dim() == 1);
  CHECK_INPUT(col.dim() == 1);
  if (optional_value.has_value()) {
    CHECK_INPUT(optional_value.value().dim() == 1);
    CHECK_INPUT(optional_value.value().size(0) == col.size(0));
  }
  CHECK_INPUT(mat.dim() >= 2);

  mat = mat.contiguous();

  auto sizes = mat.sizes().vec();
  sizes[mat.dim() - 2] = rowptr.numel() - 1;
  auto out = torch::empty(sizes, mat.options());

  torch::optional<torch::Tensor> arg_out = torch::nullopt;

  auto M = rowptr.numel() - 1;
  auto N = mat.size(-2);
  auto K = mat.size(-1);
  auto B = mat.numel() / (N * K);
  auto BLOCKS = dim3((32 * B * M + THREADS - 1) / THREADS, (K + 31) / 32);

  auto stream = at::cuda::getCurrentCUDAStream();

  AT_DISPATCH_ALL_TYPES_AND(at::ScalarType::Half, mat.scalar_type(), "_", [&] {
    auto mat_data = mat.data_ptr<scalar_t>();
    auto out_data = out.data_ptr<scalar_t>();

    AT_DISPATCH_REDUCTION_TYPES(reduce, [&] {
      AT_DISPATCH_INDEX_TYPES(col.scalar_type(), "spmm_kernel", [&]() {
        auto rowptr_data = rowptr.data_ptr<index_t>();
        auto col_data = col.data_ptr<index_t>();
        index_t *arg_out_data = nullptr;
        if (reduce2REDUCE.at(reduce) == MIN || reduce2REDUCE.at(reduce) == MAX) {
          arg_out = torch::full_like(out, col.numel(), rowptr.options());
          arg_out_data = arg_out.value().data_ptr<index_t>();
        }
        if (optional_value.has_value()) {
          auto value_data = optional_value.value().data_ptr<scalar_t>();
          spmm_kernel<scalar_t, index_t, REDUCE, true><<<BLOCKS, THREADS, 0, stream>>>(
              rowptr_data, col_data, value_data, mat_data, out_data, arg_out_data,
              B, M, N, K);
        } else {
          spmm_kernel<scalar_t, index_t, REDUCE, false><<<BLOCKS, THREADS, 0, stream>>>(
              rowptr_data, col_data, nullptr, mat_data, out_data, arg_out_data, B,
              M, N, K);
        }
      });
    });
  });

  return std::make_tuple(out, arg_out);
}

template <typename scalar_t, ReductionType REDUCE>
__global__ void
spmm_value_bw_kernel(const int64_t *row_data, const int64_t *rowptr_data,
                     const int64_t *col_data, const scalar_t *mat_data,
                     const scalar_t *grad_data, scalar_t *out_data, int B,
                     int M, int N, int E, int K) {
  int thread_idx = blockDim.x * blockIdx.x + threadIdx.x;

  int index_idx = (thread_idx >> 5);    // thread_idx / 32
  int lane_idx = thread_idx & (32 - 1); // thread_idx % 32

  if (index_idx < E) {
    int row = __ldg(row_data + index_idx);
    int col = __ldg(col_data + index_idx);

    scalar_t val = (scalar_t)0;
    for (int b = 0; b < B; b++) {
      for (int k = lane_idx; k < K; k += 32) {
        val += mat_data[b * N * K + col * K + k] *
               grad_data[b * M * K + row * K + k];
      }
    }

#pragma unroll
    for (int i = 32 / 2; i > 0; i /= 2) { // Parallel reduction inside a warp.
      val += __shfl_down_sync(FULL_MASK, val, i);
    }

    if (lane_idx == 0) {
      if (REDUCE == MEAN) {
        int row_start = __ldg(rowptr_data + row);
        int row_end = __ldg(rowptr_data + row + 1);
        val /= (scalar_t)max(row_end - row_start, 1);
      }
      out_data[index_idx] = val;
    }
  }
}

torch::Tensor spmm_value_bw_cuda(torch::Tensor row, torch::Tensor rowptr,
                                 torch::Tensor col, torch::Tensor mat,
                                 torch::Tensor grad, std::string reduce) {
  CHECK_CUDA(row);
  CHECK_CUDA(rowptr);
  CHECK_CUDA(col);
  CHECK_CUDA(mat);
  CHECK_CUDA(grad);
  cudaSetDevice(row.get_device());

  mat = mat.contiguous();
  grad = grad.contiguous();

  auto M = grad.size(-2);
  auto N = mat.size(-2);
  auto E = row.numel();
  auto K = mat.size(-1);
  auto B = mat.numel() / (N * K);
  auto BLOCKS = dim3((E * 32 + THREADS - 1) / THREADS);

  auto out = torch::zeros(row.numel(), grad.options());

  auto row_data = row.data_ptr<int64_t>();
  auto rowptr_data = rowptr.data_ptr<int64_t>();
  auto col_data = col.data_ptr<int64_t>();

  auto stream = at::cuda::getCurrentCUDAStream();

  AT_DISPATCH_ALL_TYPES_AND(at::ScalarType::Half, mat.scalar_type(), "_", [&] {
    auto mat_data = mat.data_ptr<scalar_t>();
    auto grad_data = grad.data_ptr<scalar_t>();
    auto out_data = out.data_ptr<scalar_t>();

    AT_DISPATCH_REDUCTION_TYPES(reduce, [&] {
      spmm_value_bw_kernel<scalar_t, REDUCE><<<BLOCKS, THREADS, 0, stream>>>(
          row_data, rowptr_data, col_data, mat_data, grad_data, out_data, B, M,
          N, E, K);
    });
  });

  return out;
}

template <typename scalar_t, int TB>
__global__ void
gather_csr_kernel(const scalar_t *src_data,
                  const at::cuda::detail::TensorInfo<int32_t, int> indptr_info,
                  scalar_t *out_data, size_t N, size_t E) {

  int thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
  int row_idx = thread_idx / TB;
  int lane_idx = thread_idx % TB;

  if (row_idx < N) {
    int offset = IndexPtrToOffset<int32_t>::get(row_idx, indptr_info);
    int row_start = __ldg(indptr_info.data + offset);
    int row_end = __ldg(indptr_info.data + offset +
                        indptr_info.strides[indptr_info.dims - 1]);
    scalar_t val = __ldg(src_data + row_idx);

    offset = (row_idx / (indptr_info.sizes[indptr_info.dims - 1] - 1)) * E;
    for (int out_idx = row_start + lane_idx; out_idx < row_end; out_idx += TB) {
      out_data[offset + out_idx] = val; // "Mostly" coalesced.
    }
  }
}

template <typename scalar_t>
__global__ void gather_csr_broadcast_kernel(
    const scalar_t *src_data,
    const at::cuda::detail::TensorInfo<int32_t, int> indptr_info,
    scalar_t *out_data, size_t N, size_t K, size_t E) {

  int thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
  int row_idx = thread_idx / K;
  int lane_idx = thread_idx % K;

  if (thread_idx < N * K) {
    int offset = IndexPtrToOffset<int32_t>::get(row_idx, indptr_info);
    int row_start = __ldg(indptr_info.data + offset);
    int row_end = __ldg(indptr_info.data + offset +
                        indptr_info.strides[indptr_info.dims - 1]);

    scalar_t val = src_data[thread_idx]; // Coalesced.

    offset = (row_idx / (indptr_info.sizes[indptr_info.dims - 1] - 1)) * E * K;
    for (int out_idx = row_start; out_idx < row_end; out_idx++) {
      out_data[offset + K * out_idx + lane_idx] = val; // "Mostly" coalesced.
    }
  }
}

torch::Tensor gather_csr_cuda(torch::Tensor src, torch::Tensor indptr,
                              torch::optional<torch::Tensor> optional_out) {
  CHECK_CUDA(src);
  CHECK_CUDA(indptr);
  if (optional_out.has_value())
    CHECK_CUDA(optional_out.value());
  cudaSetDevice(src.get_device());

  CHECK_INPUT(src.dim() >= indptr.dim());

  auto sizes = indptr.sizes().vec();
  for (auto i = 0; i < indptr.dim() - 1; i++)
    sizes[i] = src.size(i);
  indptr = indptr.expand(sizes);

  auto dim = indptr.dim() - 1;
  CHECK_INPUT(src.size(dim) == 0 || src.size(dim) == indptr.size(dim) - 1);

  src = src.contiguous();

  torch::Tensor out;
  if (optional_out.has_value()) {
    out = optional_out.value().contiguous();
    for (auto i = 0; i < out.dim(); i++)
      if (i != dim)
        CHECK_INPUT(src.size(i) == out.size(i));
  } else {
    auto sizes = src.sizes().vec();
    if (src.numel() > 0) {
      sizes[dim] = indptr.flatten()[-1].cpu().data_ptr<int32_t>()[0];
    } else {
      sizes[dim] = 0;
    }
    out = torch::empty(sizes, src.options());
  }

  if (src.numel() == 0) {
    if (!optional_out.has_value())
      out.fill_(0);
    return out;
  }

  auto N = src.size(dim) * (indptr.numel() / indptr.size(-1));
  auto K = src.numel() / N;
  auto E = out.size(dim);

  auto indptr_info = at::cuda::detail::getTensorInfo<int32_t, int>(indptr);
  auto stream = at::cuda::getCurrentCUDAStream();
  AT_DISPATCH_ALL_TYPES_AND(at::ScalarType::Half, src.scalar_type(), "_", [&] {
    auto src_data = src.data_ptr<scalar_t>();
    auto out_data = out.data_ptr<scalar_t>();
    if (K == 1)
      gather_csr_kernel<scalar_t, 4><<<(4 * N + THREADS - 1) / THREADS, THREADS, 0, stream>>>(
          src_data, indptr_info, out_data, N, E);
    else
      gather_csr_broadcast_kernel<scalar_t>
          <<<(1 * N * K + THREADS - 1) / THREADS, THREADS, 0, stream>>>(src_data, indptr_info,
                                                     out_data, N, K, E);
  });

  return out;
}

__global__ void ind2ptr_kernel(const int32_t *ind_data, int32_t *out_data,
                               int64_t M, int64_t numel) {

  int64_t thread_idx = blockDim.x * blockIdx.x + threadIdx.x;

  if (thread_idx == 0) {
    for (int64_t i = 0; i <= ind_data[0]; i++)
      out_data[i] = 0;
  } else if (thread_idx < numel) {
    for (int64_t i = ind_data[thread_idx - 1]; i < ind_data[thread_idx]; i++)
      out_data[i + 1] = thread_idx;
  } else if (thread_idx == numel) {
    for (int64_t i = ind_data[numel - 1] + 1; i < M + 1; i++)
      out_data[i] = numel;
  }
}

torch::Tensor ind2ptr_cuda(torch::Tensor ind, int64_t M) {
  CHECK_CUDA(ind);
  cudaSetDevice(ind.get_device());

  auto out = torch::empty({M + 1}, ind.options());

  if (ind.numel() == 0)
    return out.zero_();

  auto ind_data = ind.data_ptr<int32_t>();
  auto out_data = out.data_ptr<int32_t>();
  auto stream = at::cuda::getCurrentCUDAStream();
  ind2ptr_kernel<<<(ind.numel() + 2 + THREADS - 1) / THREADS, THREADS, 0,
                   stream>>>(ind_data, out_data, M, ind.numel());
  return out;
}