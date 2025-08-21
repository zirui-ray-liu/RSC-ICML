import pdb
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import numpy as np
import torch
import torch_sparse
from torch.nn import functional as F
from torch_sparse.matmul import matmul as baseline_matmul
from fastgnn.spmm import ApproxMatmul
from timeit_v2 import py_benchmark
from fastgnn import get_memory_usage, compute_tensor_bytes, approx_matmul, get_A_row_norms
from fastgnn.conf import config
from fastgnn.op import approxlinear
from data import get_data
import torch_geometric.transforms as T
from torch_geometric.nn.conv.gcn_conv import gcn_norm
from fastgnn import Scheme

# config.sample_ratio controls the number of sampled column-row pairs. 
# sample_ratio = 0.1 means we only utilize 10% of the total column-row pairs
config.sample_ratio = 0.3
# Scheme.k is the cache interval, k=10 means we sample the graph every 10 iterations
Scheme.k = 10

data, _, C = get_data('~/data', 'proteins')
# data = T.ToSparseTensor()(data)
data.adj_t = data.adj_t.set_diag()
data.adj_t = gcn_norm(data.adj_t, add_self_loops=False)
adj_t = data.adj_t
N, D = data.x.shape[0], 256
Scheme.A_row_norms = get_A_row_norms(adj_t)
Scheme.A_row_norms = Scheme.A_row_norms.cuda()


# B = torch.randn((N, D), device='cuda')
# tmp1 = baseline_matmul(adj_t.cuda(), B, 'mean')
# tmp2 = baseline_matmul(adj_t.cuda(), B, 'sum')
# deg = Scheme.A_row_norms  ** 2
# pdb.set_trace()

def test_linear_speed():
    print("========== Linear Speed Test ==========")
    n = N
    d = D
    for dtype in ['float32']:
        print(f"test {dtype}...")
        data_np = np.random.randn(n, d).astype(dtype)
        w_np = np.random.randn(d, d).astype(dtype)
        b_np = np.random.randn(d).astype(dtype)
        def test_implementation(func):
            data = torch.tensor(data_np).to("cuda").requires_grad_()
            weight = torch.tensor(w_np).to("cuda").requires_grad_()
            bias = torch.tensor(b_np).to("cuda").requires_grad_()
            stmt = "func(data, weight, bias)"
            t_forward = py_benchmark(stmt, {**globals(), **locals()},
                                     setup="torch.cuda.synchronize()", finish="torch.cuda.synchronize()")

            output = func(data, weight, bias)
            head = torch.ones_like(output)
            stmt = "output.backward(head, retain_graph=True)"
            t_backward = py_benchmark(stmt, {**globals(), **locals()},
                                      setup="torch.cuda.synchronize()", finish="torch.cuda.synchronize()")

            return t_forward, t_backward

        forward_ref, backward_ref = test_implementation(F.linear)
        forward_us, backward_us = test_implementation(approxlinear.apply)

        print("Exact.     forward: %.2f ms\tbackward: %.2f ms\tsum: %.2f ms" %
              (forward_ref * 1e3, backward_ref * 1e3, (forward_ref + backward_ref) * 1e3))
        print("Approximated. forward: %.2f ms\tbackward: %.2f ms\tsum: %.2f ms" %
              (forward_us * 1e3, backward_us * 1e3, (forward_us + backward_us) * 1e3))



def test_linear_correctness():
    print("========== Linear Correctness Test ==========")
    data_np = np.random.randn(100000, 256).astype('float32')
    data = torch.tensor(data_np).to("cuda")
    # data = arxiv_data.x.continguous()
    ce = torch.nn.CrossEntropyLoss().cuda()
    y = torch.empty(100000, dtype=torch.long).random_(4).cuda()

    def test_implementation(func, weight, bias):
        pred = func(data, weight, bias)
        pred = F.relu(pred)
        pred = pred.reshape(pred.shape[0], 4, pred.shape[1] // 4).mean(2)
        loss = ce(pred, y)
        weight.grad = None
        bias.grad = None
        loss.backward()
        return weight.grad.cpu().numpy()

    w = torch.randn((256, 256), requires_grad=True, device='cuda')
    b = torch.randn((256,), requires_grad=True, device='cuda')
    qw = torch.randn((256, 256), requires_grad=True, device='cuda')
    qb = torch.randn((256,), requires_grad=True, device='cuda')
    with torch.no_grad():
        qw.copy_(w)
        qb.copy_(b)
    true_grad = test_implementation(F.linear, w, b)
    grads = []
    for i in range(10):
        grads.append(test_implementation(approxlinear.apply, qw, qb))
    grads = np.stack(grads, 0)
    grad_mean = grads.mean(0)
    grad_std = grads.std(0)
    bias = np.linalg.norm(grad_mean - true_grad)
    print('Grad = {}, Bias = {}, Std = {}'.format(np.linalg.norm(true_grad), bias, np.linalg.norm(grad_std)))


def test_linear_memory():
    print("========== Linear Memory Test ==========")
    data_np = np.random.randn(169343, 256).astype('float32')
    w = torch.randn((256, 256), requires_grad=True, device='cuda')
    b = torch.randn((256,), requires_grad=True, device='cuda')
    qw = torch.randn((256, 256), requires_grad=True, device='cuda')
    qb = torch.randn((256,), requires_grad=True, device='cuda')
    with torch.no_grad():
        qw.copy_(w)
        qb.copy_(b)

    def test_implementation(func, weight, bias, n_layers):
        data = torch.tensor(data_np).to("cuda").requires_grad_()
        output = data

        before = get_memory_usage(0)

        for i in range(n_layers):
            output = func(output, weight, bias)

        after = get_memory_usage(0) - compute_tensor_bytes([output])
        if func == F.linear:
            after += compute_tensor_bytes([data])

        return after - before

    usage_ref = test_implementation(F.linear, w, b, 5)
    usage_us = test_implementation(approxlinear.apply, qw, qb, 5)
    print("5 layer: Exact.     Usage: %.2f MB" % (usage_ref / 2 ** 20))
    print("5 layer: Approximated. Usage: %.2f MB" % (usage_us / 2 ** 20))
    print("5 layer: Ratio: %.2f" % (usage_ref / usage_us))
    print("")

    usage_ref = test_implementation(F.linear, w, b, 10)
    usage_us = test_implementation(approxlinear.apply, qw, qb, 10)
    print("10 layer: Exact.     Usage: %.2f MB" % (usage_ref / 2 ** 20))
    print("10 layer: Approximated. Usage: %.2f MB" % (usage_us / 2 ** 20))
    print("10 layer: Ratio: %.2f" % (usage_ref / usage_us))
    print("")


def test_spmm_matmul_correctness():
    print('============test spmm correctness============')
    N, D = 10000, 128
    nnz = 50000
    i = torch.randint(high=N, size=(2, nnz), dtype=torch.int64)
    v = torch.randn(size=(nnz,))
    dense_mat_cpu = torch.randn(N, D)
    def test_implementation_has_value(func, reduce):
        v_ = v.clone().requires_grad_()
        torch_sp = torch.sparse_coo_tensor(i, v_, [N, N])
        tsp = torch_sparse.SparseTensor.from_torch_sparse_coo_tensor(torch_sp).cuda()
        dense_mat = dense_mat_cpu.cuda().requires_grad_()
        output = func(tsp, dense_mat, reduce)
        output.backward(torch.ones_like(output))
        return [x.detach().cpu().numpy() for x in [output, dense_mat.grad, v_.grad]]

    def test_implementation_non_value(func, reduce):
        tsp = torch_sparse.SparseTensor.from_edge_index(i, sparse_sizes=[N, N]).cuda()
        dense_mat = dense_mat_cpu.cuda().requires_grad_()
        output = func(tsp, dense_mat, reduce)
        output.backward(torch.ones_like(output))
        return [x.detach().cpu().numpy() for x in [output, dense_mat.grad]]

    for reduce in ['sum']:
        print(f'============test spmm {reduce} correctness============')
        output_ref, grad_data_ref, grad_value_ref = test_implementation_has_value(torch_sparse.matmul, reduce)
        output_ref_nonvalue, grad_data_ref_nonvalue  = test_implementation_non_value(torch_sparse.matmul, reduce)
        value_grads, data_grads = [], []
        for _ in range(10):
            output_us, grad_data_us, grad_value_us = test_implementation_has_value(approx_matmul, reduce)
            np.testing.assert_allclose(output_ref, output_us)
            value_grads.append(grad_value_us)
            data_grads.append(grad_data_us)
        value_grads = np.stack(value_grads, 0)
        value_grads_mean = value_grads.mean(0)
        value_grads_std = value_grads.std(0)
        bias = np.linalg.norm(value_grads_mean - grad_value_ref)
        print('(value.requires_grad = True) Value Grad = {}, Bias = {}, Std = {}'.format(np.linalg.norm(grad_value_ref), bias, np.linalg.norm(value_grads_std)))

        data_grads = np.stack(data_grads, 0)
        data_grads_mean = data_grads.mean(0)
        data_grads_std = data_grads.std(0)
        bias = np.linalg.norm(data_grads_mean - grad_data_ref)
        print('(value.requires_grad = True) Data Grad = {}, Bias = {}, Std = {}'.format(np.linalg.norm(grad_data_ref), bias, np.linalg.norm(data_grads_std)))

        data_grads = []
        for _ in range(10):
            output_us, grad_data_us = test_implementation_non_value(approx_matmul, reduce)
            np.testing.assert_allclose(output_ref_nonvalue, output_us)
            data_grads.append(grad_data_us)

        data_grads = np.stack(data_grads, 0)
        data_grads_mean = data_grads.mean(0)
        data_grads_std = data_grads.std(0)
        bias = np.linalg.norm(data_grads_mean - grad_data_ref_nonvalue)
        print('(value.requires_grad = False) Data Grad = {}, Bias = {}, Std = {}'.format(
            np.linalg.norm(grad_data_ref_nonvalue), bias, np.linalg.norm(data_grads_std)))

# def test_spmm_matmul_speed():
#     N, D = int(1e6), 256
#     nnz = int(1e5)
#     i = torch.randint(high=N, size=(2, nnz), dtype=torch.int64)
#     v = torch.randn(size=(nnz,))
#     dense_mat_cpu = np.random.rand(N, D)
#     def test_implementation_non_value(func, reduction):
#         tsp = torch_sparse.SparseTensor.from_edge_index(i, sparse_sizes=[N, N]).cuda()
#         dense_mat = torch.tensor(dense_mat_cpu).to("cuda").requires_grad_()
#         stmt = "func(tsp, dense_mat, reduction)"
#         t_forward = py_benchmark(stmt, {**globals(), **locals()},
#                                     setup="torch.cuda.synchronize()", finish="torch.cuda.synchronize()")

#         output = func(tsp, dense_mat)
#         head = torch.ones_like(output, dtype=torch.float)
#         stmt = "output.backward(head, retain_graph=True)"
#         t_backward = py_benchmark(stmt, {**globals(), **locals()},
#                                     setup="torch.cuda.synchronize()", finish="torch.cuda.synchronize()")
#         return t_forward, t_backward

#     for reduction in ['sum', 'mean']:
#         print(f"========== SpMM {reduction} Speed Test ==========")
#         forward_ref, backward_ref = test_implementation_non_value(baseline_matmul, reduction)
#         forward_us, backward_us = test_implementation_non_value(approx_matmul, reduction)
#         print("Exact.     forward: %.2f ms\tbackward: %.2f ms\tsum: %.2f ms" %
#                 (forward_ref * 1e3, backward_ref * 1e3, (forward_ref + backward_ref) * 1e3))
#         print("Approximated. forward: %.2f ms\tbackward: %.2f ms\tsum: %.2f ms" %
#                 (forward_us * 1e3, backward_us * 1e3, (forward_us + backward_us) * 1e3))

def test_spmm_matmul_speed():
    dense_mat_cpu = np.random.rand(N, D)
    def test_implementation_non_value(func, reduction):
        # tsp = torch_sparse.SparseTensor.from_edge_index(i, sparse_sizes=[N, N]).cuda()
        tsp = adj_t.cuda()
        dense_mat = torch.tensor(dense_mat_cpu).float().to("cuda").requires_grad_()
        if isinstance(func, ApproxMatmul):
            stmt = "func(tsp, dense_mat)"
        else:
            stmt = "func(tsp, dense_mat, reduction)"
        t_forward = py_benchmark(stmt, {**globals(), **locals()},
                                    setup="torch.cuda.synchronize()", finish="torch.cuda.synchronize()")

        output = func(tsp, dense_mat)
        head = torch.randn_like(output, dtype=torch.float)
        stmt = "output.backward(head, retain_graph=True)"
        t_backward = py_benchmark(stmt, {**globals(), **locals()},
                                    setup="torch.cuda.synchronize()", finish="torch.cuda.synchronize()")
        return t_forward, t_backward

    for reduction in ['sum']:
        print(f"========== SpMM {reduction} Speed Test ==========")
        approx_matmul = ApproxMatmul(reduction)
        forward_ref, backward_ref = test_implementation_non_value(baseline_matmul, reduction)
        forward_us, backward_us = test_implementation_non_value(approx_matmul, reduction)
        print("Exact.     forward: %.2f ms\tbackward: %.2f ms\tsum: %.2f ms" %
                (forward_ref * 1e3, backward_ref * 1e3, (forward_ref + backward_ref) * 1e3))
        print("Approximated. forward: %.2f ms\tbackward: %.2f ms\tsum: %.2f ms" %
                (forward_us * 1e3, backward_us * 1e3, (forward_us + backward_us) * 1e3))

def test_get_row_norm_correctness():
    row = [0, 0, 0, 1, 1, 2, 2, 2, 3]
    col = [2, 4, 7, 3, 6, 0, 4, 7, 0]  
    val = [1., 2., 3., 4. ,5. ,6. ,7., 8. ,9.] 
    cuda0 = torch.device('cuda:0')
    row, col, val = torch.tensor(row, dtype=torch.long, device=cuda0), torch.tensor(col,  dtype=torch.long, device=cuda0), torch.tensor(val, device=cuda0),
    edge_index = torch.cat([row.view(1, -1), col.view(1, -1)], dim=0)
    sp_tensor = torch_sparse.SparseTensor.from_edge_index(edge_index)
    row_norms = get_A_row_norms(sp_tensor)
    assert torch.sum(torch.abs(row_norms - torch.tensor([3., 2., 3., 1.], device=cuda0))) == 0.
    sp_tensor = sp_tensor.set_value(val, layout='coo')
    row_norms = get_A_row_norms(sp_tensor)
    assert torch.sum(torch.abs(row_norms - torch.tensor([14., 41., 149., 81.], device=cuda0))) == 0.


def test_spmm_matmul_memory():
    print("========== Spmm Memory Test ==========")
    dense_mat_cpu = np.random.rand(N, D)
    def test_implementation(func, n_layers, reduce):
        tsp = adj_t.cuda()
        tsp.fill_cache_()
        before = get_memory_usage(0)
        data = torch.tensor(dense_mat_cpu).float().to("cuda").requires_grad_()
        output = data
        print(before / 2 ** 20)

        for i in range(n_layers):
            if isinstance(func, ApproxMatmul):
                output = func(tsp, output)
            else:
                output = func(tsp, output, reduce)

        after = get_memory_usage(0)
        
        return after - before
    
    for reduce in ['mean']:
        approx_matmul = ApproxMatmul(reduce)
        print(f'============test spmm {reduce} Memory============')    
        usage_ref = test_implementation(torch_sparse.matmul, 1, reduce)
        usage_us = test_implementation(approx_matmul, 1, reduce)
        print("Exact.     Usage: %.2f MB" % (usage_ref / 2 ** 20))
        print("Our. Usage: %.2f MB" % (usage_us / 2 ** 20))
        print("")


if __name__ == "__main__":
    # test_linear_correctness()
    # test_linear_memory()
    # test_linear_speed()
    test_spmm_matmul_speed()
    # test_spmm_matmul_memory()
    # test_get_row_norm_correctness() 