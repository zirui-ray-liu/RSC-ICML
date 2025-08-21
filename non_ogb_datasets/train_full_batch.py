import sys
import os
import numpy as np
from torch.autograd import grad
sys.path.append(os.getcwd())
import argparse
import random
import time
import warnings
import yaml

import torch
import torch.nn.functional as F
import torch.nn.parallel
import torch.backends.cudnn as cudnn
from torch.cuda.amp import autocast, GradScaler

from torch_geometric.nn.conv.gcn_conv import gcn_norm
from torch_sparse import SparseTensor, matmul, fill_diag, sum as sparsesum, mul

from fastgnn import get_memory_usage, compute_tensor_bytes, exp_recorder, config, ApproxModule, get_A_row_norms, Scheme, tune_layer_ratio
import models
from data import get_data
from fastgnn import Logger
from sklearn.metrics import f1_score
import torch_geometric.transforms as T
from train_utils import get_optimizer

MB = 1024**2
GB = 1024**3


parser = argparse.ArgumentParser()
parser.add_argument('--conf', type=str, required=True, 
                    help='the path to the configuration file')
parser.add_argument('--dataset', type=str, required=True, 
                    help='the name of the applied dataset')
parser.add_argument('--root', type=str, default='../data')
parser.add_argument('--reorder', action='store_true')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--gpu', default=0, type=int,
                    help='GPU id to use.')
parser.add_argument('--num_workers', type=int, default=12)
parser.add_argument('--sample_ratio', type=float, default=1.0)
parser.add_argument('--runs', type=int, default=10)
parser.add_argument('--grad_norm', type=float, default=None)
parser.add_argument('--debug_mem', action='store_true')
parser.add_argument('--test_speed', action='store_true')
parser.add_argument('--amp', help='whether to enable apx mode', action='store_true')
parser.add_argument('--tune_layer_ratio', action='store_true', help='whether to tune the layer-wise ratio')
parser.add_argument('--tune_inter', type=int, default=10)
parser.add_argument('--save_grads', action='store_true', default=False)
parser.add_argument('--dynamic_ratio', action='store_true', default=False)
parser.add_argument('--last_ratio', type=float, default=1.0)
parser.add_argument('--switch_time', type=float, default=1.0)
parser.add_argument('--cache_inter', type=int, default=1)


def preprocess_data(model_config, data, model):
    loop, normalize = model_config['loop'], model_config['normalize']
    if config.use_approx_op:
        loop = False

    if loop:
        t = time.perf_counter()
        print('Adding self-loops...', end=' ', flush=True)
        data.adj_t = data.adj_t.set_diag()
        print(f'Done! [{time.perf_counter() - t:.2f}s]')
    
    if normalize:
        t = time.perf_counter()
        print('Normalizing data...', end=' ', flush=True)
        deg = sparsesum(data.adj_t, dim=1)
        if config.use_approx_op:
            deg = deg + 1.0
        deg_inv_sqrt = deg.pow_(-0.5)
        data.adj_t = mul(data.adj_t, deg_inv_sqrt.view(-1, 1))
        data.adj_t = mul(data.adj_t, deg_inv_sqrt.view(1, -1))
        deg_inv = deg_inv_sqrt.pow_(2).view(-1, 1).cuda()
        if config.use_approx_op:
            for conv in model.model.convs:
                conv._deg_inv = deg_inv
        print(f'Done! [{time.perf_counter() - t:.2f}s]')


def train(model, optimizer, data, loss_op, grad_norm, scaler, amp_mode):
    model.train()
    optimizer.zero_grad()
    with autocast(enabled=amp_mode):
        out = model(data.x, data.adj_t)
        loss = loss_op(out[data.train_mask], data.y[data.train_mask])
    del data
    if amp_mode:
        scaler.scale(loss).backward()
        if grad_norm is not None:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_norm)
        scaler.step(optimizer)
        scaler.update()
    else:
        loss.backward()
        if grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_norm)
        optimizer.step()
    return loss.item()


def compute_micro_f1(logits, y, mask=None) -> float:
    if mask is not None:
        logits, y = logits[mask], y[mask]

    if y.dim() == 1:
        return int(logits.argmax(dim=-1).eq(y).sum()) / y.size(0)
        
    else:
        y_pred = logits > 0
        y_true = y > 0.5

        tp = int((y_true & y_pred).sum())
        fp = int((~y_true & y_pred).sum())
        fn = int((y_true & ~y_pred).sum())

        try:
            precision = tp / (tp + fp)
            recall = tp / (tp + fn)
            return 2 * (precision * recall) / (precision + recall)
        except ZeroDivisionError:
            return 0.

@torch.no_grad()
def test(model, data, amp_mode):
    model.eval()
    with autocast(enabled=amp_mode):
        out = model(data.x, data.adj_t)
    y_true = data.y
    train_acc = compute_micro_f1(out, y_true, data.train_mask)
    valid_acc = compute_micro_f1(out, y_true, data.val_mask)
    test_acc = compute_micro_f1(out, y_true, data.test_mask)
    return train_acc, valid_acc, test_acc


def main():
    global args 
    args = parser.parse_args()
    with open(args.conf, 'r') as fp:
        model_config = yaml.load(fp, Loader=yaml.FullLoader)
        name = model_config['name']
        loop = model_config.get('loop', False)
        normalize = model_config.get('norm', False)
        if args.dataset == 'reddit2':
            model_config = model_config['params']['reddit']
        else:
            model_config = model_config['params'][args.dataset]
        model_config['name'] = name
        model_config['loop'] = loop
        model_config['normalize'] = normalize
    print(args)
    print(f'model config: {model_config}')
    if args.dataset == 'yelp':
        multi_label = True
    else:
        multi_label = False
    print(f'clipping grad norm: {args.grad_norm}')
    args.model = model_config['arch_name']
    assert model_config['name'] in ['GCN', 'SAGE', 'GCN2']
    if args.amp:
        print('activate amp mode')
        config.amp = True
        scaler = GradScaler()
    else:
        config.amp = False
        scaler = None
    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    if args.gpu is not None:
        print("Use GPU {} for training".format(args.gpu))
    if args.reorder:
        print('reorder the input graph using rabbit reorder...')
        config.reorder = True

    if args.tune_layer_ratio:
        config.tune_layer_ratio = True
    
    if args.save_grads:
        Scheme.save_grads = True
        Scheme.model = model_config['arch_name'].lower()
        Scheme.dataset = args.dataset

        
    torch.cuda.set_device(args.gpu)
    data, num_features, num_classes = get_data(args.root, args.dataset, args.reorder)
    Scheme.num_samples = data.num_nodes
    GNN = getattr(models, model_config['arch_name'])
    model = GNN(in_channels=num_features, out_channels=num_classes, **model_config['architecture'])
    loss_op = F.binary_cross_entropy_with_logits if multi_label else F.cross_entropy

    if args.cache_inter > 1:
        Scheme.k = args.cache_inter
        print(f'sample the graph every {Scheme.k} epoch')

    if args.sample_ratio <= 1.0:
        print(f'sample ratio: {args.sample_ratio}')
        config.sample_ratio = args.sample_ratio
        config.use_approx_op = True 
    else:
        config.use_approx_op = False
    if config.use_approx_op:
        print('convert the model')
        model = ApproxModule(model)

    print(model)
    model.cuda(args.gpu)

    if args.debug_mem:
        print("========== Model and Optimizer only ===========")
        optimizer = get_optimizer(model_config, model)
        optimizer.zero_grad()
        model.reset_parameters()
        model.train()
        usage = get_memory_usage(args.gpu, False)
        exp_recorder.record("network", args.model)
        exp_recorder.record("model_only", usage / MB, 4)
        print("========== Load data to GPU ===========")
        print('converting data form...')
        s_time = time.time()
        data = T.ToSparseTensor()(data.to('cuda'))
        print(f'done. used {time.time() - s_time} sec')
        preprocess_data(model_config, data, model)
        Scheme.A_row_norms = get_A_row_norms(data.adj_t)
        data.adj_t.fill_cache_()
        init_mem = get_memory_usage(args.gpu, False)
        data_mem = init_mem / MB - exp_recorder.val_dict['model_only']
        exp_recorder.record("data", init_mem / MB - exp_recorder.val_dict['model_only'], 4)
        if config.tune_layer_ratio:
            print('tune layer ratio...')
            t = time.perf_counter()
            optimizer = get_optimizer(model_config, model)
            tune_layer_ratio(model, data, loss_op, optimizer)
            print(f'Done! [{time.perf_counter() - t:.2f}s]')
        out = model(data.x, data.adj_t)[data.train_mask]
        loss = loss_op(out, data.y[data.train_mask])
        print("========== Before Backward ===========")
        before_backward = get_memory_usage(args.gpu, True)
        act_mem = get_memory_usage(args.gpu, False) - init_mem - compute_tensor_bytes([loss, out])

        res = "Total Mem: %.2f MB\tData Mem: %.2f MB\tAct Mem: %.2f MB" % (before_backward / MB,
                                                                           data_mem,
                                                                           act_mem / MB)
        print(res)
        loss.backward()
        optimizer.step()
        del loss, out
        print("========== After Backward ===========")
        after_backward = get_memory_usage(args.gpu, True)
        total_mem = before_backward + (after_backward - init_mem)
        res = "Total Mem: %.2f MB\tData Mem: %.2f MB\tAct Mem: %.2f MB" % (total_mem / MB,
                                                                           data_mem,
                                                                           act_mem / MB)
        print(res)
        exp_recorder.record("total", total_mem / MB, 2)
        exp_recorder.record("activation", act_mem / MB, 2)
        # exp_recorder.dump('mem_results.json')
        s_time = time.time()
        if args.test_speed:
            model.reset_parameters()
            optimizer.zero_grad()
            epoch_per_sec = []
            for i in range(100):
                t = time.time()
                torch.cuda.synchronize()
                optimizer.zero_grad()
                with autocast(enabled=config.amp):
                    out = model(data.x, data.adj_t)
                    loss = loss_op(out[data.train_mask], data.y[data.train_mask])
                loss.backward()
                optimizer.step()
                torch.cuda.synchronize()
                duration = time.time() - t
                epoch_per_sec.append(duration)
                # print(f'epoch {i}, duration: {duration} sec')
            print(f's/epoch: {np.mean(epoch_per_sec)}')
            print(f'training epoch/s: {100/(time.time() - s_time) }')
            model.eval()
            s_time = time.time()
            torch.cuda.synchronize()
            with torch.no_grad():
                for _ in range(100):
                    out = model(data.x, data.adj_t)           
            torch.cuda.synchronize()
            print(f'inference epoch/s: {100/(time.time() - s_time) }') 
        exit()

    print('converting data form...')
    s_time = time.time()
    data = T.ToSparseTensor()(data.to('cuda'))
    print(f'done. used {time.time() - s_time} sec')

    preprocess_data(model_config, data, model)

    Scheme.A_row_norms = get_A_row_norms(data.adj_t)
    logger = Logger(args.runs, args)
    if config.tune_layer_ratio:
        print('tune layer ratio...')
        t = time.perf_counter()
        optimizer = get_optimizer(model_config, model)
        tune_layer_ratio(model, data, loss_op, optimizer, prior_to_train=True)
        print(f'Done! [{time.perf_counter() - t:.2f}s]')
    for run in range(args.runs):
        config.sample_ratio = args.sample_ratio
        model.reset_parameters()
        optimizer = get_optimizer(model_config, model)
        duration_time = []
        for epoch in range(1, 1 + model_config['epochs']):
            epoch_start = time.time()
            if args.dynamic_ratio and epoch == int(model_config['epochs'] * args.switch_time):
                print(f'switch ratio at epoch {epoch}, fine tune ratio: {args.last_ratio}')
                config.sample_ratio = args.last_ratio
            if config.tune_layer_ratio and epoch % args.tune_inter == 0 and config.sample_ratio < 1.0:
                print('tune layer ratio...')
                t = time.perf_counter()
                optimizer = get_optimizer(model_config, model)
                tune_layer_ratio(model, data, loss_op, optimizer)
                print(f'Done! [{time.perf_counter() - t:.2f}s]')
            loss = train(model, optimizer, data, loss_op, args.grad_norm, scaler, args.amp)
            duration_time.append(time.time() - epoch_start)
            result = test(model, data, args.amp)
            logger.add_result(run, result)
            train_acc, valid_acc, test_acc = result
            print(f'Run: {run + 1:02d}, '
                   f'Epoch: {epoch:02d}, '
                   f'Train f1: {100 * train_acc:.2f}%, '
                   f'Valid f1: {100 * valid_acc:.2f}% '
                   f'Test f1: {100 * test_acc:.2f}%')
        print(f'run {run}, total time: {np.sum(duration_time)}')
        logger.add_result(run, result)
        logger.print_statistics(run)
    logger.print_statistics()


if __name__ == '__main__':
    main()