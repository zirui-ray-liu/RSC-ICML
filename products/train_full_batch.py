import sys
import os
import numpy as np
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
from torch_sparse import SparseTensor, matmul, fill_diag, sum as sparsesum, mul

from torch_geometric.nn.conv.gcn_conv import gcn_norm

from fastgnn import get_memory_usage, compute_tensor_bytes, exp_recorder, config, ApproxModule, get_A_row_norms, Scheme, tune_layer_ratio
from fastgnn.utils import cast_int32
import models
from data import get_data
from fastgnn import Logger
from sklearn.metrics import f1_score
import torch_geometric.transforms as T
from train_utils import get_optimizer
from ogb.nodeproppred import Evaluator

MB = 1024**2
GB = 1024**3


parser = argparse.ArgumentParser()
parser.add_argument('--conf', type=str, required=True, 
                    help='the path to the configuration file')
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


def get_optimizer(model_config, model):
    if model_config['optim'] == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=model_config['lr'])
    elif model_config['optim'] == 'rmsprop':
        optimizer = torch.optim.RMSprop(model.parameters(), lr=model_config['lr'])
    else:
        raise NotImplementedError
    if model_config['optim'] == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=model_config['lr'])
    elif model_config['optim'] == 'rmsprop':
        optimizer = torch.optim.RMSprop(model.parameters(), lr=model_config['lr'])
    else:
        raise NotImplementedError
    return optimizer

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
    import pdb; pdb.set_trace()
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
    return loss


@torch.no_grad()
def test(model, data, evaluator, amp_mode=False):
    model.eval()
    with autocast(enabled=amp_mode):
        out = model(data.x, data.adj_t)
    y_pred = out.argmax(dim=-1, keepdim=True)
    y_true = data.y.unsqueeze(1)
    train_acc = evaluator.eval({
        'y_true': y_true[data.train_mask],
        'y_pred': y_pred[data.train_mask],
    })['acc']
    valid_acc = evaluator.eval({
        'y_true': y_true[data.val_mask],
        'y_pred': y_pred[data.val_mask],
    })['acc']
    test_acc = evaluator.eval({
        'y_true': y_true[data.test_mask],
        'y_pred': y_pred[data.test_mask],
    })['acc']
    return train_acc, valid_acc, test_acc


def main():
    args = parser.parse_args()
    with open(args.conf, 'r') as fp:
        model_config = yaml.load(fp, Loader=yaml.FullLoader)
    args.model = model_config['name']
    assert args.model.lower() in ['gcn', 'sage', 'gat']
    print(args)
    if args.sample_ratio < 1.0:
        print(f'sample ratio: {args.sample_ratio}')
        config.sample_ratio = args.sample_ratio
        config.use_approx_op = True 
    else:
        config.use_approx_op = False

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


    if args.amp:
        config.amp = True
        print(f'amp mode: {config.amp}')

    if args.tune_layer_ratio:
        config.tune_layer_ratio = True

    if args.save_grads:
        Scheme.save_grads = True
        Scheme.model = args.model.lower()
        Scheme.dataset = 'products'
    if args.cache_inter > 1:
        Scheme.k = args.cache_inter
        print(f'sample the graph every {Scheme.k} epoch')

    grad_norm = model_config.get('grad_norm', None)
    print(f'clipping grad norm: {grad_norm}')
    torch.cuda.set_device(args.gpu)

    data, num_features, num_classes = get_data(args.root, 'products')
    data = T.ToSparseTensor()(data)
    evaluator = Evaluator(name='ogbn-products')
    logger = Logger(args.runs, args)
    GNN = getattr(models, args.model)
    model = GNN(in_channels=num_features, out_channels=num_classes, **model_config['architecture'])
    loss_op = F.cross_entropy

    if config.use_approx_op:
        print('convert the model')
        model = ApproxModule(model)
    preprocess_data(model_config, data, model)
    print(model)
    model.cuda(args.gpu)

    if args.debug_mem:
        print("========== Model Only ===========")
        usage = get_memory_usage(args.gpu, True)
        exp_recorder.record("network", 'GCN')
        exp_recorder.record("model_only", usage / MB, 2)
        print("========== Load data to GPU ===========")
        data.adj_t.fill_cache_()
        data = data.to('cuda')
        Scheme.A_row_norms = get_A_row_norms(data.adj_t)
        init_mem = get_memory_usage(args.gpu, True)
        data_mem = init_mem / MB - exp_recorder.val_dict['model_only']
        exp_recorder.record("data", init_mem / MB - exp_recorder.val_dict['model_only'], 2)
        model.reset_parameters()
        model.train()
        optimizer = get_optimizer(model_config, model)
        optimizer.zero_grad()
        if config.tune_layer_ratio:
            print('tune layer ratio...')
            t = time.perf_counter()
            optimizer = get_optimizer(model_config, model)
            tune_layer_ratio(model, data, loss_op, optimizer)
            print(f'Done! [{time.perf_counter() - t:.2f}s]')

        out = model(data.x, data.adj_t)[data.train_mask]
        loss = F.nll_loss(out, data.y.squeeze(1)[data.train_mask])
        print(f'max allocated mem (MB): {torch.cuda.max_memory_allocated(0) / MB}')
        print("========== Before Backward ===========")
        del out
        before_backward = get_memory_usage(args.gpu, True)
        act_mem = get_memory_usage(args.gpu, False) - init_mem - compute_tensor_bytes([loss])
        res = "Total Mem: %.2f MB\tData Mem: %.2f MB\tAct Mem: %.2f MB" % (before_backward / MB,
                                                                           data_mem,
                                                                           act_mem / MB)
        print(res) 
        loss.backward()
        optimizer.step()
        del loss
        print("========== After Backward ===========")
        after_backward = get_memory_usage(args.gpu, True)
        total_mem = before_backward + (after_backward - init_mem)
        res = "Total Mem: %.2f MB\tData Mem: %.2f MB\tAct Mem: %.2f MB" % (total_mem / MB,
                                                                           data_mem,
                                                                           act_mem / MB)
        print(f'max allocated mem (MB): {torch.cuda.max_memory_allocated(0) / MB}')
        print(res)
        exp_recorder.record("total", total_mem / MB, 2)
        exp_recorder.record("activation", act_mem / MB, 2)
        exp_recorder.dump('mem_results.json')
        s_time = time.time()
        torch.cuda.synchronize()
        if args.test_speed:
            model.reset_parameters()
            optimizer.zero_grad()
            epoch_per_sec = []

            for i in range(100):
                t = time.time()
                optimizer.zero_grad()
                out = model(data.x, data.adj_t)[data.train_mask]
                loss = F.nll_loss(out, data.y.squeeze(1)[data.train_mask])
                loss.backward()
                optimizer.step()
                duration = time.time() - t
                epoch_per_sec.append(duration)
                print(f'epoch {i}, duration: {duration} sec')
            torch.cuda.synchronize()
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

    data = data.to('cuda')
    # ======================================================================
    # Note: if you want to train the model with 32-bit graph structure data,
    # uncomment the below code.

    # cast_int32(data)

    # ======================================================================
    data.y = data.y.squeeze()
    Scheme.A_row_norms = get_A_row_norms(data.adj_t)
    if config.tune_layer_ratio:
        print('tune layer ratio...')
        t = time.perf_counter()
        optimizer = get_optimizer(model_config, model)
        tune_layer_ratio(model, data, loss_op, optimizer, prior_to_train=True, skip_prior=True)
        print(f'Done! [{time.perf_counter() - t:.2f}s]')
    for run in range(args.runs):
        config.sample_ratio = args.sample_ratio
        model.reset_parameters()
        optimizer = get_optimizer(model_config, model)
        if args.amp:
            print('activate amp mode')
            scaler = GradScaler()
        else:
            scaler = None
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
            torch.cuda.synchronize()
            cur_epoch_time = time.time() - epoch_start
            duration_time.append(cur_epoch_time)
            # ===========================
            result = test(model, data, evaluator, args.amp)
            logger.add_result(run, result)
            if  model_config['log_steps'] > 0 and epoch % model_config['log_steps'] == 0:
                train_acc, valid_acc, test_acc = result
                print(f'Run: {run + 1:02d}, '
                      f'Epoch: {epoch:02d}, '
                      f'Loss: {loss.item():.4f}, '
                      f'Train: {100 * train_acc:.2f}%, '
                      f'Valid: {100 * valid_acc:.2f}% '
                      f'Test: {100 * test_acc:.2f}%',
                      f'epoch time: {cur_epoch_time}')
        print(f'run {run}, total time: {np.sum(duration_time)}')
        logger.print_statistics(run)
    logger.print_statistics()


if __name__ == '__main__':
    main()