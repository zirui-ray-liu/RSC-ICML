
import pdb
import numpy as np
import torch
from fastgnn import config, Scheme
from fastgnn.layers import ApproxGCNConv, ApproxSAGEConv, ApproxGCN2Conv

STEP_SIZE = 0.02

def cal_budget(r, N, out_dim, degs, mapping):
    ind = mapping[0:int(r * N)]
    return torch.sum(degs[ind]) * out_dim

def cal_loss_delta(sorted_norm_mult, r, delta_ratio, N, grad_norm, F_norm):
    e = int(N * r)
    s = int(N * (r - delta_ratio))
    return torch.sum(sorted_norm_mult[s:e]) / F_norm

def cal_budget_delta(r, N, delta_ratio, out_dim, degs, mapping):
    e = int(N * r)
    s = int(N * (r - delta_ratio))
    ind = mapping[s:e]
    return torch.sum(degs[ind]) * out_dim

def get_delta(sorted_norm_mults, ratios, N, out_dims, degs, mappings, grad_norms, F_norms):
    delta_losses = []
    delta_budgets = []
    for sorted_norm_mult, cur_ratio, out_dim, mapping, grad_norm, F_norm in zip(sorted_norm_mults, ratios, out_dims, mappings, grad_norms, F_norms):
        delta_loss = cal_loss_delta(sorted_norm_mult, cur_ratio, STEP_SIZE, N, grad_norm, F_norm)
        delta_budget = cal_budget_delta(cur_ratio, N, STEP_SIZE, out_dim, degs, mapping)
        delta_losses.append(delta_loss.item())
        delta_budgets.append(delta_budget.item())
    return delta_losses, delta_budgets

#greedy algorithm
def tune_layer_ratio(model, data, loss_op, optimizer, prior_to_train=False, skip_prior=False, minibatch=False):
    if minibatch:
        degs = Scheme.A_row_norms ** 2
    else:
        degs = data.adj_t.storage.rowcount()
    if hasattr(data, 'train_idx'):
        train_idx = data.train_idx
    else:
        train_idx = data.train_mask
    N = len(degs)
    if prior_to_train and not skip_prior:
        for _ in range(5):
            print(_)
            optimizer.zero_grad()
            # import ipdb; ipdb.set_trace()
            out = model(data.x, data.adj_t)
            if isinstance(loss_op, torch.nn.BCEWithLogitsLoss):
               loss = loss_op(out[train_idx], data.y[train_idx].to(torch.float))
            else: 
                loss = loss_op(out[train_idx], data.y[train_idx])
            loss.backward()
            optimizer.step()
    schemes, out_dims, norm_mults, sorted_norm_mults, mappings, grad_norms = [], [], [], [], [], []
    F_norms = []
    if skip_prior:
        for scheme in schemes:
        # scheme.filled = True
        # scheme.grad_norm = None
        # scheme.norm_mult = None
            scheme.sample_ratio = config.sample_ratio
        return

    for idx, conv in enumerate(model.model.convs):
        if isinstance(conv, ApproxSAGEConv) and idx == 0:
            continue
        scheme = conv.msg_and_aggr_func.scheme
        if isinstance(conv, ApproxSAGEConv):
            norm_mult = scheme.norm_mult
        else:
            norm_mult = scheme.norm_mult
        schemes.append(scheme)
        if isinstance(conv, ApproxGCN2Conv):
            out_dims.append(conv.channels)
        else:
            out_dims.append(conv.out_channels)

        norm_mults.append(norm_mult)
        sorted_norm_mult, mapping = torch.sort(norm_mult, descending=True)
        sorted_norm_mults.append(sorted_norm_mult)
        mappings.append(mapping)
        grad_norms.append(scheme.grad_norm)
        F_norms.append(scheme.F_norm)
    if minibatch:
        nnzs = int(torch.sum(Scheme.A_row_norms ** 2).item())
    else:
        nnzs = data.adj_t.nnz()
    total_budget = config.sample_ratio * nnzs * sum(out_dims)
    ratios = [torch.mean((norm_mult != 0).float()).item() for norm_mult in norm_mults]
    budget = 0
    for mapping, r, out_dim in zip(mappings, ratios, out_dims):
        budget += cal_budget(r, N, out_dim, degs, mapping)
    # print(f'initial ratio: {budget/total_budget}')
    ratios = find_ratios_st_budget_contrain(budget, total_budget, sorted_norm_mults,
                                            ratios, N, out_dims, degs, mappings, grad_norms, F_norms)
    print(ratios)
    for scheme, r in zip(schemes, ratios):
        # scheme.filled = True
        # scheme.grad_norm = None
        # scheme.norm_mult = None
        scheme.sample_ratio = r

def find_ratios_st_budget_contrain(budget, total_budget, sorted_norm_mults, ratios, N, out_dims, degs, mappings, grad_norms, F_norms):
    it = 0
    total_delta_loss = 0
    while budget > total_budget:
        delta_losses, delta_budgets = get_delta(sorted_norm_mults, ratios, N, out_dims, degs, mappings, grad_norms, F_norms)
        idx = np.argmin(delta_losses)
        if ratios[idx] > STEP_SIZE:
            ratios[idx] -= STEP_SIZE
            budget -= delta_budgets[idx]
            total_delta_loss += delta_losses[idx]
        else:
            print('terminate the tuning process')
            break
        it += 1
        # print(f'{it}-th current ratio: {budget/total_budget}, mean delta loss: {total_delta_loss/it}, total detla loss: {total_delta_loss}')
    return ratios
