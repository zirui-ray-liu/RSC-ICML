This is the official codes for RSC: Accelerating Graph Neural Networks Training via Randomized Sparse Computations.


## Install
This code is tested with Python 3.8 and CUDA 11.0. To reproduce the results in this paper, please follow the below configuration.


- Create and activate conda environment.

<!-- ```
torch == 1.9.0
torch_geometric == 1.7.2
torch_scatter == 2.0.8
torch_sparse == 0.6.12
``` -->

```
conda env create -f environment.yml
conda activate graph
pip install torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.0.0+cu117.html
```

- Build
```bash
cd src
pip install -v -e .
```

## Reproduce results

### Reproduce ogbn-proteins/ogbn-products results.
```bash
python ./$DATASET/train_full_batch.py --conf ./$DATASET/conf/$MODEL.yaml --tune_layer_ratio --efficient_eval --cache_inter 10 --switch_time 0.8 --dynamic_ratio
```
MODEL must be chosen from {gcn, sage, gcn2}. DATASET must be chosn from {proteins, products}.

### Combining RSC and AMP
Add the flag **--amp** to the above commends. You may need this for training GNNs on ogbn-products.



## Acknowledgment
Our code is based on the official code of [ActNN](https://arxiv.org/abs/2104.14129), [BLPA](https://github.com/ayanc/blpa), and [GNNAutoScale](https://arxiv.org/abs/2106.05609).