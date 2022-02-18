# Layer-wise Transformer Attention Head Pruning

This repository contains a code for the paper:

**Layer-wise Pruning of Transformer Heads for Efficient Language Modeling**

Kyuhong Shim, Iksoo Choi, Wonyong Sung, Jungwook Choi

* ISOCC 2021 version (2p): https://ieeexplore.ieee.org/document/9613933
* Arxiv version (6p): https://arxiv.org/abs/2110.03252




## Summary

Attention head pruning, which removes unnecessary attention heads in the multihead attention, is a promising technique to reduce the burden of heavy Transformer computation. 
However, it does not evenly reduce the overall load because the heavy feedforward module is not affected by head pruning. 
In this work, we apply layer-wise attention head pruning on All-attention[1] Transformer so that the entire computation and the number of parameters can be reduced proportionally to the number of pruned heads.
While the architecture has the potential to fully utilize head pruning, we propose three training methods that are especially helpful to minimize performance degradation and stabilize the pruning process.
Our pruned model shows consistently lower perplexity within a comparable parameter size than TransformerXL on WikiText-103 language modeling benchmark.

[1] Augmenting Self-Attention with Persistent Memory https://arxiv.org/abs/1907.01470




## Key contributions
 
1. We implement All-attention transformer for the autoregressive language modeling.
2. We utilize a trainable method for the layer-wise attention head pruning.
3. we propose three techniques that modify the pruning process to solve the unstable behavior of during the pruning: 
   * sparsity loss warm-up
   * proper initialization
   * attention output scaling




## Performance
Our All-attention based model and Transformer-XL[2] baselines achieve almost same perplexity and the parameter size. (53.8M)
With the same parameter size, our models with attention head pruning achieve substantially better parameter efficiency than the TXL models. 
For example, pruned All-att model with 43% sparsity (30.7M) achieves similar perplexity as TXL with only 25% sparsity (47.9M).
For all sparsity levels, our method achieves much less perplexity compared to TXL.

[2] Transformer-XL: Attentive Language Models Beyond a Fixed-Length Context https://arxiv.org/abs/1901.02860 




## How to Run

You should first prepare the WikiText-103 dataset and change the dataset directory in JSON configurations.

#### Step 1. Train the baseline
This step trains the baseline All-attention transformer, which is very similar to the Transformer-XL.

```python launch.py --nproc_per_node NUM_GPUS train.py --config config/wt103_all_16l_train.json```

#### Step 2. Prune the model
This step prunes out less important attention heads from the fully converged model.

```python launch.py --nproc_per_node NUM_GPUS train_gating.py --config config/wt103_all_16l_train_gating.json```

You can control the final sparsity by changing the value `l0_coefficient` inside the config JSON.

#### Step 3. Evaluate the Speed
To evaluate the actual inference speed, you can put your pruned results into the script and run the code.

```python inference.py```
