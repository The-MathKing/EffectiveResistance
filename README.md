# Effective Resistance and Curvature Have Distinct Blind Spots to Intra-Cluster Over-squashing

This repository contains the official evaluation codebase, raw empirical data, and analysis scripts for our study on geometric metrics for bottleneck detection in Graph Neural Networks (GNNs).

## Overview

Over-squashing in GNNs limits the capture of long-range dependencies. While local metrics like Weighted Augmented Forman-3 (WAF3) curvature and global metrics like Effective Resistance (ER) have been proposed to identify structural bottlenecks, our rigorous empirical evaluation on real-world graphs reveals fundamental limitations for both:

1. **Intra-cluster over-squashing is empirically rare:** In our low-betweenness evaluation, true bottlenecks are almost non-existent. A rigorous power analysis demonstrates that evaluating metrics on this topological regime is statistically underpowered.
2. **WAF3 succeeds on inter-cluster bridges:** WAF3 properly flags true structural bottlenecks between graph clusters, achieving a robust out-of-sample Missed Over-Squashing Ratio (MOSR) of ~13% on Texas.
3. **Effective Resistance collapses via Leaf-Node Hijacking:** Effective Resistance completely fails on inter-cluster bridges (MOSR ~95%). We identify the mechanism as **Leaf-Node Hijacking**: over 26% of edges in these graphs are cut-edges ($R=1.0$), of which the vast majority are trivial degree-1 leaf appendages. The top 25% threshold budget is completely saturated by floating-point tie-breaking among these irrelevant leaves, entirely blinding ER to the true structural bridges connecting the core clusters.

## Repository Structure

- `src/`: Core evaluation pipeline.
  - `metrics.py`: Implementations of WAF3, Effective Resistance, Commute Time, and Link Resistance Curvature.
  - `jacobian.py`: GNN Jacobian norm computations to establish ground-truth over-squashing.
  - `mosr.py`: Implementation of the Missed Over-Squashing Ratio (MOSR) thresholding metric.
- `scratch/`: One-off analysis and validation scripts.
  - `bootstrap_mosr.py`: Generates 200-sample bootstrap 95% Confidence Intervals for robust out-of-sample claims.
  - `verify_leaf_node_295.py` & `clean_ablation.py`: Validates the leaf-node hijacking mechanism and performs a clean ablation by excluding leaf-adjacent edges from the $q=25$ flagging budget.

## Pipeline 1: Raw Benchmarking (Diagnostic)
To evaluate the catastrophic collapse of Effective Resistance on unpruned graphs due to leaf-node hijacking, run:
```bash
python scratch/dump_csv.py
```

## Pipeline 2: Core-Shielded Effective Resistance (CSER) [Recommended]
To deploy the prescriptive algorithmic solution, which shields the cubic-time $O(|V|^3)$ spectral inversion from threshold saturation by applying a strict linear-time $O(|V|+|E|)$ $k$-core filter ($k \ge 2$), run:
```bash
python src/core_shielded_pipeline.py --dataset Texas
```
This mathematically guarantees the elimination of dangling tie-breaking noise, restoring the viability of Effective Resistance as a global bottleneck detector.

- `*.json` / `*.csv` / `*.txt`: Raw serialized output counts, logged results, and exact cross-sectional benchmarks for full reproducibility.

## Requirements

Dependencies include standard Python ML and Graph libraries:
- `torch`
- `torch_geometric`
- `networkx`
- `numpy`

## Note on Commute Time

We explicitly omit Commute Time results from the main comparisons because the relationship $C(u,v) = 2|E|R(u,v)$ is a *definitional accounting identity* by construction. Any rank-based thresholding metric (such as MOSR) is mathematically forced to yield identical empirical results for both Commute Time and Effective Resistance.
