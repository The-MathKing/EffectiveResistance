# Anonymous GitHub Repository for Submission

This repository contains the official evaluation codebase, raw empirical data, and analysis scripts for our study on geometric metrics for bottleneck detection in Graph Neural Networks (GNNs).

## Overview

Over-squashing in GNNs limits the capture of long-range dependencies. While local metrics like Weighted Augmented Forman-3 (WAF3) curvature and global metrics like Effective Resistance (ER) have been proposed to identify structural bottlenecks, our rigorous empirical evaluation on real-world graphs reveals fundamental limitations for both. We establish that Effective Resistance suffers from Cut-Edge Saturation on trivial appendages.

## Repository Structure

- `src/`: Core evaluation pipeline.
  - `metrics.py`: Implementations of WAF3, Effective Resistance, Commute Time, and Link Resistance Curvature.
  - `jacobian.py`: GNN Jacobian norm computations to establish ground-truth over-squashing.
  - `mosr.py`: Implementation of the Missed Over-Squashing Ratio (MOSR) thresholding metric.
