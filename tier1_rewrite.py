import re

with open("submission.tex", "r") as f:
    text = f.read()

# Abstract
old_abs = r"...while naive Effective Resistance collapses catastrophically. We prove this collapse is caused by a ``Leaf-Node Hijacking'' mechanism"
new_abs = r"...while naive Effective Resistance collapses catastrophically on these benchmarks (MOSR up to 0.96). We prove this collapse is caused by a ``Leaf-Node Hijacking'' mechanism"
if old_abs in text: text = text.replace(old_abs, new_abs)
else: print("Error 1")

# Sec 4.5
old_gat = r"As depicted in Figure \ref{fig:attention}, the GAT mechanism correctly identifies the statistical properties of the injected bridges. While CSER forces these connections to exist topologically to bypass the Laplacian inversion bottleneck, the GAT dynamically down-weights their aggregation influence compared to native edges (mean $\alpha$ of $0.030$ vs. $0.222$). This 7.3x suppression provides a deep, mechanistic explanation for why architectures utilizing parameterized neighborhood attention safely leverage geometric rewiring pipelines without suffering from catastrophic noise aggregation, whereas uniform samplers (like GraphSAGE) are structurally forced to aggregate heterophilic noise."
new_gat = r"As depicted in Figure \ref{fig:attention}, the GAT mechanism correctly identifies the statistical properties of the injected bridges. While CSER forces these connections to exist topologically to bypass the Laplacian inversion bottleneck, the GAT dynamically down-weights their aggregation influence compared to native edges. On CiteSeer, we observe a 7.3x gap between mean attention weight on native edges (0.222) vs. CSER-injected bridges (0.030), suggesting GAT's attention mechanism can partially down-weight injected long-range edges. We do not claim this generalizes to other architectures or datasets without further study; GraphSAGE's uniform aggregation has no equivalent down-weighting mechanism by construction, which is consistent with (but not directly tested as an explanation for) its comparatively larger accuracy drop under rewiring in Table \ref{tab:gnn_benchmarks}."
if old_gat in text: text = text.replace(old_gat, new_gat)
else: print("Error 2")

# Sec 8 Conclusion
old_conc = r"We mandate that future GNN rewiring pipelines must deploy linear-time appendage pruning---such as our proposed Core-Shielded Effective Resistance---before executing cubic-time spectral measurements, ensuring optimal hardware utilization and uncorrupted structural thresholding."
new_conc = r"Our results suggest that GNN rewiring pipelines using Effective Resistance should apply linear-time appendage pruning (such as CSER) before spectral measurement, at least on graphs with WebKB-like leaf-node prevalence, ensuring optimal hardware utilization and uncorrupted structural thresholding."
if old_conc in text: text = text.replace(old_conc, new_conc)
else: print("Error 3")

# Broader Impact
old_bi = r"Future implementations of geometric rewiring must aggressively prune tree-like appendages to minimize unnecessary carbon footprint and hardware utilization."
new_bi = r"Future implementations of geometric rewiring should consider pruning tree-like appendages to minimize unnecessary carbon footprint and hardware utilization."
if old_bi in text: text = text.replace(old_bi, new_bi)
else: print("Error 4")

# Homophily
old_homo = r"The results generate a clear predictive boundary at $h = 0.75$. On networks where $h > 0.75$ (e.g., Cora), CSER rewiring safely improves geometric routing. However, on highly heterophilic graphs (e.g., Chameleon, $h \approx 0.23$), geometric rewiring injects destructive noise, slightly degrading performance. Thus, $h = 0.75$ serves as a strict diagnostic gate dictating when structural rewiring should be deployed."
new_homo = r"Across the 8 datasets tested, a homophily ratio around $h \approx 0.75$ separates datasets where CSER improved accuracy from those where it didn't (Figure \ref{fig:diagnostic}). This is a descriptive pattern in a small sample of datasets, not a validated decision rule---we have not tested it on held-out datasets or established its robustness to seed variance, and it should be treated as a hypothesis for future work rather than a deployment gate."
if old_homo in text: text = text.replace(old_homo, new_homo)
else: print("Error 5")

# Table 2 addition
old_table2 = r"""Intra-cluster over-squashing is empirically rare & Low-Betweenness denominators & Statistically underpowered. We cannot distinguish performance from chance. \\ \hline
\end{tabular}%"""
new_table2 = r"""Intra-cluster over-squashing is empirically rare & Low-Betweenness denominators & Statistically underpowered. We cannot distinguish performance from chance. \\
GAT attention down-weights injected edges & CiteSeer attention log (7.3x gap) & Single-seed observation on CiteSeer; requires broader validation. \\
Homophily ratio $h \approx 0.75$ separates success & Evaluated across 8 benchmark datasets & Descriptive hypothesis only; lacks seed variance and held-out validation. \\ \hline
\end{tabular}%"""
if old_table2 in text: text = text.replace(old_table2, new_table2)
else: print("Error 6")

# Limitations addition
old_limit = r"Our conclusions regarding intra-cluster metrics are thus limited by the scarcity of real-world ground truth to evaluate against."
new_limit = r"Our conclusions regarding intra-cluster metrics are thus limited by the scarcity of real-world ground truth to evaluate against. Furthermore, the GAT attention analysis (Figure \ref{fig:attention}) and homophily diagnostic (Figure \ref{fig:diagnostic}) are each based on a single training run per dataset; unlike the MOSR and end-to-end classification results (Tables \ref{tab:graph_class} and \ref{tab:gnn_benchmarks}), we do not report variance across seeds for these two analyses. We flag this as a difference in evidentiary strength between these diagnostic figures and the paper's primary quantitative claims."
if old_limit in text: text = text.replace(old_limit, new_limit)
else: print("Error 7")

# Rename n=40 to N=40
old_n40 = r"On Cornell ($n=40$)"
new_n40 = r"On Cornell ($N=40$)"
if old_n40 in text: text = text.replace(old_n40, new_n40)
else: print("Error 8")

# Verify Texas edges reverted to 295
text = text.replace(r"79 undirected edges in the Texas LCC (279 total edges)", r"79 undirected edges in the Texas LCC (295 total edges)")
text = text.replace(r"budget = 69 edges", r"budget = 74 edges")

with open("submission.tex", "w") as f:
    f.write(text)

print("Done")
