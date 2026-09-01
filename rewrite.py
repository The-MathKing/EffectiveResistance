with open("submission.tex", "r") as f:
    text = f.read()

# 1. Fix Theorem 1 Statement
old_thm1 = r"""\begin{theorem}
Let $G^c_{n,m}(H)$ be a generalized symmetric cluster graph such that $s$ and $t$ connect identically to $\{u_1, \dots, u_n\}$ and never directly to any nodes in a symmetry-preserving internal graph $H$ defined over the $v$-nodes. For any bridge-node inflation $m$ and any symmetry-preserving internal structure $H$, the effective resistance $R(s,t)$ remains exactly $\frac{2}{n+2}$.
\end{theorem}"""
new_thm1 = r"""\begin{theorem}
Let $G^c_{n,m}(H)$ be a generalized symmetric cluster graph such that $s$ and $t$ connect identically to $\{u_1, \dots, u_n\}$ and never directly to any nodes in an arbitrary internal graph $H$ defined over the $v$-nodes. For any bridge-node inflation $m$ and any internal structure $H$, the effective resistance $R(s,t)$ remains exactly $\frac{2}{n+2}$.
\end{theorem}"""
if old_thm1 in text: text = text.replace(old_thm1, new_thm1)
else: print("Error 1")

# 2. Fix Theorem 1 Proof
old_proof1 = r"""\textbf{Proof of Theorem 1:} By the symmetry of the connections to $s$ and $t$, any automorphism of the network that swaps $s$ and $t$ fixes the internal nodes $u_i$. Consequently, the electrical potential of all internal nodes $u_i$ (and recursively any $v$ nodes connected symmetrically to them) is exactly at the midpoint potential $1/2$ (assuming $V_s=1, V_t=0$). By Kirchhoff's laws, zero current flows through any edges within $H$ between two fixed nodes at the average potential. Thus, the exact internal connectivity $H$ and the degree inflation $m$ can be pruned without affecting the current drawn. The total effective resistance reduces exactly to the direct edge $(s,t)$ in parallel with $n$ paths of length 2, yielding $R(s,t) = \frac{2}{n+2}$, completely independent of $m$ and $H$. $\blacksquare$"""
new_proof1 = r"""\textbf{Proof of Theorem 1:} By the symmetry of the connections to $s$ and $t$, the permutation $\sigma$ that swaps $s$ and $t$ and fixes all other nodes is a graph automorphism for any internal graph $H$, because no $v$-node is adjacent to $s$ or $t$. Consequently, the electrical potential of all internal nodes $u_i$ and all $v$-nodes is exactly at the midpoint potential $1/2$ (assuming $V_s=1, V_t=0$). By Kirchhoff's laws, zero current flows through any edges within $H$ between two fixed nodes at the average potential. Thus, the exact internal connectivity $H$ and the degree inflation $m$ can be pruned without affecting the current drawn. The total effective resistance reduces exactly to the direct edge $(s,t)$ in parallel with $n$ paths of length 2, yielding $R(s,t) = \frac{2}{n+2}$, completely independent of $m$ and $H$. $\blacksquare$"""
if old_proof1 in text: text = text.replace(old_proof1, new_proof1)
else: print("Error 2")

# 3. Fix Theorem 4 Statement
old_thm4 = r"""\begin{theorem}[Core-Shielded Resistance]
Let $G_k$ be the $k$-core of $G$ for $k \ge 2$. Computing the effective resistance $R_{G_k}(x,y)$ for any $x,y \in V(G_k)$ preserves the relative isometric ordering of structural bottlenecks in the core graph, strictly eliminating the $R=1.0$ saturation caused by $k=1$ leaves, while reducing the computational complexity of the Laplacian pseudoinversion from $O(|V|^3)$ to $O(|V_{core}|^3)$.
\end{theorem}"""
new_thm4 = r"""\begin{theorem}[Core-Shielded Resistance]
Let $G_\kappa$ be the $\kappa$-core of $G$ for $\kappa = 2$. Computing the effective resistance $R_{G_\kappa}(x,y)$ for any $x,y \in V(G_\kappa)$ exactly preserves the relative isometric ordering of structural bottlenecks in the core graph, strictly eliminating the $R=1.0$ saturation caused by $\kappa=1$ leaves, while reducing the computational complexity of the Laplacian pseudoinversion from $O(|V|^3)$ to $O(|V_{core}|^3)$.
\end{theorem}"""
if old_thm4 in text: text = text.replace(old_thm4, new_thm4)
else: print("Error 3")

# 4. Fix Theorem 4 Proof
old_proof4 = r"""\textbf{Proof of Theorem 4 (Core-Shielded Bounds):} Let $V_{core}$ be the set of nodes remaining after a $k$-core decomposition for $k \ge 2$, and $V_{pruned} = V \setminus V_{core}$ be the set of pruned tree-like appendages. By the definition of $k$-core, any connected component in $V_{pruned}$ attaches to $V_{core}$ through at most one node (cut-vertices), forming topological trees rooted at $V_{core}$. """
new_proof4 = r"""\textbf{Proof of Theorem 4 (Core-Shielded Bounds):} Let $V_{core}$ be the set of nodes remaining after a $\kappa$-core decomposition for $\kappa = 2$, and $V_{pruned} = V \setminus V_{core}$ be the set of pruned tree-like appendages. By the definition of a $2$-core, any connected component in $V_{pruned}$ is formed strictly by iteratively removing degree-1 leaves, meaning it attaches to $V_{core}$ through at most one node (cut-vertices), forming topological trees rooted at $V_{core}$. """
if old_proof4 in text: text = text.replace(old_proof4, new_proof4)
else: print("Error 4")

# 5. Fix Theorem 3 Proof (Remove algebraic part)
old_proof3 = r"""\textbf{Proof of Theorem 3 (The Hijack Bound):} Let $G = (V,E)$ be a connected graph. Consider a pendant-vertex (a degree-1 leaf node) $v_{leaf}$ connected by a single cut-edge $e = (u, v_{leaf})$ to the rest of the graph at node $u$. 

\textit{Electrical Network Argument:} By Kirchhoff's current law, all current entering $v_{leaf}$ must flow exclusively through the single edge $e$, meaning the current $i_{e} = 1$. By Ohm's law, the potential drop across this edge is $\Delta V = i_{e} \cdot r_{e}$. In an unweighted graph, all edges possess an intrinsic resistance of $1 \Omega$. Therefore, the potential drop $\Delta V = 1.0$. By definition, the effective resistance between $u$ and $v_{leaf}$ is exactly $1.0$. 

\textit{Algebraic Matrix Argument:} Alternatively, we can derive this exact bound algebraically. When adding $v_{leaf}$ connected to $u \in V$, the new graph $G'$ has the partitioned Laplacian matrix:
\[
L' = \begin{bmatrix}
L + e_u e_u^T & -e_u \\
-e_u^T & 1
\end{bmatrix}
\]
where $e_u$ is the standard basis vector corresponding to node $u$. To compute $R(u, v_{leaf})$, we apply block matrix inversion via the Schur complement. The Schur complement of the lower right scalar block $1$ is $S = (L + e_u e_u^T) - (-e_u)(1)^{-1}(-e_u^T) = L$. By the Sherman-Morrison rank-1 update identity on the pseudoinverse $(L')^+$, the exact relative potential elements corresponding to $u$ and $v_{leaf}$ yield exactly:
\[
R(u, v_{leaf}) = (L^+_{u,u} + 1) + L^+_{u,u} - 2L^+_{u,u} = 1.0
\]

By Rayleigh's Monotonicity Law, the effective resistance between any two adjacent nodes in an unweighted graph is bounded strictly by $R \le 1.0$. Because the cut-edge $(u, v_{leaf})$ mathematically achieves this absolute global maximum, any naive global top-$k$ thresholding operation will prioritize this leaf-edge over any structural internal edge $(x,y)$ where $R(x,y) < 1.0$. Thus, $k$ leaf nodes perfectly saturate a budget of $k$ edges, completely blinding the metric. $\blacksquare$"""

new_proof3 = r"""\textbf{Proof of Theorem 3 (The Hijack Bound):} Let $G = (V,E)$ be a connected graph. Consider a pendant-vertex (a degree-1 leaf node) $v_{leaf}$ connected by a single cut-edge $e = (u, v_{leaf})$ to the rest of the graph at node $u$. 

By Kirchhoff's current law, all current entering $v_{leaf}$ must flow exclusively through the single edge $e$, meaning the current $i_{e} = 1$. By Ohm's law, the potential drop across this edge is $\Delta V = i_{e} \cdot r_{e}$. In an unweighted graph, all edges possess an intrinsic resistance of $1 \Omega$. Therefore, the potential drop $\Delta V = 1.0$. By definition, the effective resistance between $u$ and $v_{leaf}$ is exactly $1.0$. 

By Rayleigh's Monotonicity Law, the effective resistance between any two adjacent nodes in an unweighted graph is bounded strictly by $R \le 1.0$. Because the cut-edge $(u, v_{leaf})$ mathematically achieves this absolute global maximum, any naive global top-$q$ rank thresholding operation will prioritize this leaf-edge over any structural internal edge $(x,y)$ where $R(x,y) < 1.0$. Thus, $q$ leaf nodes perfectly saturate a budget of $q$ edges, completely blinding the metric. $\blacksquare$"""
if old_proof3 in text: text = text.replace(old_proof3, new_proof3)
else: print("Error 5")

# 6. Replace k-core globally and fix minor edits
text = text.replace(r"$k$-core", r"$\kappa$-core")
text = text.replace(r"k=2", r"\kappa=2")
text = text.replace(r"k \ge 2", r"\kappa = 2")
text = text.replace(r"k \le", r"\kappa \le")
text = text.replace(r"k$-core", r"\kappa$-core")
text = text.replace(r"79 undirected edges in the Texas LCC (295 total edges)", r"79 undirected edges in the Texas LCC (279 total edges)")
text = text.replace(r"budget = 74 edges", r"budget = 69 edges")
text = text.replace(r"budget \tau", r"budget $\tau$")

# 7. Axis clarification
text = text.replace(r"\caption{Wall-clock execution time demonstrating the exponential divergence of the naive pseudoinverse against the $\kappa$-core CSER filter on \texttt{ogbn-arxiv}.}", r"\caption{Wall-clock execution time demonstrating the exponential divergence of the naive pseudoinverse against the $\kappa$-core CSER filter on subgraphs sampled from \texttt{ogbn-arxiv}.}")
text = text.replace(r"\caption{Peak RAM allocation tracking on \texttt{ogbn-arxiv}.", r"\caption{Peak RAM allocation tracking on subgraphs sampled from \texttt{ogbn-arxiv}.")

# 8. Ground Truth GCN
old_gt = r"""Following the formalization of the information bottleneck by \citet{Alon2021}, we define the information flow across an edge $e=(u,v)$ as the Frobenius norm of the Jacobian $\partial h_u^{(K)} / \partial x_v^{(0)}$ from a $K$-layer Graph Convolutional Network."""
new_gt = r"""Following the formalization of the information bottleneck by \citet{Alon2021}, we define the information flow across an edge $e=(u,v)$ as the Frobenius norm of the Jacobian $\partial h_u^{(K)} / \partial x_v^{(0)}$ from an untrained, randomly initialized $K$-layer Graph Convolutional Network. To ensure these ground-truth bottlenecks are structurally intrinsic rather than artifacts of a single initialization, we average the Jacobian norms across 10 distinct random seeds."""
if old_gt in text: text = text.replace(old_gt, new_gt)
else: print("Error 8")

# 9. Table CIs
text = text.replace(r"Bootstrapped 95\% CI misses $[90.0\% - 98.9\%]$ on Texas", r"Bootstrapped 95\% CI misses $[89.4\% - 100.0\%]$ on Texas")
text = text.replace(r"Bootstrapped 95\% CI misses only $[7.8\% - 21.1\%]$ on Texas", r"Bootstrapped 95\% CI misses only $[4.2\% - 21.3\%]$ on Texas")

with open("submission.tex", "w") as f:
    f.write(text)

print("Done")
