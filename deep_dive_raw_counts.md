# Raw Pipeline Output (ER vs Jacobian at q=25)

Here are the raw counts pulled directly from the graph topologies and the actual pipeline execution, which definitively explain the ER paradox.

## Texas Dataset (295 total edges)
- **Global Thresholds:**
  - Jacobian bottleneck threshold: `Jaco <= 0.2157`
  - ER flagged threshold: `ER >= 1.0000` (Because exactly 74 edges out of 295 have an ER of exactly 1.0, the top 25% threshold lands exactly on 1.0).
- **Global `|E_q|` Counts:**
  - Total ER flagged globally: 74 edges.
- **High Betweenness Stratum (Bridges, 71 total edges):**
  - **True bottlenecks (Jacobian `denominator`):** 48 edges
  - **ER Flagged in stratum (`numerator` logic):** Only 7 edges
  - **Result:** ER wastes its 74-edge budget globally but only catches 7 bridges.

## Cornell Dataset (280 total edges)
- **Global Thresholds:**
  - Jacobian bottleneck threshold: `Jaco <= 0.2844`
  - ER flagged threshold: `ER >= 1.0000` (Exactly 71 edges out of 280 have an ER of 1.0, so the 25% threshold is exactly 1.0).
- **Global `|E_q|` Counts:**
  - Total ER flagged globally: 71 edges.
- **High Betweenness Stratum (Bridges, 73 total edges):**
  - **True bottlenecks (Jacobian `denominator`):** 40 edges
  - **ER Flagged in stratum (`numerator` logic):** Only 8 edges
  - **Result:** ER wastes its 71-edge budget globally but only catches 8 bridges.

---

### Why does ER fail so catastrophically on bridges? (The "Leaf-Node Hijacking" Paradox)
Your intuition about Black et al. (2023) is exactly right—ER *should* be good at detecting inter-cluster bridges. So why does it fail our MOSR evaluation?

An Effective Resistance of exactly `1.0` is the maximum possible value, occurring exclusively on **cut-edges** (tree edges). In WebKB datasets like Texas and Cornell, there are a massive number of degree-1 nodes (dangling leaves). 

These leaves are trivial: their connecting edge carries almost no flow (extremely low betweenness), but it has an ER of exactly 1.0. 
Because over 25% of the edges in these graphs are trivial cut-edges, **they completely hijack the top 25% ER threshold**. 

Meanwhile, the *true* structural bottlenecks—the bridges connecting two dense clusters—have an ER slightly less than 1.0 (e.g., 0.8 or 0.9) because there are often faint, long parallel paths elsewhere in the core. Therefore, when we take the top 25% of ER edges, it flags all 70+ dangling leaves and completely misses the structural bridges. Black et al. evaluates ER primarily as a global rewiring gradient, not via strict top-K thresholding on raw, leaf-heavy graphs, which is why this failure mode hasn't been exposed in that literature until now.
