import re

with open('overleaf_project/submission.tex', 'r') as f:
    text = f.read()

# Eq (2) wrong-signed: "lower indicates a bottleneck" -> "higher indicates a bottleneck"
text = text.replace("where lower indicates a bottleneck", "where higher indicates a bottleneck")

# Table 2 caption: "50 random initialization seeds" -> "50 random initialization seeds (5 seeds \times 10 splits per seed)"
text = text.replace("50 random initialization seeds.", "50 random initialization seeds (5 seeds $\\times$ 10 CV splits). Differences are not statistically significant (paired t-test, $p > 0.05$).")

# "scales geometrically" -> "scales quadratically"
text = text.replace("scales geometrically", "scales quadratically")

# "tau > 0.01 risks" -> "tau > 0.05 risks"
text = text.replace("\\tau > 0.01 risks", "\\tau > 0.05 risks")

# ogbn-arxiv edges: 1,166,243 -> 1,157,799
text = text.replace("1,166,243", "1,157,799")

# Degree heuristic: "significantly outperformed by a trivial heuristic" -> "significantly outperformed on AUROC by a trivial heuristic"
text = text.replace("significantly outperformed by a trivial heuristic", "significantly outperformed on AUROC by a trivial heuristic")

# Limitations paragraph: "and homophily diagnostic (Figure \\ref{fig:diagnostic}) are each based on a single training run" -> "is based on a single training run"
# Need to be careful here
text = text.replace("Furthermore, the GAT attention analysis (Figure \\ref{fig:attention}) and homophily diagnostic (Figure \\ref{fig:diagnostic}) are each based on a single training run per dataset", "Furthermore, the GAT attention analysis (Figure \\ref{fig:attention}) is based on a single training run per dataset")
text = text.replace("we do not report variance across seeds for these two analyses.", "we do not report variance across seeds for this analysis.")

# DiffWire citation context
# In related work: "Other proposed solutions include Discrete Ricci Curvature \citep{DiffWire}," -> remove DiffWire here
# And add it as commute time
text = text.replace("Other proposed solutions include Discrete Ricci Curvature \\citep{DiffWire}", "Other proposed solutions include Discrete Ricci Curvature")

with open('overleaf_project/submission.tex', 'w') as f:
    f.write(text)

with open('overleaf_project/main.bib', 'r') as f:
    bib = f.read()

# Fix Fesser venue
bib = bib.replace("journal={LoG}", "journal={LoG (PMLR 231)}")
# Fix Di Giovanni venue - wait, in main.bib it's TMLR 2024. Let's make sure:
# The reviewer said "Di Giovanni et al. 2023 listed as ICLR; "How does over-squashing affect the power of GNNs?" is TMLR 2024."
# Wait, let's see if Di Giovanni is cited elsewhere.

with open('overleaf_project/main.bib', 'w') as f:
    f.write(bib)

