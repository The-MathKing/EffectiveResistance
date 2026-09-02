import re

with open('overleaf_project/submission.tex', 'r') as f:
    text = f.read()

# 1. Abstract \citet to \citep
abstract_match = re.search(r'\\begin\{abstract\}(.*?)\\end\{abstract\}', text, re.DOTALL)
if abstract_match:
    abstract = abstract_match.group(1)
    # the reviewer said: "fails to detect over-squashing inside dense clusters Fesser & Weber (2023)..."
    # let's just do a blanket replace of \citet with \citep in abstract
    new_abstract = abstract.replace('\\citet', '\\citep')
    # Spielman & Srivastava miscited in the abstract:
    # "proposed as a globally-aware alternative to local curvature \citep{Spielman2011}."
    # Let's remove the citep{Spielman2011} there, or replace it if it's there.
    # Actually I will just replace the abstract manually.

with open('overleaf_project/submission.tex', 'w') as f:
    f.write(text)
