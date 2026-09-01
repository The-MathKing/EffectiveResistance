import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def main():
    df = pd.read_csv('attention_weights.csv')
    
    native_weights = df[df['edge_type'] == 'Native']['attention_weight']
    cser_weights = df[df['edge_type'] == 'CSER_Injected']['attention_weight']
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Plot histograms
    bins = np.linspace(0, 0.5, 50)
    ax.hist(native_weights, bins=bins, alpha=0.6, label='Native Edges', color='tab:blue', density=True)
    ax.hist(cser_weights, bins=bins, alpha=0.6, label='CSER-Injected Bridges', color='tab:red', density=True)
    
    ax.set_xlabel('Learned Attention Weight ($\\alpha$)', fontsize=14)
    ax.set_ylabel('Density', fontsize=14)
    ax.set_title('Mechanistic Ablation: GAT Attention Distribution', fontsize=16)
    
    # Add vertical lines for means
    ax.axvline(native_weights.mean(), color='darkblue', linestyle='dashed', linewidth=2)
    ax.axvline(cser_weights.mean(), color='darkred', linestyle='dashed', linewidth=2)
    
    # Add text for means
    ax.text(native_weights.mean() + 0.01, ax.get_ylim()[1]*0.8, f"Native Mean: {native_weights.mean():.3f}", color='darkblue')
    ax.text(cser_weights.mean() + 0.01, ax.get_ylim()[1]*0.6, f"CSER Mean: {cser_weights.mean():.3f}", color='darkred')
    
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.legend(fontsize=12)
    
    fig.tight_layout()
    plt.savefig('attention_histogram.pdf', format='pdf', dpi=300)
    print("Saved attention_histogram.pdf")

if __name__ == '__main__':
    main()
