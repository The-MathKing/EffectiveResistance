import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def main():
    plt.rcParams.update({'font.size': 20, 'axes.labelsize': 22, 'axes.titlesize': 24, 'xtick.labelsize': 20, 'ytick.labelsize': 20, 'legend.fontsize': 16})
    df = pd.read_csv('benchmark_scaling.csv')
    
    plt.figure(figsize=(8, 6))
    
    # Plot Naive ER
    plt.loglog(df['N_nodes'], df['Naive_ER_time_sec'], marker='o', color='red', label='Naive ER $O(|V|^3)$')
    
    # Plot CSER
    plt.plot(df['N_nodes'], df['CSER_time_sec'], marker='s', color='tab:green', linewidth=2, label='CSER (k-core + Core Inversion)')
    
    # Plot theoretical O(V^3) curve for reference (dashed)
    # Fit C * V^3 to the first point of Naive ER
    C = df['Naive_ER_time_sec'].iloc[0] / (df['N_nodes'].iloc[0]**3)
    v3_curve = C * (df['N_nodes']**3)
    plt.loglog(df['N_nodes'], v3_curve, linestyle='--', color='gray', label='Theoretical $O(|V|^3)$')
    
    plt.xlabel('Number of Nodes $|V|$ (ogbn-arxiv)')
    plt.ylabel('Wall-clock Time (seconds)')
    plt.title('Computational Scaling of Effective Resistance')
    plt.legend()
    plt.grid(True, which='both', linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    plt.savefig('ogb_complexity.pdf', format='pdf', dpi=300, bbox_inches='tight')
    print("Saved ogb_complexity.pdf")

if __name__ == '__main__':
    main()
