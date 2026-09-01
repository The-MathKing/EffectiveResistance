import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def main():
    df = pd.read_csv('benchmark_scaling.csv')
    
    plt.figure(figsize=(8, 6))
    
    # Plot Naive ER
    plt.loglog(df['N_nodes'], df['Naive_ER_time_sec'], marker='o', color='red', label='Naive ER $O(|V|^3)$')
    
    # Plot CSER
    plt.loglog(df['N_nodes'], df['CSER_time_sec'], marker='s', color='blue', label='CSER (k-core) $O(|V|+|E|)$')
    
    # Plot theoretical O(V^3) curve for reference (dashed)
    # Fit C * V^3 to the first point of Naive ER
    C = df['Naive_ER_time_sec'].iloc[0] / (df['N_nodes'].iloc[0]**3)
    v3_curve = C * (df['N_nodes']**3)
    plt.loglog(df['N_nodes'], v3_curve, linestyle='--', color='gray', label='Theoretical $O(|V|^3)$')
    
    plt.xlabel('Number of Nodes $|V|$ (ogbn-arxiv)', fontsize=14)
    plt.ylabel('Wall-clock Time (seconds)', fontsize=14)
    plt.title('Computational Scaling of Effective Resistance', fontsize=16)
    plt.legend(fontsize=12)
    plt.grid(True, which='both', linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    plt.savefig('complexity_loglog.pdf', format='pdf', dpi=300)
    print("Saved complexity_loglog.pdf")

if __name__ == '__main__':
    main()
