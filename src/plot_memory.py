import pandas as pd
import matplotlib.pyplot as plt
import os

def main():
    if not os.path.exists('memory_scaling.csv'):
        print("memory_scaling.csv not found")
        return
        
    plt.rcParams.update({'font.size': 20, 'axes.labelsize': 22, 'axes.titlesize': 24, 'xtick.labelsize': 20, 'ytick.labelsize': 20, 'legend.fontsize': 16})
    df = pd.read_csv('memory_scaling.csv')
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Plot Naive ER
    ax.plot(df['N_nodes'], df['naive_er_gb'], marker='o', color='tab:red', linestyle='--', linewidth=2, label=r'Naive Dense Pseudoinverse $O(|V|^2)$')
    
    # Plot CSER
    ax.plot(df['N_nodes'], df['cser_gb'], marker='s', color='tab:blue', linestyle='-', linewidth=2, label='CSER (2-Core Shielded)')
    
    ax.set_xscale('log')
    ax.set_yscale('log')
    
    ax.set_xlabel('Graph Size (Number of Nodes $|V|$)')
    ax.set_ylabel('Peak RAM Allocation (GB)')
    ax.set_title('Memory Scalability: Naive vs CSER')
    
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.legend(fontsize=14, loc='lower right')
    
    # Add a horizontal line for standard consumer hardware (e.g. 16GB)
    ax.axhline(y=16.0, color='black', linestyle=':', linewidth=2, label='16GB RAM Limit')
    ax.text(1200, 18, 'Standard 16GB RAM Limit', fontsize=16, color='black')
    
    fig.tight_layout()
    plt.savefig('ogb_memory.pdf', format='pdf', dpi=300, bbox_inches='tight')
    print("Saved memory_footprint.pdf")

if __name__ == '__main__':
    main()
