import pandas as pd
import matplotlib.pyplot as plt
import os

def main():
    if not os.path.exists('memory_scaling.csv'):
        print("memory_scaling.csv not found")
        return
        
    df = pd.read_csv('memory_scaling.csv')
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Plot Naive ER
    ax.plot(df['N_nodes'], df['naive_er_gb'], marker='o', color='tab:red', linewidth=2, label=r'Naive ER ($O(|V|^3)$ Laplacian Pseudoinverse)')
    
    # Plot CSER
    ax.plot(df['N_nodes'], df['cser_gb'], marker='s', color='tab:blue', linewidth=2, label=r'CSER ($O(|V|+|E|)$ $k$-core Filter)')
    
    ax.set_xscale('log')
    ax.set_yscale('log')
    
    ax.set_xlabel('Graph Size (Number of Nodes $|V|$)', fontsize=14)
    ax.set_ylabel('Peak RAM Allocation (GB)', fontsize=14)
    ax.set_title('Memory Scalability: Naive vs CSER', fontsize=16)
    
    ax.grid(True, which="both", ls="--", alpha=0.5)
    ax.legend(fontsize=12, loc='upper left')
    
    # Add a horizontal line for standard consumer hardware (e.g. 16GB)
    ax.axhline(y=16.0, color='black', linestyle=':', linewidth=2, label='16GB RAM Limit')
    ax.text(1200, 18, 'Standard 16GB RAM Limit', fontsize=10, color='black')
    
    fig.tight_layout()
    plt.savefig('memory_footprint.pdf', format='pdf', dpi=300)
    print("Saved memory_footprint.pdf")

if __name__ == '__main__':
    main()
