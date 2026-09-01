import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

def main():
    if not os.path.exists('diagnostic_sweep.csv'):
        print("diagnostic_sweep.csv not found")
        return
        
    df = pd.read_csv('diagnostic_sweep.csv')
    
    plt.figure(figsize=(8, 6))
    
    # Color points based on homophily threshold
    colors = ['green' if h > 0.75 else 'orange' if h > 0.3 else 'red' for h in df['Homophily']]
    
    plt.scatter(df['Homophily'], df['Delta_Accuracy'], c=colors, s=100, alpha=0.8, edgecolors='k')
    
    # Add labels
    for i, row in df.iterrows():
        plt.annotate(row['Dataset'], (row['Homophily'], row['Delta_Accuracy']), 
                     textcoords="offset points", xytext=(0,10), ha='center')
                     
    # Add vertical line at h=0.75
    plt.axvline(x=0.75, color='gray', linestyle='--', label=r'Observed separation, $h \approx 0.75$')
    
    # Add horizontal line at 0
    plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    
    plt.xlabel('Edge Homophily Ratio ($h$)', fontsize=14)
    plt.ylabel(r'$\Delta$ Accuracy (CSER - Raw) %', fontsize=14)
    plt.title('Homophily-based Diagnostic for Rewiring', fontsize=16)
    plt.legend(fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    plt.savefig('diagnostic_scatter.pdf', format='pdf', dpi=300)
    print("Saved diagnostic_scatter.pdf")

if __name__ == '__main__':
    main()
