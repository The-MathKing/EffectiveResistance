import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

def main():
    if not os.path.exists('diagnostic_sweep_new.csv'):
        print("diagnostic_sweep_new.csv not found")
        return
        
    plt.rcParams.update({'font.size': 20, 'axes.labelsize': 22, 'axes.titlesize': 24, 'xtick.labelsize': 20, 'ytick.labelsize': 20, 'legend.fontsize': 16})
    df = pd.read_csv('diagnostic_sweep_new.csv')
    
    plt.figure(figsize=(10, 8))
    
    # Color points based on homophily threshold
    colors = ['green' if h > 0.75 else 'orange' if h > 0.3 else 'red' for h in df['Homophily']]
    
    plt.errorbar(df['Homophily'], df['Delta_Accuracy'], yerr=df['Delta_Std'], fmt='none', ecolor='gray', alpha=0.5, capsize=5)
    plt.scatter(df['Homophily'], df['Delta_Accuracy'], c=colors, s=150, alpha=0.8, edgecolors='k')
    
    # Add labels
    for i, row in df.iterrows():
        xytext = (0, 15)
        if row['Dataset'] == 'Squirrel':
            xytext = (0, 15)
        elif row['Dataset'] == 'Chameleon':
            xytext = (0, -25)
        elif row['Dataset'] == 'Cornell':
            xytext = (-30, 15)
        elif row['Dataset'] == 'Texas':
            xytext = (30, 15)
            
        plt.annotate(row['Dataset'], (row['Homophily'], row['Delta_Accuracy']), 
                     textcoords="offset points", xytext=xytext, ha='center', fontsize=16)
                     
    # Add horizontal line at 0
    plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    
    plt.xlabel('Edge Homophily Ratio ($h$)')
    plt.ylabel(r'$\Delta$ Accuracy (CSER - Raw) %')
    plt.title('Homophily-based Diagnostic for Rewiring')
    plt.grid(True, linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    plt.savefig('diagnostic_scatter.pdf', format='pdf', dpi=300)
    print("Saved diagnostic_scatter.pdf")

if __name__ == '__main__':
    main()
