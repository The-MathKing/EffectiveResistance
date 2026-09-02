import json
import matplotlib.pyplot as plt
import os
import numpy as np

def main():
    if not os.path.exists('results/benchmark_results.json'):
        print("results/benchmark_results.json not found")
        return
        
    with open('results/benchmark_results.json', 'r') as f:
        data = json.load(f)
        
    datasets = []
    fraction_pendant = []
    mosr_collapse = [] # MOSR of Effective Resistance on All edges, q=25
    
    for ds_name, ds_results in data.items():
        if 'fraction_pendant' in ds_results and 'All' in ds_results:
            datasets.append(ds_name)
            fraction_pendant.append(ds_results['fraction_pendant'])
            
            # Extract MOSR of ER (neg) for q=25
            mosr_val = ds_results['All']['q=25']['Eff. Resistance (neg)']['mosr']
            mosr_collapse.append(mosr_val)
            
    plt.rcParams.update({'font.size': 20, 'axes.labelsize': 22, 'axes.titlesize': 24, 'xtick.labelsize': 20, 'ytick.labelsize': 20, 'legend.fontsize': 16})
    plt.figure(figsize=(8, 6))
    
    plt.scatter(fraction_pendant, mosr_collapse, s=150, color='tab:red', alpha=0.8, edgecolors='k')
    
    for i, ds in enumerate(datasets):
        xytext = (0, 15)
        if ds == 'Cora':
            xytext = (-20, 15)
        elif ds == 'CiteSeer':
            xytext = (20, 15)
            
        plt.annotate(ds, (fraction_pendant[i], mosr_collapse[i]), 
                     textcoords="offset points", xytext=xytext, ha='center', fontsize=16)
                     
    # Add a trendline
    if len(datasets) > 1:
        z = np.polyfit(fraction_pendant, mosr_collapse, 1)
        p = np.poly1d(z)
        x_vals = np.linspace(min(fraction_pendant)*0.9, max(fraction_pendant)*1.1, 100)
        plt.plot(x_vals, p(x_vals), 'r--', alpha=0.5, label='Trend')
        
    plt.xlabel('Fraction of Pendant Edges')
    plt.ylabel('MOSR of Effective Resistance ($q=25$)')
    plt.title('Topological Vulnerability of ER')
    plt.grid(True, linestyle='--', alpha=0.5)
    if len(datasets) > 1:
        plt.legend()
    
    plt.tight_layout()
    plt.savefig('mosr_collapse_vs_pendant.pdf', format='pdf', dpi=300, bbox_inches='tight')
    print("Saved mosr_collapse_vs_pendant.pdf")

if __name__ == '__main__':
    main()
