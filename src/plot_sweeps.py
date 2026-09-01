import pandas as pd
import matplotlib.pyplot as plt
import os

def plot_tau():
    df = pd.read_csv('sweeps_tau.csv')
    fig, ax1 = plt.subplots(figsize=(10, 8))

    color = 'tab:blue'
    ax1.set_xlabel(r'Rewiring Budget ($\tau$)')
    ax1.set_ylabel(r'Spectral Gap ($\lambda_2$)', color=color)
    ax1.plot(df['budget_tau'], df['spectral_gap'], marker='o', color=color, linestyle='-', linewidth=2)
    ax1.tick_params(axis='y', labelcolor=color)

    ax2 = ax1.twinx()  
    color = 'tab:red'
    ax2.set_ylabel('GCN Test Accuracy (%)', color=color)
    ax2.plot(df['budget_tau'], df['accuracy'], marker='s', color=color, linestyle='--', linewidth=2)
    ax2.tick_params(axis='y', labelcolor=color)

    plt.title(r'Effect of Rewiring Budget ($\tau$) on Spectral Gap & Accuracy', pad=20)
    plt.grid(True, linestyle='--', alpha=0.5)
    fig.tight_layout()
    plt.savefig('sweeps_tau.pdf', format='pdf', dpi=300)
    print("Saved sweeps_tau.pdf")

def plot_k():
    df = pd.read_csv('sweeps_k.csv')
    fig, ax1 = plt.subplots(figsize=(10, 8))

    color = 'tab:blue'
    ax1.set_xlabel(r'$\kappa$-Core Threshold ($\kappa$)')
    ax1.set_ylabel(r'Spectral Gap ($\lambda_2$)', color=color)
    ax1.plot(df['k_core'], df['spectral_gap'], marker='o', color=color, linestyle='-', linewidth=2)
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.set_xticks(df['k_core'])

    ax2 = ax1.twinx()  
    color = 'tab:red'
    ax2.set_ylabel('GCN Test Accuracy (%)', color=color)
    ax2.plot(df['k_core'], df['accuracy'], marker='s', color=color, linestyle='--', linewidth=2)
    ax2.tick_params(axis='y', labelcolor=color)

    plt.title(r'Effect of $\kappa$-Core Filter on Spectral Gap & Accuracy', pad=20)
    plt.grid(True, linestyle='--', alpha=0.5)
    fig.tight_layout()
    plt.savefig('sweeps_k.pdf', format='pdf', dpi=300)
    print("Saved sweeps_k.pdf")

if __name__ == '__main__':
    plt.rcParams.update({'font.size': 20, 'axes.labelsize': 22, 'axes.titlesize': 24, 'xtick.labelsize': 20, 'ytick.labelsize': 20, 'legend.fontsize': 16})
    if os.path.exists('sweeps_tau.csv'):
        plot_tau()
    if os.path.exists('sweeps_k.csv'):
        plot_k()
