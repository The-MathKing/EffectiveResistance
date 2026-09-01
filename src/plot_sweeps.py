import pandas as pd
import matplotlib.pyplot as plt
import os

def plot_tau():
    df = pd.read_csv('sweeps_tau.csv')
    fig, ax1 = plt.subplots(figsize=(8, 6))

    color = 'tab:blue'
    ax1.set_xlabel(r'Rewiring Budget ($\tau$)', fontsize=14)
    ax1.set_ylabel(r'Spectral Gap ($\lambda_2$)', color=color, fontsize=14)
    ax1.plot(df['budget_tau'], df['spectral_gap'], marker='o', color=color, linestyle='-', linewidth=2)
    ax1.tick_params(axis='y', labelcolor=color)

    ax2 = ax1.twinx()  
    color = 'tab:red'
    ax2.set_ylabel('GCN Test Accuracy (%)', color=color, fontsize=14)
    ax2.plot(df['budget_tau'], df['accuracy'], marker='s', color=color, linestyle='--', linewidth=2)
    ax2.tick_params(axis='y', labelcolor=color)

    fig.tight_layout()
    plt.title(r'Effect of Rewiring Budget ($\tau$) on Spectral Gap and Accuracy', fontsize=16)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.savefig('sweeps_tau.pdf', format='pdf', dpi=300)
    print("Saved sweeps_tau.pdf")

def plot_k():
    df = pd.read_csv('sweeps_k.csv')
    fig, ax1 = plt.subplots(figsize=(8, 6))

    color = 'tab:blue'
    ax1.set_xlabel(r'$\kappa$-Core Threshold ($\kappa$)', fontsize=14)
    ax1.set_ylabel(r'Spectral Gap ($\lambda_2$)', color=color, fontsize=14)
    ax1.plot(df['k_core'], df['spectral_gap'], marker='o', color=color, linestyle='-', linewidth=2)
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.set_xticks(df['k_core'])

    ax2 = ax1.twinx()  
    color = 'tab:red'
    ax2.set_ylabel('GCN Test Accuracy (%)', color=color, fontsize=14)
    ax2.plot(df['k_core'], df['accuracy'], marker='s', color=color, linestyle='--', linewidth=2)
    ax2.tick_params(axis='y', labelcolor=color)

    fig.tight_layout()
    plt.title(r'Effect of $\kappa$-Core Filter on Spectral Gap and Accuracy', fontsize=16)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.savefig('sweeps_k.pdf', format='pdf', dpi=300)
    print("Saved sweeps_k.pdf")

if __name__ == '__main__':
    if os.path.exists('sweeps_tau.csv'):
        plot_tau()
    if os.path.exists('sweeps_k.csv'):
        plot_k()
