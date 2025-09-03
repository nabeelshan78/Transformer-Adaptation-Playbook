import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

def create_summary_plots():
    """
    Generates and saves summary visualizations for the Transformer fine-tuning experiments.
    """
    plots_dir = "plots"
    if not os.path.exists(plots_dir):
        os.makedirs(plots_dir)
    
    sns.set_theme(style="whitegrid")
    performance_data = {
        'Method': [
            'Train from Scratch', 
            'Linear Probing', 
            'PEFT with LoRA', 
            'PEFT with Adapters', 
            'Full Fine-Tuning'
        ],
        'Accuracy': [83.0, 64.0, 69.2, 85.6, 86.0]
    }
    df_perf = pd.DataFrame(performance_data)
    
    plt.figure(figsize=(12, 7))
    ax1 = sns.barplot(
        data=df_perf, 
        x='Method', 
        y='Accuracy', 
        palette='viridis',
        order=df_perf.sort_values('Accuracy', ascending=False).Method 
    )
    
    for p in ax1.patches:
        ax1.annotate(
            f'{p.get_height():.1f}%', 
            (p.get_x() + p.get_width() / 2., p.get_height()), 
            ha='center', va='center', 
            fontsize=12, 
            color='black', 
            xytext=(0, 8), 
            textcoords='offset points'
        )
        
    ax1.set_title('Fine-Tuning Strategy Performance on IMDB Sentiment Analysis', fontsize=16, pad=20)
    ax1.set_xlabel('Fine-Tuning Method', fontsize=12)
    ax1.set_ylabel('Test Accuracy (%)', fontsize=12)
    ax1.set_ylim(0, 100)
    plt.xticks(rotation=15, ha="right")
    plt.figtext(0.5, 0.01, '*Note: LoRA was applied to a simpler baseline model, not the full Transformer.', ha="center", fontsize=9, style='italic')
    
    performance_plot_path = os.path.join(plots_dir, 'performance_comparison_imdb.png')
    plt.savefig(performance_plot_path, dpi=300, bbox_inches='tight')
    print(f"Performance comparison plot saved to {performance_plot_path}")
    plt.show()

    # Data extracted from your LoRA notebook
    efficiency_data = {
        'Method': ['Full Layer Update', 'PEFT with LoRA'],
        'Trainable Parameters': [12800, 456]
    }
    df_eff = pd.DataFrame(efficiency_data)

    plt.figure(figsize=(10, 6))
    ax2 = sns.barplot(data=df_eff, x='Trainable Parameters', y='Method', palette='mako', orient='h')

    # Use a log scale to visualize the huge difference
    ax2.set_xscale('log')
    
    # Add annotations
    for p in ax2.patches:
        width = p.get_width()
        ax2.text(
            width * 1.2, 
            p.get_y() + p.get_height() / 2, 
            f'{int(width):,}', 
            va='center', 
            fontsize=12
        )

    ax2.set_title('Parameter Efficiency: Full Update vs. LoRA', fontsize=16, pad=20)
    ax2.set_xlabel('Trainable Parameters (Log Scale)', fontsize=12)
    ax2.set_ylabel('')
    
    efficiency_plot_path = os.path.join(plots_dir, 'parameter_efficiency_comparison.png')
    plt.savefig(efficiency_plot_path, dpi=300, bbox_inches='tight')
    print(f"Parameter efficiency plot saved to {efficiency_plot_path}")
    plt.show()

if __name__ == '__main__':
    create_summary_plots()