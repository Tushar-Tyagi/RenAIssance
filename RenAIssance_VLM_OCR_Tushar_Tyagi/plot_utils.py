"""
Plotting utilities for model evaluation analysis.
"""
import os
import json
import re
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def load_base_model_results(outputs_dir='outputs'):
    """
    Load evaluation results for base models (excluding LLM correction files).
    
    Args:
        outputs_dir: Directory containing evaluation JSON files
        
    Returns:
        pd.DataFrame: DataFrame with columns ['Model', 'Dataset', 'CER', 'WER']
    """
    results = []
    
    if os.path.exists(outputs_dir):
        for filename in os.listdir(outputs_dir):
            # Skip LLM correction files to only include base model results
            if filename.endswith('.json') and 'llm-correction' not in filename.lower():
                filepath = os.path.join(outputs_dir, filename)
                with open(filepath, 'r') as f:
                    try:
                        data = json.load(f)
                        cer = data.get('cer', None)
                        wer = data.get('wer', None)
                        # Only append rows with valid CER and WER values
                        if cer is not None and wer is not None:
                            results.append({
                                'Model': data.get('model_id', 'Unknown'),
                                'Dataset': data.get('data_dir', 'Unknown'),
                                'CER': cer,
                                'WER': wer
                            })
                    except json.JSONDecodeError:
                        print(f"Warning: Could not parse {filename}")
    
    df = pd.DataFrame(results)
    
    # Filter to only include data_alltest and sort
    if not df.empty:
        df = df[df['Dataset'] == 'data_alltest']
        df = df.sort_values(by=['CER']).reset_index(drop=True)
    
    return df


def plot_cer_wer_comparison(df, save_path=None):
    """
    Plot CER and WER bar charts for base models.
    
    Args:
        df: DataFrame with columns ['Model', 'CER', 'WER']
        save_path: Optional path to save the figure
    """
    if df.empty:
        print("No evaluation results found for 'data_alltest' in the 'outputs' directory.")
        return
    
    sns.set_theme(style="whitegrid", palette="deep")
    
    def plot_metric(metric, title):
        plt.figure(figsize=(14, 7))
        ax = sns.barplot(data=df, x='Model', y=metric, color=sns.color_palette("deep")[0])
        plt.title(title, fontsize=16, pad=15)
        plt.xticks(rotation=45, ha='right', fontsize=11)
        plt.yticks(fontsize=11)
        plt.ylabel(metric, fontsize=12)
        plt.xlabel('Model', fontsize=12)
        
        # Add value labels on bars
        for p in ax.patches:
            height = p.get_height()
            if not pd.isna(height) and height > 0:
                ax.annotate(format(height, '.3f'), 
                            (p.get_x() + p.get_width() / 2., height), 
                            ha='center', va='center', 
                            xytext=(0, 9), 
                            textcoords='offset points',
                            fontsize=9)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
        plt.show()
    
    plot_metric('CER', 'Character Error Rate (CER) by Model\n(Lower is Better)')
    plot_metric('WER', 'Word Error Rate (WER) by Model\n(Lower is Better)')


def extract_model_size(model_name):
    """
    Extract model size (in billions) from model name.
    
    Args:
        model_name: Model name string
        
    Returns:
        float or None: Model size in billions
    """
    match = re.search(r'(\d+(?:\.\d+)?)(?:B)', model_name)
    if match:
        return float(match.group(1))
    return None


def plot_model_size_vs_accuracy(df, save_path=None):
    """
    Plot relationship between model size and accuracy (1-CER).
    
    Args:
        df: DataFrame with columns ['Model', 'CER']
        save_path: Optional path to save the figure
    """
    if df.empty:
        print("No data available for size vs accuracy plot.")
        return
    
    size_df = df.copy()
    size_df['Size (B)'] = size_df['Model'].apply(extract_model_size)
    size_df['Accuracy (1-CER)'] = 1 - size_df['CER']
    size_df = size_df.dropna(subset=['Size (B)'])
    
    if size_df.empty:
        print("Could not extract model sizes from the model names.")
        return
    
    plt.figure(figsize=(14, 8))
    
    ax = sns.scatterplot(data=size_df, x='Size (B)', y='Accuracy (1-CER)', 
                        hue='Model', style='Model', s=250, alpha=0.9, palette='tab10')
    
    plt.title('Model Size vs Accuracy (1 - CER)', fontsize=16, pad=15)
    plt.xlabel('Model Size (Billions of Parameters)', fontsize=12)
    plt.ylabel('Accuracy (1 - CER)', fontsize=12)
    
    plt.grid(True, linestyle='--', alpha=0.6)
    
    # Move the legend outside the plot
    sns.move_legend(ax, "upper left", bbox_to_anchor=(1.02, 1), title='Model', frameon=True)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    plt.show()


def load_finetuned_comparison_data(outputs_dir='outputs', finetuned_dir='finetuned_outputs'):
    """
    Load data for comparing base vs fine-tuned models.
    
    Args:
        outputs_dir: Directory containing base model evaluation files
        finetuned_dir: Directory containing fine-tuned model evaluation files
        
    Returns:
        pd.DataFrame: DataFrame with columns ['Model', 'Type', 'CER', 'WER']
    """
    base_results = []
    finetuned_results = []
    
    # Read finetuned results first to know which models to compare
    if os.path.exists(finetuned_dir):
        for filename in os.listdir(finetuned_dir):
            if filename.endswith('.json'):
                filepath = os.path.join(finetuned_dir, filename)
                with open(filepath, 'r') as f:
                    try:
                        data = json.load(f)
                        cer = data.get('cer', None)
                        wer = data.get('wer', None)
                        if cer is not None and wer is not None and data.get('data_dir') == 'data_alltest':
                            finetuned_results.append({
                                'Model': data.get('model_id', 'Unknown'),
                                'Type': 'Fine-Tuned',
                                'CER': cer,
                                'WER': wer
                            })
                    except json.JSONDecodeError:
                        print(f"Warning: Could not parse {filename}")
    
    finetuned_models = [res['Model'] for res in finetuned_results]
    
    # Read base results for the models that have finetuned counterparts
    if os.path.exists(outputs_dir) and finetuned_models:
        for filename in os.listdir(outputs_dir):
            if filename.endswith('.json'):
                filepath = os.path.join(outputs_dir, filename)
                with open(filepath, 'r') as f:
                    try:
                        data = json.load(f)
                        model_id = data.get('model_id', 'Unknown')
                        cer = data.get('cer', None)
                        wer = data.get('wer', None)
                        if model_id in finetuned_models and cer is not None and wer is not None and data.get('data_dir') == 'data_alltest':
                            base_results.append({
                                'Model': model_id,
                                'Type': 'Base',
                                'CER': cer,
                                'WER': wer
                            })
                    except json.JSONDecodeError:
                        print(f"Warning: Could not parse {filename}")
    
    all_results = base_results + finetuned_results
    df_compare = pd.DataFrame(all_results)
    
    if not df_compare.empty:
        df_compare = df_compare.sort_values(by=['Model', 'Type']).reset_index(drop=True)
    
    return df_compare


def plot_base_vs_finetuned(df, save_path=None):
    """
    Plot CER and WER comparison between base and fine-tuned models.
    
    Args:
        df: DataFrame with columns ['Model', 'Type', 'CER', 'WER']
        save_path: Optional path to save the figure
    """
    if df.empty:
        print("No matching base and fine-tuned results found for comparison.")
        return
    
    sns.set_theme(style="whitegrid", palette="deep")
    
    def plot_comparison(metric, title):
        plt.figure(figsize=(12, 6))
        ax = sns.barplot(data=df, x='Model', y=metric, hue='Type')
        plt.title(title, fontsize=16, pad=15)
        plt.xticks(rotation=15, ha='right', fontsize=11)
        plt.yticks(fontsize=11)
        plt.ylabel(metric, fontsize=12)
        plt.xlabel('Model', fontsize=12)
        plt.legend(title='Model Type', title_fontsize='13', fontsize='11')
        
        # Add value labels on bars
        for p in ax.patches:
            height = p.get_height()
            if not pd.isna(height) and height > 0:
                ax.annotate(format(height, '.3f'), 
                            (p.get_x() + p.get_width() / 2., height), 
                            ha='center', va='center', 
                            xytext=(0, 9), 
                            textcoords='offset points',
                            fontsize=9)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
        plt.show()
    
    plot_comparison('CER', 'Base vs Fine-Tuned: Character Error Rate (CER)\n(Lower is Better)')
    plot_comparison('WER', 'Base vs Fine-Tuned: Word Error Rate (WER)\n(Lower is Better)')


def load_llm_correction_comparison_data(outputs_dir='outputs'):
    """
    Load data for comparing base models with LLM-corrected versions.
    
    Args:
        outputs_dir: Directory containing evaluation JSON files
        
    Returns:
        pd.DataFrame: DataFrame with columns ['Model', 'Type', 'CER', 'WER']
    """
    base_results = []
    llm_corrected_results = []
    
    if os.path.exists(outputs_dir):
        for filename in os.listdir(outputs_dir):
            if filename.endswith('.json'):
                filepath = os.path.join(outputs_dir, filename)
                with open(filepath, 'r') as f:
                    try:
                        data = json.load(f)
                        cer = data.get('cer', None)
                        wer = data.get('wer', None)
                        model_id = data.get('model_id', 'Unknown')
                        
                        if cer is not None and wer is not None and data.get('data_dir') == 'data_alltest':
                            if 'llm-correction' in filename.lower():
                                # Extract base model name from LLM correction filename
                                base_model = model_id.replace('_llm-correction', '')
                                llm_corrected_results.append({
                                    'Model': base_model,
                                    'Type': 'LLM-Corrected',
                                    'CER': cer,
                                    'WER': wer
                                })
                            else:
                                base_results.append({
                                    'Model': model_id,
                                    'Type': 'Base',
                                    'CER': cer,
                                    'WER': wer
                                })
                    except json.JSONDecodeError:
                        print(f"Warning: Could not parse {filename}")
    
    llm_corrected_models = [res['Model'] for res in llm_corrected_results]
    
    # Filter base results to only include models that have LLM-corrected counterparts
    base_with_correction = [res for res in base_results if res['Model'] in llm_corrected_models]
    
    all_results = base_with_correction + llm_corrected_results
    df_llm_compare = pd.DataFrame(all_results)
    
    if not df_llm_compare.empty:
        df_llm_compare = df_llm_compare.sort_values(by=['Model', 'Type']).reset_index(drop=True)
    
    return df_llm_compare


def plot_llm_correction_comparison(df, save_path=None):
    """
    Plot CER and WER comparison between base and LLM-corrected models.
    
    Args:
        df: DataFrame with columns ['Model', 'Type', 'CER', 'WER']
        save_path: Optional path to save the figure
    """
    if df.empty:
        print("No matching base and LLM-corrected results found for comparison.")
        return
    
    sns.set_theme(style="whitegrid", palette="deep")
    
    def plot_llm_comparison(metric, title):
        plt.figure(figsize=(12, 6))
        ax = sns.barplot(data=df, x='Model', y=metric, hue='Type')
        plt.title(title, fontsize=16, pad=15)
        plt.xticks(rotation=15, ha='right', fontsize=11)
        plt.yticks(fontsize=11)
        plt.ylabel(metric, fontsize=12)
        plt.xlabel('Model', fontsize=12)
        plt.legend(title='Model Type', title_fontsize='13', fontsize='11')
        
        # Add value labels on bars
        for p in ax.patches:
            height = p.get_height()
            if not pd.isna(height) and height > 0:
                ax.annotate(format(height, '.3f'), 
                            (p.get_x() + p.get_width() / 2., height), 
                            ha='center', va='center', 
                            xytext=(0, 9), 
                            textcoords='offset points',
                            fontsize=9)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
        plt.show()
    
    plot_llm_comparison('CER', 'Base vs LLM-Corrected: Character Error Rate (CER)\n(Lower is Better)')
    plot_llm_comparison('WER', 'Base vs LLM-Corrected: Word Error Rate (WER)\n(Lower is Better)')


def run_all_plots(outputs_dir='outputs', finetuned_dir='finetuned_outputs'):
    """
    Run all plotting functions to generate all evaluation plots.
    
    Args:
        outputs_dir: Directory containing base model evaluation files
        finetuned_dir: Directory containing fine-tuned model evaluation files
    """
    print("=" * 60)
    print("Model Evaluation Analysis")
    print("=" * 60)
    
    # 1. Compare Accuracies (Base Models)
    print("\n1. Comparing CER/WER across base models...")
    df_base = load_base_model_results(outputs_dir)
    plot_cer_wer_comparison(df_base)
    
    # 2. Accuracy vs Model Size
    print("\n2. Plotting model size vs accuracy...")
    plot_model_size_vs_accuracy(df_base)
    
    # 3. Base vs Fine-Tuned Models
    print("\n3. Comparing base vs fine-tuned models...")
    df_finetuned = load_finetuned_comparison_data(outputs_dir, finetuned_dir)
    plot_base_vs_finetuned(df_finetuned)
    
    # 4. Base vs LLM-Corrected Models
    print("\n4. Comparing base vs LLM-corrected models...")
    df_llm = load_llm_correction_comparison_data(outputs_dir)
    plot_llm_correction_comparison(df_llm)
    
    print("\n" + "=" * 60)
    print("Analysis complete!")
    print("=" * 60)
