## Cell 8: Visualize Results

```python
# Create comparison visualizations
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Prepare data
data = []
for key, metrics in results_summary.items():
    model, horizon = key.split('-')
    for metric, value in metrics.items():
        if metric in ['AUC', 'Precision', 'Recall']:  # Filter to standard metrics
            data.append({
                'Model': model,
                'Horizon': int(horizon),
                'Metric': metric,
                'Value': value
            })

df = pd.DataFrame(data)

# Create plots
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
metrics = ['AUC', 'Precision', 'Recall']

for i, metric in enumerate(metrics):
    ax = axes[i]
    metric_df = df[df['Metric'] == metric]
    
    for model in ['GNN', 'Kalman']:
        model_data = metric_df[metric_df['Model'] == model].sort_values('Horizon')
        if len(model_data) > 0:
            ax.plot(model_data['Horizon'], model_data['Value'], 
                    'o-', label=model, linewidth=2, markersize=8)
    
    ax.set_xlabel('Prediction Horizon (ticks)')
    ax.set_ylabel(metric)
    ax.set_title(f'{metric} vs Prediction Horizon')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1.05)

plt.suptitle('🌊 Aquascan Model Performance Comparison', fontsize=16)
plt.tight_layout()
plt.savefig('results/horizon_comparison.png', dpi=300, bbox_inches='tight')
plt.show()

# Print summary table
print("\\n📊 RESULTS SUMMARY")
print("="*60)
summary_df = df.pivot_table(index=['Model', 'Horizon'], columns='Metric', values='Value')
print(summary_df.round(4))

# Key findings
print("\\n🔍 KEY FINDINGS:")
for horizon in horizons:
    gnn_auc = df[(df['Model']=='GNN') & (df['Horizon']==horizon) & (df['Metric']=='AUC')]['Value'].values
    kalman_auc = df[(df['Model']=='Kalman') & (df['Horizon']==horizon) & (df['Metric']=='AUC')]['Value'].values
    if len(gnn_auc) > 0 and len(kalman_auc) > 0:
        improvement = ((gnn_auc[0] - kalman_auc[0]) / kalman_auc[0]) * 100
        print(f"  - {horizon}-tick: GNN improves AUC by {improvement:.1f}% over Kalman")
```