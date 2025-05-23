## Cell 10: Push Results to GitHub

```python
# Commit and push results to GitHub
print("🚀 Pushing results to GitHub...\\n")

# Go to repo directory
%cd /content/aquascan-gnns

# Check what we're committing
!git status

# Add results and visualizations
!git add results/*.json
!git add results/*.png
!git add -f checkpoints/*.pt 2>/dev/null || echo "Checkpoints might be too large for git"

# Create detailed commit message
if len(results_summary) > 0:
    # Find best GNN result
    gnn_results = [(h, results_summary[f'GNN-{h}']['AUC']) for h in horizons if f'GNN-{h}' in results_summary]
    if gnn_results:
        best_gnn = max(gnn_results, key=lambda x: x[1])
        commit_msg = f"🤖 Colab run: GNN AUC={best_gnn[1]:.4f} @ {best_gnn[0]}t horizon"
        
        # Get performance drop info
        if len(gnn_results) >= 2:
            perf_30 = next((r[1] for r in gnn_results if r[0] == 30), 0)
            perf_150 = next((r[1] for r in gnn_results if r[0] == 150), 0)
            if perf_30 > 0:
                drop = (perf_30 - perf_150) / perf_30 * 100
                details = f"Performance drop from 30t to 150t: {drop:.1f}%"
            else:
                details = "Complete results in results/"
        else:
            details = "Partial run - see results/"
    else:
        commit_msg = "🤖 Colab run completed"
        details = "Results in results/"
else:
    commit_msg = "🤖 Colab pipeline test"
    details = "Testing pipeline"

# Commit
!git commit -m "{commit_msg}" -m "{details}" -m "Run on Colab Pro with {NUM_RUNS} simulations" || echo "Nothing to commit"

# Push
!git push origin main || echo "Push failed - check your token permissions"

print("\\n✅ Results pushed to repository!")
```