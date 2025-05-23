## Cell 7: Run Kalman Baselines

```python
# Run Kalman filter baseline for comparison
print("📐 Running Kalman filter baselines...\\n")

# The kalman script expects a single test.pt file path
for horizon in horizons:
    print(f"\\nEvaluating Kalman for {horizon}-tick horizon...")
    
    # Create a temporary script that accepts our arguments
    kalman_wrapper = f'''
import sys
sys.path.append('/content/aquascan-gnns')
import os
os.environ['PYTHONPATH'] = '/content/aquascan-gnns'

# Import the kalman evaluation function
from scripts.kalman_eval import evaluate_file
import json

# Run evaluation
results = evaluate_file('data/processed_{horizon}tick/test.pt')

# Save results
with open('results/kalman_{horizon}tick_results.json', 'w') as f:
    json.dump(results, f, indent=2)

print(f"Results: AUC={{results['AUC']:.4f}}")
'''
    
    # Write and run the wrapper
    with open(f'kalman_wrapper_{horizon}.py', 'w') as f:
        f.write(kalman_wrapper)
    
    !cd /content/aquascan-gnns && python kalman_wrapper_{horizon}.py
    
    # Load results
    with open(f'results/kalman_{horizon}tick_results.json', 'r') as f:
        results = json.load(f)
        results_summary[f'Kalman-{horizon}'] = results

print("\\n✅ All baselines complete!")
```