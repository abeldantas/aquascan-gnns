## Cell 3: Configuration

```python
import multiprocessing
import time
from datetime import datetime

# Configuration
NUM_RUNS = 100  # Change to 1000 for full dataset
CPUS = multiprocessing.cpu_count()

print(f"🎯 Configuration:")
print(f"   - Simulation runs: {NUM_RUNS}")
print(f"   - CPU cores: {CPUS}")
print(f"   - Estimated time: {NUM_RUNS * 0.1:.0f} minutes")
print(f"   - Estimated size: {NUM_RUNS * 2:.0f} MB")
```