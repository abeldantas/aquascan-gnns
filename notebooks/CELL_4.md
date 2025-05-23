## Cell 4: Generate Raw Data

```python
# Generate raw simulation data
start_time = time.time()

print(f"\\n🚀 Starting simulation generation at {datetime.now().strftime('%H:%M:%S')}...")

# Run the batch generator
!python -m aquascan.batch.generate \
    --cfg configs/optimal_5tick.yml \
    --runs {NUM_RUNS} \
    --out data/raw_5tick \
    --jobs {CPUS}

elapsed = time.time() - start_time
print(f"\\n✅ Raw data generation complete in {elapsed/60:.1f} minutes!")

# Check results
!echo "\\n📊 Generated files:"
!ls data/raw_5tick | wc -l
!echo "\\n💾 Total size:"
!du -sh data/raw_5tick
```