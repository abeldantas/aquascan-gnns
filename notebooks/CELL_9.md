## Cell 9: Save to Google Drive

```python
# Mount Google Drive for backup
from google.colab import drive
drive.mount('/content/drive')

# Create timestamp
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

# Create archives
print("📦 Creating archives...")
!cd /content/aquascan-gnns && tar -czf aquascan_results_{timestamp}.tar.gz results/ checkpoints/
!cd /content/aquascan-gnns && tar -czf aquascan_graphs_{timestamp}.tar.gz data/processed_*

# Copy to Drive
!mkdir -p /content/drive/MyDrive/aquascan_backups
!cp /content/aquascan-gnns/aquascan_results_{timestamp}.tar.gz /content/drive/MyDrive/aquascan_backups/
!cp /content/aquascan-gnns/aquascan_graphs_{timestamp}.tar.gz /content/drive/MyDrive/aquascan_backups/

print(f"\\n✅ Results backed up to Google Drive!")
print(f"📁 /content/drive/MyDrive/aquascan_backups/aquascan_results_{timestamp}.tar.gz")
print(f"📁 /content/drive/MyDrive/aquascan_backups/aquascan_graphs_{timestamp}.tar.gz")
```