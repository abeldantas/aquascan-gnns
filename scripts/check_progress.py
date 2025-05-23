#!/usr/bin/env python3
"""
Quick estimation of remaining time for the current graph building process.
"""

import subprocess
import re
from datetime import datetime, timedelta

# Get the current process info
result = subprocess.run(['ps', 'aux'], capture_output=True, text=True)
lines = result.stdout.strip().split('\n')

for line in lines:
    if 'build_graphs' in line and 'processed_30tick' in line:
        # Extract start time (12:06PM format)
        match = re.search(r'(\d+):(\d+)(AM|PM)', line)
        if match:
            hour = int(match.group(1))
            minute = int(match.group(2))
            period = match.group(3)
            
            if period == 'PM' and hour != 12:
                hour += 12
            elif period == 'AM' and hour == 12:
                hour = 0
            
            start_time = datetime.now().replace(hour=hour, minute=minute, second=0)
            elapsed = datetime.now() - start_time
            
            print(f"🕐 Current process started at: {start_time.strftime('%I:%M %p')}")
            print(f"⏱️  Elapsed time: {elapsed}")
            print(f"📊 Still on FIRST horizon (30-tick) after {elapsed.total_seconds()/60:.0f} minutes!")
            print()
            print("⚠️  At this rate:")
            print(f"   - 30-tick horizon: ~2-3 hours")
            print(f"   - 100-tick horizon: ~3-4 hours") 
            print(f"   - 150-tick horizon: ~4-5 hours")
            print(f"   - TOTAL: ~10-12 hours! 😱")
            print()
            print("💡 Recommendation: Kill it and use the parallel version!")
            print("   1. Ctrl+C to stop the current script")
            print("   2. Run: ./scripts/build_all_graphs_parallel.sh")
            print("   3. Should finish ALL horizons in ~15-30 minutes with parallel processing")
            break
