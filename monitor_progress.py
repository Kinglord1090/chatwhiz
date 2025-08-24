#!/usr/bin/env python3
import json
import os
import time
import sys

state_file = 'data/indexing_state.json'

print("Monitoring indexing progress... (Press Ctrl+C to stop)")
print("-" * 60)

last_batch = 0
last_time = time.time()

try:
    while True:
        if os.path.exists(state_file):
            try:
                with open(state_file, 'r') as f:
                    state = json.load(f)
                
                total_messages = len(state.get('messages', []))
                current_batch = state.get('current_batch', 0)
                percent = (current_batch / total_messages * 100) if total_messages > 0 else 0
                
                # Calculate processing speed
                time_diff = time.time() - last_time
                batch_diff = current_batch - last_batch
                if time_diff > 0:
                    speed = batch_diff / time_diff
                else:
                    speed = 0
                
                # Update for next iteration
                if current_batch != last_batch:
                    last_batch = current_batch
                    last_time = time.time()
                
                # Clear line and print progress
                sys.stdout.write('\r')
                sys.stdout.write(f"Progress: {current_batch:,}/{total_messages:,} ({percent:.1f}%) | Speed: {speed:.1f} msg/s | Remaining: {total_messages - current_batch:,}")
                sys.stdout.flush()
                
                if state.get('status') != 'indexing':
                    print(f"\nStatus changed to: {state.get('status')}")
                    break
                    
            except (json.JSONDecodeError, KeyError):
                pass
        else:
            sys.stdout.write('\r')
            sys.stdout.write("Waiting for indexing to start...")
            sys.stdout.flush()
        
        time.sleep(2)  # Check every 2 seconds
        
except KeyboardInterrupt:
    print("\n\nMonitoring stopped.")
