#!/usr/bin/env python3
import json
import os

state_file = 'data/indexing_state.json'

if os.path.exists(state_file):
    with open(state_file, 'r') as f:
        state = json.load(f)
    
    total_messages = len(state.get('messages', []))
    current_batch = state.get('current_batch', 0)
    percent = (current_batch / total_messages * 100) if total_messages > 0 else 0
    
    print(f"Indexing State:")
    print(f"  Status: {state.get('status')}")
    print(f"  Total Messages: {total_messages:,}")
    print(f"  Messages Processed: {current_batch:,}")
    print(f"  Progress: {percent:.1f}%")
    print(f"  Remaining: {total_messages - current_batch:,}")
    print(f"  Task ID: {state.get('task_id')}")
    
    # Estimate time remaining if processing 50 messages per batch
    batch_size = 50
    remaining_batches = (total_messages - current_batch) // batch_size
    print(f"  Remaining Batches: {remaining_batches}")
else:
    print("No indexing state file found")
