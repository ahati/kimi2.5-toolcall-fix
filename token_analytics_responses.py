#!/usr/bin/env python
import json
import glob
import os
def extract_data(filepath):
    with open(filepath, 'r') as f:
        data = json.load(f)
    
    filename = os.path.basename(filepath)
    # Extract timestamp from filename: 20260322-HHMMSS_xxxxxx.json
    timestamp_part = filename.split('_')[0].split('-')[1]  # HHMMSS
    timestamp = f'{timestamp_part[:2]}:{timestamp_part[2:4]}:{timestamp_part[4:6]}'
    
    # Extract model name from upstream_response.chunks[0].data.message.model
    upstream_chunks = data.get('upstream_response', {}).get('chunks', [])
    model_name = 'unknown'
    if upstream_chunks:
        first_chunk = upstream_chunks[0]
        if first_chunk.get('event') == 'message_start':
            model_name = first_chunk.get('data', {}).get('message', {}).get('model', 'unknown')
        else:
            # Find first message_start event
            for chunk in upstream_chunks:
                if chunk.get('event') == 'message_start':
                    model_name = chunk.get('data', {}).get('message', {}).get('model', 'unknown')
                    break
    
    # Extract upstream tokens from message_delta (fallback to message_start)
    upstream_tokens = {
        'input_tokens': 0,
        'output_tokens': 0,
        'cache_read_input_tokens': 0,
        'cache_creation_input_tokens': 0
    }
    
    # Try message_delta first (final usage)
    found_message_delta = False
    for chunk in reversed(upstream_chunks):
        if chunk.get('event') == 'message_delta':
            usage = chunk.get('data', {}).get('usage', {})
            if usage:
                upstream_tokens.update({
                    'input_tokens': usage.get('input_tokens', 0),
                    'output_tokens': usage.get('output_tokens', 0),
                    'cache_read_input_tokens': usage.get('cache_read_input_tokens', 0),
                    'cache_creation_input_tokens': usage.get('cache_creation_input_tokens', 0)
                })
                found_message_delta = True
                break
    
    # Fallback to message_start if no message_delta found
    if not found_message_delta:
        for chunk in upstream_chunks:
            if chunk.get('event') == 'message_start':
                usage = chunk.get('data', {}).get('message', {}).get('usage', {})
                if usage:
                    upstream_tokens.update({
                        'input_tokens': usage.get('input_tokens', 0),
                        'output_tokens': usage.get('output_tokens', 0),
                        'cache_read_input_tokens': 0,
                        'cache_creation_input_tokens': 0
                    })
                    break
    
    # Extract downstream tokens from response.completed
    downstream_tokens = {
        'input_tokens': 0,
        'output_tokens': 0,
        'total_tokens': 0,
        'cached_tokens': 0
    }
    
    downstream_chunks = data.get('downstream_response', {}).get('chunks', [])
    for chunk in downstream_chunks:
        data_chunk = chunk.get('data', {})
        if data_chunk.get('type') == 'response.completed':
            usage = data_chunk.get('response', {}).get('usage', {})
            input_details = usage.get('input_tokens_details', {})
            downstream_tokens.update({
                'input_tokens': usage.get('input_tokens', 0),
                'output_tokens': usage.get('output_tokens', 0),
                'total_tokens': usage.get('total_tokens', 0),
                'cached_tokens': input_details.get('cached_tokens', 0)
            })
            break
    
    return {
        'timestamp': timestamp,
        'model_name': model_name,
        'upstream_input_tokens': upstream_tokens['input_tokens'],
        'upstream_output_tokens': upstream_tokens['output_tokens'],
        'upstream_cache_read_input_tokens': upstream_tokens['cache_read_input_tokens'],
        'upstream_cache_creation_input_tokens': upstream_tokens['cache_creation_input_tokens'],
        'downstream_input_tokens': downstream_tokens['input_tokens'],
        'downstream_output_tokens': downstream_tokens['output_tokens'],
        'downstream_total_tokens': downstream_tokens['total_tokens'],
        'downstream_cached_tokens': downstream_tokens['cached_tokens']
    }
# Get all JSON files and sort chronologically
json_files = sorted(glob.glob('*.json'))
results = []
for filepath in json_files:
    if 'analyze_logs.py' in filepath:
        continue
    try:
        result = extract_data(filepath)
        results.append(result)
    except Exception as e:
        print(f'Error processing {filepath}: {e}')
# Print header
print('Comprehensive Token Usage Data - Chronological Order')
print('=' * 140)
header = f'{'Timestamp':<10} {'Model':<15} {'Up_In':<8} {'Up_Out':<8} {'Up_Cache_Read':<12} {'Up_Cache_Create':<14} {'Down_In':<10} {'Down_Out':<10} {'Down_Total':<10} {'Down_Cached':<10}'
print(header)
print('=' * 140)
# Print results
for result in results:
    row = f'{result["timestamp"]:<10} {result["model_name"]:<15} '
    row += f'{result["upstream_input_tokens"]:<8} {result["upstream_output_tokens"]:<8} '
    row += f'{result["upstream_cache_read_input_tokens"]:<12} {result["upstream_cache_creation_input_tokens"]:<14} '
    row += f'{result["downstream_input_tokens"]:<10} {result["downstream_output_tokens"]:<10} '
    row += f'{result["downstream_total_tokens"]:<10} {result["downstream_cached_tokens"]:<10}'
    print(row)
print('=' * 140)
print(f'Total records: {len(results)}')
