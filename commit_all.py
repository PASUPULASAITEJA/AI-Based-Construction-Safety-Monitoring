import subprocess
import os

# Get all changed files
result = subprocess.run(['git', 'status', '-s'], capture_output=True, text=True)
lines = result.stdout.strip().split('\n')

for line in lines:
    if not line:
        continue
    
    # Git status -s output format: ' M path/to/file' or 'D  path/to/file'
    parts = line.strip().split(' ', 1)
    status = parts[0]
    filepath = parts[-1].strip()
    
    # Strip quotes if any
    if filepath.startswith('"'):
        filepath = filepath.strip('"')
    
    # Get just the filename
    filename = filepath.split('/')[-1]
    
    # Add and commit
    subprocess.run(['git', 'add', filepath])
    
    # Handle deleted files
    if status == 'D' or status == 'D ':
        subprocess.run(['git', 'commit', '-m', f'remove {filename}'])
    else:
        subprocess.run(['git', 'commit', '-m', f'update {filename}'])
