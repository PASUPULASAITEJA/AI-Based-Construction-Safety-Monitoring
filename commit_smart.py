import subprocess
import os

def run_cmd(args):
    subprocess.run(args, check=False)

# Group dataset deletions
run_cmd(['git', 'add', 'construction/'])
run_cmd(['git', 'commit', '-m', 'remove redundant training datasets'])

# Group unused script deletions
scripts = ['evaluate.py', 'train.py', 'verify_full_system.py', 'test_all_user_images.py', 'tests/']
for s in scripts:
    run_cmd(['git', 'add', s])
run_cmd(['git', 'commit', '-m', 'remove unused test and training scripts'])

# Get remaining files
result = subprocess.run(['git', 'status', '-s'], capture_output=True, text=True)
lines = result.stdout.strip().split('\n')

for line in lines:
    if not line:
        continue
    
    parts = line.strip().split(' ', 1)
    status = parts[0]
    filepath = parts[-1].strip()
    
    if filepath.startswith('"'):
        filepath = filepath.strip('"')
    
    filename = filepath.split('/')[-1]
    
    run_cmd(['git', 'add', filepath])
    
    if 'D' in status:
        run_cmd(['git', 'commit', '-m', f'remove {filename}'])
    elif 'A' in status or '?' in status:
        run_cmd(['git', 'commit', '-m', f'create {filename}'])
    else:
        run_cmd(['git', 'commit', '-m', f'update {filename}'])

