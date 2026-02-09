
import subprocess
import os

def run(cmd):
    try:
        return subprocess.check_output(cmd, shell=True, text=True, stderr=subprocess.STDOUT).strip()
    except subprocess.CalledProcessError as e:
        return f"Error: {e.output}"

print(f"Total commits in HEAD: {run('git rev-list --count HEAD')}")

branches = run("git branch -r").splitlines()
print(f"Found {len(branches)} remote branches")

for branch in branches:
    branch = branch.strip()
    if "HEAD" in branch: continue
    # Get commit count ahead of master
    count = run(f"git rev-list --count master..{branch}")
    print(f"Ahead of master - {branch}: {count}")
    
    # Get commit count behind master
    behind = run(f"git rev-list --count {branch}..master")
    print(f"Behind master - {branch}: {behind}")
