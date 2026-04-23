# TorchCode Local Setup Guide

## Prerequisites

- macOS (Apple Silicon)
- Homebrew
- Python 3.11+ (install via `brew install python@3.11` if needed)

## Step 1: Create a Virtual Environment

```bash
/opt/homebrew/bin/python3.11 -m venv .venv
```

## Step 2: Install Dependencies

```bash
.venv/bin/pip install -e .
.venv/bin/pip install jupyter
```

This installs:
- `torch-judge` (the project's judge engine, in editable mode)
- PyTorch >= 2.0
- Jupyter Notebook

## Step 3: Set Up Local Notebooks

```bash
make setup-local
```

This copies all template and solution notebooks into `./notebooks/`.

## Step 4: Launch Jupyter

```bash
source .venv/bin/activate
jupyter notebook notebooks/01_relu.ipynb
```

## Solving a Problem

1. Open a template notebook (e.g., `notebooks/01_relu.ipynb`).
2. Implement the function in the `YOUR IMPLEMENTATION HERE` cell.
3. Run the debug cell to sanity-check your output.
4. Run the `check()` cell to submit your solution to the judge.

```python
from torch_judge import check
check("relu")
```

## Alternative: Docker

If you prefer not to install Python locally:

```bash
make run
# Opens JupyterLab at http://localhost:8888
```

## Remote Configuration

```bash
# Set origin to your fork
git remote set-url origin https://github.com/yliu182/TorchCode.git
```


# To run command: 

jupyter notebook notebooks/01_relu.ipynb




# Git Cheatsheet

## 1. Switch Git Account for This Repo

```bash
git config user.name "Your Name"
git config user.email "your@email.com"
```

This only affects `/Users/yaoliu/Documents/lc`. Other repos still use the global config (`~/.gitconfig`).

## 2. Initial Checkin to Remote Repo

```bash
# Create repo on GitHub first (pick one):
gh repo create yliu182/lc --public --source=. --push
# OR create manually at https://github.com/new, then:

git add .
git commit -m "Initial commit"
git branch -M main
git remote add origin https://github.com/yliu182/lc.git
git push -u origin main
```

## 3. Future Modifications

```bash
# Check what changed
git status
git diff

# Stage and commit
git add <file>            # stage specific file
git add .                 # stage all changes

git commit -m "Your commit message"

# Push to GitHub
git push
```
