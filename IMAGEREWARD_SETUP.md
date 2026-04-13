# ImageReward local setup guide

To use ImageReward from a local editable repository instead of pip:

## One-time setup

```bash
chmod +x setup_imagereward.sh
./setup_imagereward.sh
```

This will:
1. Clone ImageReward from https://github.com/THUDM/ImageReward.git into `./third_party/ImageReward/`
2. Install it in editable mode (`pip install -e`)

## After setup

Your imports in `daas/scorers/ImageReward_scorer.py` will use the local cloned version.

You can now:
- Patch compatibility issues directly in `third_party/ImageReward/`
- Test changes without reinstalling
- Pin specific commits for reproducibility

## To use a specific commit

```bash
cd third_party/ImageReward
git checkout <commit-hash>
cd ../..
```

## To sync with upstream

```bash
cd third_party/ImageReward
git pull origin main
cd ../..
```

The editable install means changes to source files are reflected immediately in imports.
