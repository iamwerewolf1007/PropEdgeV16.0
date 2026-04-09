# PropEdge V16.0 — GitHub Repo

## This folder: `~/Documents/GitHub/PropEdgeV16.0`
GitHub repository — serves the PropEdge dashboard via GitHub Pages.

## What's in this repo
- `index.html` — the PropEdge dashboard (served by GitHub Pages)
- `*.py` — all Python scripts (version controlled here)
- `data/*.json` — dashboard data (pushed here by PropEdge REST API from Local)
- `models/*/player_trust.json` — trust score tables

## Important: Do NOT manually commit data files via GitHub Desktop
The JSON data files are managed automatically by PropEdge's `git_push.py`:
- `data/today.json` — pushed after every predict batch (B1-B5)
- `data/season_2025_26.json` — pushed after every B0 grade
- `data/season_2024_25.json` — pushed after generate
- `data/dvp_rankings.json` — pushed daily

## For code changes (Python + dashboard)
Edit files in `PropEdgeV16.0-Local/` → GitHub Desktop picks up changes here → commit + push normally.

## GitHub Pages setup
Settings → Pages → Source: main branch, root folder
Dashboard URL: https://iamwerewolf1007.github.io/PropEdgeV16.0/
