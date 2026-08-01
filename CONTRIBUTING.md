# Contributing to Tear0

Thanks for considering a contribution.

## Development setup

1. Fork and clone the repo.
2. Run `install.ps1` on Windows, or use `uv sync --extra test --extra build` for development.
3. Run tests with `uv run pytest -q` or `.venv/Scripts/python.exe -m pytest -q` on Windows.

## Contribution guidelines

- Keep local paths, usernames, model files, audio captures, screenshots, secrets, and generated config out of commits.
- Do not commit `tear0.config.json`; commit `tear0.config.example.json` only.
- Add or update tests for behavior changes.
- Keep Windows-first behavior working; avoid hard-coded machine assumptions.
- Run the public-readiness check before opening a PR.
