# Publication Checklist

Before making this repository public:

- [ ] `git status --short` reviewed; only intentional source/doc/test changes are present.
- [ ] `git ls-files tear0.config.json smoke .env` returns no tracked files.
- [ ] `python scripts/public_readiness_check.py` passes.
- [ ] `pytest -q` passes.
- [ ] README install path examples are generic.
- [ ] No local username, absolute local path, API key, token, password, or model artifact is tracked.
- [ ] `tear0.config.example.json` is present and `tear0.config.json` is ignored.
- [ ] Optional GPU/CUDA behavior is documented as optional and auto-detected.
