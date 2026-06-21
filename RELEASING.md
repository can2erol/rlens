# Releasing

rlens publishes to [PyPI](https://pypi.org/project/rlens/) via **Trusted Publishing**
(OIDC) — no API token is stored anywhere. A published GitHub Release triggers
[`.github/workflows/publish.yml`](.github/workflows/publish.yml), which builds and uploads.

## One-time setup (before the first release)

On PyPI, add a *pending publisher* so the very first upload is authorized:

1. Log in to https://pypi.org → **Account settings → Publishing → Add a pending publisher**.
2. Fill in:
   - **PyPI project name:** `rlens`
   - **Owner:** `can2erol`
   - **Repository name:** `rlens`
   - **Workflow name:** `publish.yml`
   - **Environment name:** `pypi`
3. In the GitHub repo, create an Environment named `pypi`
   (Settings → Environments → New environment). No secrets are needed.

(Recommended: do a dry run against [TestPyPI](https://test.pypi.org) first by repeating the
pending-publisher step there and temporarily pointing the publish step at TestPyPI.)

## Cutting a release

1. Bump the version in **one** place — `rlens/__init__.py` (`__version__`); `pyproject.toml`
   reads it dynamically.
2. Sanity check locally:
   ```bash
   ruff check rlens tests
   pytest -q
   python -m build && twine check dist/*
   ```
3. Commit, tag, and push:
   ```bash
   git commit -am "release: vX.Y.Z"
   git tag vX.Y.Z && git push --tags
   ```
4. Create a GitHub Release for the tag. Publishing it runs the workflow, which uploads to
   PyPI.

## Manual fallback

If you'd rather upload by hand (with a PyPI API token in `~/.pypirc`):

```bash
python -m build
twine check dist/*
twine upload dist/*
```

## Versioning

Single source of truth: `__version__` in `rlens/__init__.py`. Follow semantic versioning;
the project is `0.x` (alpha) until the API stabilizes.
