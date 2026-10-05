## Summary

Describe the change and why it is needed.

## Verification

- [ ] `python scripts/check_public_release.py`
- [ ] `python -m compileall -q foreact scripts deployment tools tests`
- [ ] Relevant unit tests
- [ ] Training or benchmark smoke test, when applicable

For behavior or benchmark changes, provide the exact config, checkpoint digest,
command, seed, metric, and result.

## Checklist

- [ ] The change is focused and documented.
- [ ] Public configs use portable paths.
- [ ] No credentials, datasets, model weights, generated videos, or private paths are included.
- [ ] Third-party copyright and license notices are preserved.
- [ ] New behavior includes a reproducible test or evaluation note.
