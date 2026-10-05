# Contributing

Contributions are welcome through focused pull requests. Before submitting:

1. describe the behavior change and its evaluation scope;
2. keep datasets, checkpoints, generated videos, and credentials out of Git;
3. run `python scripts/check_public_release.py`;
4. run `python -m compileall -q foreact scripts deployment tools tests`;
5. run `python -m unittest discover -s tests -v` when dependencies are available.

Preserve third-party copyright and license notices. New code should follow the
repository license and document any additional third-party dependency or asset.
