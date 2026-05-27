# Contributing to DANCE

We want to make contributing to this project as easy and transparent as
possible.

## Our Development Process

DANCE is developed in the open: changes are reviewed and merged here on
GitHub. We use the public `main` branch as the source of truth; internal
Meta forks are kept in sync via ShipIt.

## Pull Requests

We actively welcome your pull requests.

1. Fork the repo and create your branch from `main`.
2. Install with the dev extras: `pip install -e ".[dev]"`. This adds `pytest`, `ruff` and `pre-commit`.
3. Run `pre-commit install` so lint and format run on every commit.
4. If you've added code that should be tested, add tests under [`dance/tests/`](dance/tests/).
5. If you've changed APIs, update the docstrings and the README.
6. Ensure the test suite passes: `pytest dance/tests/`.
7. Make sure your code lints: `ruff check dance/` and `ruff format --check dance/`.
8. If you haven't already, complete the Contributor License Agreement ("CLA").

## Contributor License Agreement ("CLA")

In order to accept your pull request, we need you to submit a CLA. You only need
to do this once to work on any of Meta's open source projects.

Complete your CLA here: <https://code.facebook.com/cla>

## Issues

We use GitHub issues to track public bugs. Please ensure your description is
clear and has sufficient instructions to be able to reproduce the issue (the
dataset slug, the exact CLI invocation, the traceback or unexpected behaviour).

Meta has a [bounty program](https://bugbounty.meta.com/) for the safe
disclosure of security bugs. In those cases, please go through the process
outlined on that page and do not file a public issue.

## Coding Style

Code style is enforced by `ruff`. Run `ruff check dance/` and
`ruff format dance/` before opening a PR; the pre-commit hook does this
automatically.

## License

By contributing to DANCE, you agree that your contributions will be licensed
under the LICENSE file in the root directory of this source tree.
