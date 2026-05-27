# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2026-05-27

Initial public release.

### Added
- Standalone `Dance` `nn.Module` with auto-derived `dense` target.
- Training pipeline and CLI (`dance run NAME --debug | --local | --submit`).
- Reference configs for the datasets used in the paper.
- Unit tests and GitHub Actions CI (lint + tests).
