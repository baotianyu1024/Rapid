# Framework source snapshots

This directory contains the source trees used for the revision comparisons:
HugeCTR, TorchRec, HET-GMP, Herald and BagPipe. Their upstream URLs and pinned
commits are in `../../experiments/dependencies.json`. RAPID's own CPS/ACP
implementation remains in the repository's `cps/` and `acp/` directories;
EMBark is selected in the HugeCTR experiment driver.

The snapshots exclude downloaded datasets, logs, notebooks and documentation
assets, build directories, bytecode, binaries and environment-specific files.
They retain the source and build definitions. Keep each upstream license and
copyright notice when redistributing or modifying these trees. The exact
experiment commands and input layout are in `../README.md`.
