# Third-party sources

The native Herald harness contains the upstream LAIA scheduling routine with the set-container substitution documented in experiments/README.md. Its license is retained in licenses/Herald-LICENSE.

The official T-series generator scripts preserved in the historical archive originate from NVIDIA Merlin HugeCTR. Its license is retained in licenses/HugeCTR-LICENSE. The portable generator entrypoint executes a separately fetched, pinned upstream checkout.

GMP, BagPipe and TorchRec are fetched separately and retain their upstream source and licensing terms. The dependency URLs and exact revisions are recorded in experiments/dependencies.json. The GMP patch records only the lookup-allocation safety correction used in the final comparison; the algorithm is not replaced.

All dependency repositories and compiled artifacts are excluded from this release's Git tree. See docs/source_manifest.json for the provenance of the research scripts.
