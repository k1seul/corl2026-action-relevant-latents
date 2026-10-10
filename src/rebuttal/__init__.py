"""CoRL 2026 rebuttal utilities.

This package holds the strictly new code introduced for the rebuttal:
matched-control action shuffling, past-only ("causal-context") extractor
wrappers, action-free adaptation loss, and provenance recording. Existing
scripts under ``src/training`` and ``src/comparison`` are left unchanged so
the submitted result matrix remains reproducible bit-for-bit.

Nothing in this package must be imported by the pre-rebuttal training code.
Nothing here modifies the canonical evaluation probe.
"""
