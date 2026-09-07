"""Phase J -- resource management and safe long-run execution infrastructure.

This package is INFRASTRUCTURE ONLY: it builds the job state machine,
execution-budget wiring, process-ownership tracking, checkpoint/resume
contract, resource-accounting ledger, and multi-seed bookkeeping a future
training-heavy experiment (e.g. an eventual EXP-0006) would use. Nothing in
this package launches, trains, or benchmarks anything real -- every runner
used by this package's own tests is a fake/mock. See
reports/phase_j/PHASE_J_IMPLEMENTATION.md for the full architecture writeup.
"""
