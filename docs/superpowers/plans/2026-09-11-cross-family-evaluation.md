# Cross-family evaluation implementation and verification

Goal: after the existing +2000 clean stop, automatically evaluate nine frozen
policies in three-nine only, without modifying training or its pool.

- [x] Extend `tournament.py` with explicit family filtering and Junhwa actor kind.
  Test: 4+4+1 policies produce 24 pairs / 1200 games, no same-family games,
  excluded matrix cells stay null, rerunning completed pairs does not replay them.
- [x] Connect Junhwa loading in sequential and parallel entrypoints; both use
  the same `selected_pairs()` function. Preserve default round-robin behavior.
- [x] Configure original 69000/67500/42500, branch 69500 (+2000), Junhwa four,
  submission4499. Keep external policies outside training pools.
- [x] Require process exit, stop receipt, final status and both checkpoint
  iteration values (72000 internal = 69500 policy) before evaluation starts.
- [x] Unit/regression verification: 14 tests pass. Existing eight checkpoint
  adapters pass CPU load/action checks; branch +2000 is pending its save.
- [ ] Automated GPU preflight: eight full-episode mirrored two-game matches
  covering all nine models; fail closed on error. These games are not scored.
- [ ] Run 24 mirrored 50-game pairs and save pair records, progress and reports.

Observation contract: shared 214D semantics/action ordering assumed from the
user's common-code provenance; original Junhwa forward-source parity is not
independently established. GRUCell structure, hidden-state carry/reset and
normalization are tested. Deterministic argmax actions; 200-second episodes.

Rank within each family (identical opponent set). Cross-family aggregate scores
have different opponents and are not an unbiased universal ordering. Fifty games
per matchup provide a finite-sample estimate, not proof of the strongest model.

Launch: `evaluation/start_after_training.ps1` waits for the known training PID,
then `evaluation/after_training.py` validates saves, runs GPU smoke, then league.
An exclusive launch.lock prevents duplicate launches. Failure leaves diagnostics
and completed pairs intact; no automatic training resume or retry is performed.
