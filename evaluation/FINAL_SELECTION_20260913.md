# Final 3-9 evaluation

- Candidates: continuation 20k, 23k, 26k, 30k, 32k (previous average win rate >=85%).
  69k was removed at the user's request, including from the opponent set.
- Opponents: Junhwa final, GRU, wide v2, survive v1, survive 15500; 4499;
  Cutoff 60Hz and 10Hz; hard-deck dive. Shin BTs and MPC were removed.
- All RL: argmax 10Hz, using the established native RL observation provider.
  Derivatives and action-history lags remain 0.1 seconds. Frozen normalizers and
  architecture adapters are retained. GRU updates hidden state every decision and
  resets it each episode. Legacy 4499 keeps raw throttle history.
- Hard-deck dive: command each 60Hz physics frame. Cutoff: explicit action-repeat 1/6.
- Each matchup: 100 unique seeds, 50 left and 50 right, no mirrored reuse of a seed.
  The same independent seed bank is reused across candidates for fair comparison.
- Total: 5 x 9 x 100 = 4,500 games, CPU 7 workers, 200-second limit, 304.8m deck.
- Results: W/L/D, win rate W/N, score (W+0.5D)/N, side totals, HP difference,
  own/opponent crash counts, mutually exclusive loss reasons (crash, destroyed,
  both, timeout HP, other). These describe terminal observations, not causality.
- Faults are errors, never silently converted to draws or excluded games.
  Finished 10-game chunks persist. Re-run identical command to resume.
- All final games are newly evaluated: previous 10Hz heatmap results are not mixed in.
- Output: `report.json`, `final_results.txt`, `final_heatmap.png` after completion.

Run from repository root:

```powershell
python -u evaluation/final_selection.py --output artifacts/evaluations/final_selection_10hz_reduced_20260913 --workers 7
```

Verification before the revised 10Hz launch: 9 unit tests passed, then 26 short smoke games
covering all candidates and opponent adapters completed. Each RL provider's
physical frame count and decision count are checked in every game. Short smoke
games test compatibility, not playing strength. Ranking is only over this fixed
opponent set; small differences do not establish a universally strongest model.
