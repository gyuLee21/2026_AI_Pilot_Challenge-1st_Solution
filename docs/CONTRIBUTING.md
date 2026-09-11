# Contribution and commit conventions

Use one intent per commit: `type(scope): imperative summary`.

- `feat(storage): add scenario-scoped training launcher`
- `fix(eval): reject mixed scenario rosters`
- `refactor(league): separate roster selection from commit`
- `test(storage): cover resume path validation`
- `docs: document external model review`
- `chore(training): synchronize verified runtime snapshot`

Commit messages describe actual changes; do not label a behavioral change as
formatting/refactoring. Explain important compatibility changes in the body.
Prefer a focused feature branch and reviewed, non-force pushes to this project's
existing origin. Never rewrite published history without explicit approval.

Before commit/push:

1. Run the relevant tests and `git diff --check`.
2. Review `git diff --cached --stat` and `git diff --cached`.
3. Keep weights, archives, teammate source, generated binaries, local rosters,
   authentication data and run logs out of the index. Never use `git add -f`
   for artifacts to bypass this boundary.
4. Preserve original third-party authorship and licensing; an ignored local
   copy is not permission to redistribute it.
5. Do not move or modify code currently used by an evaluation process. Freeze
   run identity and use a new output directory for changed inputs.

The old historical commit is retained. New commits use this convention;
past published messages are not rewritten solely for stylistic consistency.
