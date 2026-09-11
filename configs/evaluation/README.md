# Evaluation configurations

Copy `evaluation/models.template.json` here as `<name>.local.json`, then fill
in explicit frozen model paths. Relative model paths resolve against the spec
file's parent directory. Head-on and 3-9 use separate lists and output folders.

Current local configs are `three_nine.local.json` and `headon.local.json`.
They refer to copied, categorized artifacts and are not committed or pushed.
Only the 20k 15000/17500/20000 models and 4499 are shared between them.

Adding an external package to storage does not add it to a tournament.
