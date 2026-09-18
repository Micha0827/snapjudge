# Contributing

Issues and pull requests are welcome. Please keep changes small and focused.

## Run the tests

```bash
pip install -e ".[test]"
pytest
```

The tests need Apple Silicon (they import `mlx`), but load no model and download nothing. CI runs
them on `macos-14` for every pull request.

## Add obstacle sentences to Snap Run

The game's sentences live in `snapjudge/game/obstacles.en.json` and `obstacles.de.json`, as
`{"label", "kind", "text"}` entries under `obstacles`. `label` is the only correct action (`run`,
`jump`, `long_jump`, `duck`), `kind` is `normal`, `tricky` or `trap`. Say *where* the obstacle is
(on the ground, a gap, overhead): the model reads literally. Tricky sentences and traps are
especially useful; add them to both languages where possible. Check a new set with
`python eval/runner_labels.py --lang en` against a running server and include the result in the PR.

## Reproduce the benchmarks

See [Evaluate](README.md#evaluate) and
[Comparison with Jev](README.md#comparison-with-jev-on-typesafes-public-cases) in the README.
When you report numbers, state the chip, memory, model (with quantization), the command you ran
and whether the GPU was otherwise idle. Result files go to `results/`.
