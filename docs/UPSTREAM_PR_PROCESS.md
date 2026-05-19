# Upstream PR process

The single source of truth for how we contribute to `keras-team/keras`.
Each PR is a focused, atomic addition that builds on the previous one.

## TL;DR per-PR checklist

For any PR in the sequence:

1. **Construct the branch** from the previous PR (or `upstream/master` for PR 1 / 6).
2. **Edit code + tests** with only that PR's scope.
3. **Run validation** — `python pr_validation/validate.py <PR_NUMBER>`.
4. **Push the branch** to `brianmanderson/keras`.
5. **Wait for maintainer ack** on the upstream issue.
6. **Open the upstream PR** with `gh pr create --repo keras-team/keras --base master --head brianmanderson:pr/NN-name`.
7. **Iterate on review** — push to the same branch; GitHub auto-updates the PR.

If `validate.py` fails: do not push. Fix locally, re-run, then push.

## The 7-PR sequence

Each row is one upstream PR. Each is meant to be small enough to merge in
isolation; dependencies are explicit.

| # | Branch | Adds | Depends on | Risk |
|---|---|---|---|---|
| 1 | `pr/01-basic-reconstruct` | Basic non-overlap `reconstruct_patches` + `ReconstructPatches{2,3}D` layers | upstream/master | Low |
| 2 | `pr/02-overlap-reduction` | Overlap path (`strides < size`) + `reduction="mean"`/`"sum"` | PR 1 | Low |
| 3 | `pr/03-channels-first` | `data_format="channels_first"` support | PR 1 | Med (TF-CPU skip) |
| 4 | `pr/04-dilation-rate` | `dilation_rate` parameter (overlap path) | PR 2 | Med (TF-CPU skip) |
| 5 | `pr/05-ergonomics` | Auto-infer `output_size` for `valid`; better errors | PR 1 | Low |
| 6 | `pr/06-extract-patches-layers` | `ExtractPatches2D` / `ExtractPatches3D` Layer wrappers | upstream/master (independent) | High (thin-wrapper rejection risk) |
| 7 | `pr/07-dynamic-output-size` | Dual-input `[patches, reference]` mode | PR 1 | High (niche API) |

**Issue drafts** for each PR live in [docs/pr-issues/](pr-issues/). Open the issue,
wait for ack, then open the PR.

## Constructing a new PR branch

Standard pattern (replace `NN-name` and the parent branch as needed):

```bash
cd C:/Users/BRA008/Modular_Projects/keras

# Always start clean
git fetch upstream master
git checkout pr/NN-name-of-parent  # or `upstream/master` for PR 1 / 6
git pull --ff-only

# New branch
git checkout -b pr/NN-name

# Make the changes for this PR only (use the full mirror as a reference:
# `git checkout backup/full-mirror -- <path>` to copy a file, then edit)

# Validate
python C:/Users/BRA008/Modular_Projects/tensorflowwork/pr_validation/validate.py NN

# Commit + push
git add keras/
git commit -m "..."
git push -u origin pr/NN-name
```

## Validation

`pr_validation/validate.py` is the gate. It:

1. Checks you're on the right branch (`pr/NN-*`).
2. Adds the keras checkout to `PYTHONPATH` so `import keras` resolves to
   it (no pip install needed; site-packages is bypassed).
3. Under each backend (TF, JAX, PyTorch):
   - Runs the **focused keras-checkout tests** that live alongside the
     upstream code.
   - Runs the **`pr_validation/direct_tests/` suite** subset selected for
     this PR. Direct tests import `keras.layers.X` / `keras.ops.image.X`
     directly and use feature-detection markers
     (`direct_tests/conftest.py`) to skip tests for features not yet
     present in the installed keras.
4. Reports pass / fail per backend and per suite.

Exits non-zero if anything fails. Don't push if it fails. See
`pr_validation/README.md` for more.

## What if validation fails?

| Failure | Likely cause | Action |
|---|---|---|
| Import error during install | requirements changed or backend missing | `pip install -r requirements.txt` in tensorflowwork's `.venv` |
| Test fails on TF but passes on JAX/torch | TF-CPU op limitation (NCHW conv, dilated conv_transpose) | Confirm by reading the test docstring; if expected, add a `skipif` |
| Roundtrip identity fails | Likely a padding-distribution bug | Compare with the corresponding test in the full mirror (`backup/full-mirror`) |
| All three backends fail same way | Logic bug in the PR's code | Diff against `backup/full-mirror` to find what's different |

## Rebasing when upstream changes

When `keras-team/keras` advances and we need to rebase one of our PR branches:

```bash
cd C:/Users/BRA008/Modular_Projects/keras
git fetch upstream master
git checkout pr/NN-name
git rebase upstream/master
# Resolve conflicts if any (likely only in image.py and the api/ inits)
git push --force-with-lease origin pr/NN-name
```

When the previous PR in the chain merges upstream and our branch needs to
reflect that:

```bash
git checkout pr/02-overlap-reduction
git rebase upstream/master  # PR 1 is now part of master
git push --force-with-lease origin pr/02-overlap-reduction
```

## Safety net (never lose work)

Three handles on the full feature set, kept indefinitely on `origin`:

- **Tag** `pr-mirror/full` — immortalized commit with every feature.
- **Branch** `backup/full-mirror` — same commit, second handle.
- **TensorflowAutoEncoder repo** — the local prototype has every feature
  as its own commit in git log; can always be re-ported.

To recover a feature from the full mirror after losing local work:

```bash
git checkout pr/NN-name
git checkout backup/full-mirror -- keras/src/path/to/file.py
# Edit to keep only this PR's scope, then commit
```

## Opening the upstream issue (before the PR)

For each PR:

1. Read the corresponding `docs/pr-issues/pr-NN-name.md`.
2. Copy its content to a new issue at https://github.com/keras-team/keras/issues/new.
3. **Wait for a maintainer to ack** (usually within a week).
4. Only then open the PR via:
   ```bash
   gh pr create --repo keras-team/keras --base master \
     --head brianmanderson:pr/NN-name \
     --title "..." \
     --body-file docs/pr-issues/pr-NN-name.md
   ```

If the maintainer rejects the issue: the working branch and tests still
have value as a standalone PyPI package or as a contribution to
`keras-cv` / `keras-hub`. The code does not become useless.

## When a PR merges upstream

When PR N lands on keras-team/keras:

1. Update local upstream: `git fetch upstream master`.
2. Update fork's master: `git checkout master && git merge --ff-only upstream/master && git push origin master`.
3. Rebase the next PR in the chain (PR N+1) onto the new master.
4. Re-run `validate.py N+1` to confirm clean.
5. Push the rebased branch.
6. If the upstream issue for PR N+1 is already ack'd, open the PR.

## Summary diagram

```
upstream/master
  │
  ├─── pr/01-basic-reconstruct ◄── PR 5 (ergonomics) builds here too
  │      │
  │      └─── pr/02-overlap-reduction
  │               │
  │               └─── pr/04-dilation-rate
  │
  ├─── pr/01-basic-reconstruct ◄── PR 3 (channels_first) builds here
  ├─── pr/01-basic-reconstruct ◄── PR 7 (dual-input) builds here
  │
  └─── pr/06-extract-patches-layers (independent of all others)
```

PRs 3, 5, 6, 7 can be submitted in parallel after PR 1 merges (assuming
maintainer ack on their issues). PR 4 waits for PR 2. The dependency
graph is reflected by the branch parent.
