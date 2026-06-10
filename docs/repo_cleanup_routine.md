# Bi-weekly repo cleanup sweep — procedure

Executed by the scheduled cloud routine "ssl-cp repo cleanup" (runs on the
10th and 24th of each month) against the GitHub copy of this repo. The
routine's prompt points here, so editing this file changes the routine's
behavior — no need to touch the schedule. A human (or local agent) can also
run the same procedure by hand.

## Hard guardrails

- **NEVER** delete or touch `output/` or `data/` in any clone or checkout,
  and never run `git clean -x` anywhere. (They are gitignored; on the local
  machine they hold irreplaceable embeddings/results.)
- Never commit to `main` directly. All changes go on a branch
  `routine/repo-cleanup-<YYYY-MM-DD>`; deliverable is a pull request.
- Never archive a module that is imported by any non-archived script
  (check with grep before every move).
- If the latest commit on main is older than ~3 weeks, the local repo may
  be ahead of GitHub: do the read-only checks and open an issue/PR comment
  instead of moving files.

## Sweep steps

1. **Watch list pass.** Read the table in `src/archive/README.md`
   ("Watch list — archive candidates") and `grep -rn "ARCHIVE-CANDIDATE" src/`.
   For each entry whose review date has passed:
   - *Plain candidates*: confirm no substantive commit touched the file since
     the note was added (`git log --since=<note date> -- <file>`; ignore
     repo-wide mechanical commits) and no new reference appeared in
     `docs/findings.md`. If unused -> archive it.
   - *HOLD entries*: check the referenced pending item in `docs/findings.md`
     §10. Archive only if that item is marked done or dropped; otherwise leave
     in place (past-due is fine) and note it in the PR body.
   - Archiving a file = `git mv src/<f>.py src/archive/`, fix its
     `sys.path.insert` (one extra `os.path.dirname` level if it used
     `dirname(abspath(__file__))`), delete its ARCHIVE-CANDIDATE note lines,
     move its watch-list row into the archived-scripts table with a one-line
     supersession rationale, and remove it from the CLAUDE.md active-scripts
     list if present.
2. **Staleness scan.** For every `src/*.py` not on the watch list, find the
   last substantive commit (ignore commits touching >10 files). If a file has
   had none for ~30 days, is not referenced by a pending `docs/findings.md`
   §10 item, is not imported by any active script, and is not pipeline infra
   (`extract_features`, `download_datasets`, `run_conformal_experiment`) or a
   library (`conformal_prediction`, `split_cp_baselines`,
   `exchangeable_features`, `autoencoder_utils`, `mscs_gpu`,
   `macs_experiment`, `mscs_unlabeled_experiment`, `semicp_experiment`):
   add an `# ARCHIVE-CANDIDATE (review <today+2 weeks>): <reason>` note after
   its module docstring and a row to the watch list. Notes only — the actual
   move happens at a later sweep once the review date passes.
3. **Order checks.**
   - `tests/` contains only real tests (currently the two GPU parity tests);
     anything else -> `src/archive/`.
   - CLAUDE.md "Active experiment scripts" matches what exists in `src/`.
   - `src/archive/README.md` covers every file in `src/archive/`.
   - Report (don't fix) references in `docs/` to files that no longer exist.
4. **Verify.** `python -m py_compile` every changed file. If
   `pip install -r requirements.txt` succeeds in the sandbox, also run the
   import sweep: `sys.path.insert(0, "src")` then import every non-archived
   module under `src/`. The GPU parity tests (`tests/`) are NOT part of this
   routine (no GPU in the sandbox); they are required only when
   `conformal_prediction.py` / `split_cp_baselines.py` / `mscs_gpu.py`
   change, which this routine must not do.
5. **Deliver.** Commit per logical step on the routine branch, push, and open
   a PR titled "Repo cleanup sweep <date>" whose body lists: files archived
   (with rationale), new watch-list entries, HOLD entries past their review
   date, and order-check findings. If the sweep finds nothing actionable,
   open nothing — just exit after logging "nothing to do".
