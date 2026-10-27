# About

Infostop detects stop locations in mobility data. It takes time-ordered location traces (GPS or otherwise) and labels each point with the stop it belongs to, or `-1` if the point was recorded while moving. The method is published in Aslak & Alessandretti (2020), [arXiv:2003.14370](https://arxiv.org/abs/2003.14370) (see `CITATION.cff`).

It is a small Python library with one C++ extension, published on PyPI as `infostop`. Researchers cite the paper and build on its labels, so **reproducible output is part of the product**. The repo was unmaintained from 2022 to 2026 and owes its users bug fixes. That is the current work.

The core algorithm is sound and settled. Most of what is broken is around it: packaging, dependency pins, the post-processing and plotting helpers, and the docs. **Fix the edges. Do not redesign the middle.**

# Persistent agent context

## Hard context: DNA

The four sections under "DNA" below are the project's architectural guardrails: technical decisions, structure, runtime contracts and how to work on the code. The DNA grows with the project, but must never drift from the code. It evolves but doesn't change. Contributions that violate DNA cause cancer and must be avoided. Rules:

1. Don't write code which violates DNA.
2. Grow DNA: when your work adds new structure, record it.
3. Don't let it drift: if something in DNA no longer matches the code, fix it. If it's unclear whether DNA or code should change, think deeply and resolve it only if you are certain, otherwise ask the human.
4. Do not take DNA changes lightly. If you make changes, you must have applied deep reasoning before doing so. Err on the side of asking the human before changing an existing DNA item.

**Each section holds one class of facts.** A fact is only recorded in one place.

| Section          | Holds                                                                 | Test                                                                                      |
| ---------------- | --------------------------------------------------------------------- | ----------------------------------------------------------------------------------------- |
| § Decisions      | A **choice** among alternatives we could have made differently.       | Could someone violate it by choosing otherwise? Disagreeing means arguing with the human. |
| § Invariants     | What must stay **true at runtime**.                                   | Could you write a test that fails when it stops being true?                               |
| § Architecture   | **Where** code lives: files, modules, the pipeline-to-code map.       | Could you verify it with `ls` or by opening the file?                                     |
| § Development    | How to **work on** the library: setup, testing ideology, releasing.   | Is it a thing you _do_, not a thing the library does?                                     |

Statements migrate as they change kind. A decision that has been implemented and now has guards around it usually belongs in § Invariants. If a section grows past roughly 80 lines, move it to `AGENTS/DNA/<SECTION>.md` and leave a one-line pointer here.

## Soft context

- **GitHub issues** on `ulfaslak/infostop` are the single source of truth for owed work: bugs, feature requests, user questions. Run `gh issue list` to see it, and look at defects first. Several issues are years old; check whether a later commit already fixed one before you start (e.g. #14 was fixed by PR #26 but is still open).
- **Claiming work.** The repo is public, so don't claim issues with comments. Claim an issue by opening a **draft PR** early whose body says `Fixes #<N>`. Before picking an issue, check that no open PR already references it: `gh pr list --state open --search "<N> in:body"`.
- **Filing a bug.** When you find a bug you will not fix in your current task, file it as an issue (one defect per issue, with repro steps, expected vs actual, and the Python and dependency versions). It is public: write it short and factual.
- **AGENTS/AGENT_MISTAKES.md** collects the mistakes agents made here. Create it on the first entry (see Phase 3). Before designing a change, grep it for your area.
- **AGENTS/DEFERRED.md** tracks **technical items** that are known but intentionally postponed: tech debt, known limitations, cleanups better bundled with later work. **Do NOT put features or bugs here**; those are GitHub issues. Each item has **What**, **Why deferred** and **Trigger**. Create it on the first entry.
- **AGENTS/HUMAN_TODO.md** lists tasks only the human can do (PyPI account and tokens, GitHub repo settings, Read the Docs admin). Create it on the first entry.
- **§ State of the repo** and **§ Environment notes** at the end of this file are snapshots, not DNA. Delete a line when its fix merges.

# DNA

## Decisions

- **Clustering is two-level Infomap** on a network that links stationary points closer than `r2`. Flow-based clustering is the paper's contribution. Replacing or tuning it is not a bug fix.
- **Step 1 (stationary events) runs in C++** via pybind11 (`cpputils/main.cpp`), for speed. Python orchestrates. Don't move the per-point loop into Python.
- **A stop event's location is the coordinate-wise median** (median of latitudes, median of longitudes), not a geometric median.
- **Neighbour search is scikit-learn's `BallTree.query_radius`.**
- **Two distance metrics only:** `"haversine"` (inputs in degrees, `r1`/`r2` in meters) and `"euclidean"` (input units).
- **Multi-user mode:** a list of arrays is clustered as one shared network, so stop labels are shared across users. This is a headline feature of the paper.
- **The public API is published and cited in other people's code.** `Infostop`, `SpatialInfomap`, `plot_map`, `compute_intervals`, the constructor keyword names and their defaults, including the misspelt `min_spacial_resolution`. Don't rename or remove any of it. If a name must change, add the new name and keep the old one as an alias that emits a `DeprecationWarning`.
- **The removed functional API** (`label_trace`, `label_static_points`, `label_network` at package level) stays as stubs in `infostop/deprecated.py` that raise with a pointer to the new API.
- **`infostop/metadata.py` is the single source for the version.** `setup.py` execs it.
- **Formatting is `black`.** We don't argue about formatting.

## Invariants

- **Labels align with input rows.** `Infostop.fit_predict` returns one label per input row, in input order. An array in gives a 1-D array out. A list of arrays in gives a list of 1-D arrays out, one per input array, with matching lengths. `SpatialInfomap.fit_predict` takes one `(N, 2)` array and returns `N` labels.
- **`-1` means "not a stop"**: moving, too few points (`min_size`), too short (`min_staying_time`), or a singleton when `label_singleton=False`. Stop labels are integers ≥ 0. In multi-user mode the same label means the same place for every user.
- **Columns:** 0 is latitude, 1 is longitude, 2 (optional) is time, ascending. Time is in whatever unit the caller uses; `min_staying_time` and `max_time_between` are in that unit (the defaults, 300 and 86400, assume seconds).
- **Earth radius is 6371000 m** in `cpputils/main.cpp` (`haversine`) and in `infostop/utils.py` (`query_neighbors` and `infomap_communities`). All three must agree.
- **Step 1 and step 3 use the same `distance_metric`.**
- **`query_radius` returns every point as its own neighbour.** A node is a singleton if and only if `len(neighbors) == 1`. Singletons are left out of the Infomap network and labelled afterwards (own label, or `-1`).
- **Edge weight is `max(counts[a], counts[b])`**, times `distance ** -weight_exponent` when `weighted=True`. `counts` are the multiplicities from the deduplication in step 2.
- **Both downsampling steps are exactly reversible.** Step 5 maps labels back through `inverse_indices`; step 6 maps them back through `event_map`. In step 6, `event_map` uses `-1` for "not a stop", and the code relies on Python negative indexing into a `-1` appended to the end of the label vector. Change one, change both.
- **Labels for valid input don't change silently.** The labels the current code produces are the reference. A change that alters labels for valid input is a behaviour change, not a fix, unless an issue shows the old labels were wrong. Pin the current output with a characterization test (§ Development) before you touch steps 1 to 6.

## Architecture

| Path                           | What                                                                                                                       |
| ------------------------------ | -------------------------------------------------------------------------------------------------------------------------- |
| `infostop/__init__.py`         | Public API re-exports. Also runs `import cpputils`.                                                                        |
| `infostop/models.py`           | `Infostop` (steps 1 to 6) and `SpatialInfomap` (steps 2 to 5 on unordered points). Input checks are `assert`s.             |
| `infostop/utils.py`            | `query_neighbors` (BallTree), `infomap_communities` and `label_network` (Infomap), `convex_hull` and `max_pdist` (plotting). |
| `infostop/postprocess.py`      | `compute_intervals`: labels and times to `[label, start, end]` rows.                                                       |
| `infostop/visualize.py`        | `plot_map` and `FoliumMap` (folium). Haversine models only.                                                                |
| `infostop/deprecated.py`       | Stubs for the removed functional API.                                                                                      |
| `infostop/metadata.py`         | `__version__` and author metadata.                                                                                         |
| `cpputils/main.cpp`            | pybind11 extension with one function, `get_stationary_events`. Built as the **top-level module `cpputils`**, not `infostop.cpputils`. |
| `setup.py`                     | setuptools build with the pybind11 `Extension` and a custom `BuildExt` for compiler flags.                                 |
| `tests/`                       | pytest. `test_models.py` holds only a placeholder.                                                                         |
| `docs/`                        | Sphinx sources. `docs/_build/` is committed build output (despite being in `.gitignore`).                                  |
| `increment_version.py`         | Bumps the patch version and copyright year in `metadata.py`.                                                               |
| `.github/workflows/deploy.yml` | CI: tests on push and PR to `master`, then build and PyPI publish steps gated on tag refs.                                 |

The pipeline, as numbered in the `Infostop` docstring:

| Step | What                                                                                                                                                                                                                                                                      | Where                                                                     |
| ---- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------- |
| 1    | Group consecutive points into stop events. A point joins the current group if it is within `r1` of the group's running median (and, with time, within `max_time_between` of the previous point). A group is kept if it has ≥ `min_size` points (and lasts ≥ `min_staying_time`). | `cpputils.get_stationary_events`, returns `(event medians, event_map)`    |
| 2    | Optionally round to a `min_spacial_resolution` grid, then deduplicate with counts.                                                                                                                                                                                         | `models.py`, `np.unique(..., axis=0, return_inverse=True, return_counts=True)` |
| 3    | Find neighbours within `r2`.                                                                                                                                                                                                                                               | `utils.query_neighbors`                                                   |
| 4    | Build the network and run two-level Infomap.                                                                                                                                                                                                                               | `utils.label_network` → `utils.infomap_communities`                      |
| 5    | Undo step 2.                                                                                                                                                                                                                                                               | `self._stat_labels[inverse_indices]`                                      |
| 6    | Undo step 1 and split per user.                                                                                                                                                                                                                                            | `event_map` loop at the end of `Infostop.fit_predict`                     |

`plot_map` reads the fitted model's private attributes (`_is_fitted`, `_distance_metric`, `_data`, `_stat_coords`, `_stat_labels`). Renaming them breaks plotting.

## Development

### Setup

Each worktree gets its own virtual environment and its own build of the C++ extension:

```bash
uv venv -p 3.13 .venv
uv pip install --python .venv -e . pytest black
```

Python edits are live in an editable install. After you edit `cpputils/main.cpp`, re-run the install to rebuild the extension. The editable build puts `cpputils.<abi>.so` in the repo root. Because the extension is the top-level module `cpputils`, the first `cpputils` on `sys.path` wins, and a stale `.so` in the repo root shadows a fresh build. When behaviour makes no sense, check `python -c "import cpputils; print(cpputils.__file__)"`.

The `Makefile` predates this file: `make install` runs the deprecated `python setup.py install`, and `make clean` hardcodes a Python 3.8 path. `make test` (`pytest -ra`) is fine.

### Testing ideology

- Tests build small synthetic traces inside the test: a few Gaussian blobs of points with timestamps, plus moving segments between them. No data files, no network access. Use `np.random.default_rng(<seed>)`, never NumPy's global random state.
- **Every bug fix lands with a test that fails without the fix.** Run the negative control (revert the fix, watch the test fail, re-apply).
- **Characterization tests before touching the pipeline.** Pin the partition the current code produces on a fixed trace, for single-user and multi-user input and both metrics. Compare partitions **up to relabelling**: label integers are arbitrary, and different Infomap versions may number modules differently. `-1` positions must match exactly.
- Test the C++ step directly through `cpputils.get_stationary_events` for its edges: 0 and 1 points, one event covering everything, the last group, the `min_size` boundary, distance exactly `r1` (the check is `<=`), and time gaps around `max_time_between`.
- Don't test plotting pixels. Test that `plot_map` runs for both model types and every `display_data` option, and returns a folium map.
- CI must test the lowest and the highest supported Python.

### Releasing

**Releases are irreversible.** A version number on PyPI can never be reused, and users pin versions in their papers' code. Never release, and never push a tag, without the human's explicit go in the current session. This holds under 🤘 too.

The intended procedure (from `README.md`): run `python increment_version.py` once, commit, tag the commit `v<version>` with the version from `metadata.py`, then `git push --tags && git push`. CI then builds and publishes. This pipeline is currently broken (§ State of the repo).

# Workflow rules

In sessions where you edit files in this repository you must follow certain rules. Sessions where files aren't being changed are not covered by these rules unless explicitly stated.

## Always use worktrees when editing files in this repository

Branch from `origin/master`. Edits on `master` are forbidden and there are **no size exceptions** (read-only tasks such as research, exploration and answering questions require no worktree). Use `git gtr`:

1. **List:** Run `git gtr list` to see all existing worktrees.
2. **Create:** From the primary clone, run `git gtr new <branch-name>` (example: `git gtr new fix/infomap-2-api`). There is no `.gtrconfig` here, so gtr does not set up Python for you: create the venv and install as in § Development › Setup.
3. **Work:** `cd` into the worktree path shown by gtr. All edits, commits, and pushes happen there. **Commit early and frequently in worktrees.** This is to minimize the risk of losing work in the unlikely case that the worktree is deleted.
4. **After merge:** From the primary clone, remove your worktree. Run `git gtr rm <branch-name>`, then `git checkout master && git pull`. If the worktree directory is deleted while it's your cwd, the session becomes permanently stuck. **Never remove another agent's worktree.**

**One issue per PR.** Users read these PRs to decide whether to upgrade. Keep each PR to one fix, and put `Fixes #<N>` in the body.

## PR body structure

At minimum, include `## Summary` and `## Test plan` (checklist), and `Fixes #<N>` when the PR closes an issue. Add `## Decisions taken without asking` if any autonomous scope calls were made. Add `## Verification` with the actual output of the checks you ran in Phase 2. Add `## Mistakes found during self-testing` if you found any. The session ID footer goes last.

- **Include session ID in every PR description.** Before creating a PR, get the current Claude session ID from the `CLAUDE_CODE_SESSION_ID` environment variable (`echo $CLAUDE_CODE_SESSION_ID`). Add it as a footer line in the PR body: `Session: <session-id>`. This lets future agents trace back to the conversation that produced the changes and avoid reverting past decisions without context.

## Making decisions autonomously

Given a prompt and the DNA, most decisions can be made autonomously, because they derive nicely from the context. Therefore you **default to deciding, then surfacing what was decided**. Decide confidently when the cost of a wrong call is a follow-up PR. **Ask only when:**

- The call is **irreversible or public**. E.g. a PyPI release, pushing a tag, commenting on or closing someone else's issue, reviewing or merging an outside contributor's PR.
- It affects **the library's users**. E.g. a change to labels for valid input, a changed default, a renamed or removed public name, a changed meaning of a parameter, dropping a supported Python version.
- It's **architectural and cross-cutting**. E.g. replacing the build system, moving logic across the C++/Python boundary, dropping or swapping a dependency (Infomap, scikit-learn, folium).

**Surface every autonomous decision in the PR body.** Under a `## Decisions taken without asking` section, list each one briefly: what you decided, and why (one line each is enough). Do not document decisions in commit messages.

## The test-fix-learn cycle

**This cycle is non-negotiable for every PR, no size exceptions.** This cycle ensures errors are caught before the human ever sees the PR, and that the project learns from each one.

### Phase 1: First pass PR

The first implementation pass should be your best effort. Think carefully, handle edge cases, get it right.

- Run `pytest -ra` in your worktree's venv, and `black` on the Python files you changed. When writing or modifying tests, read § Development › Testing ideology first.
- Commit, push, and open a PR. The **last commit message** of this phase must include: `[not user-tested]`. This signals that the implementation is complete but hasn't been validated end-to-end yet.

### Phase 2: Self-testing

Pick, without asking, the approaches below that apply. Several can apply at once.

**If the change touches the pipeline (`cpputils/`, `models.py`, `utils.py`):**

1. Run the characterization tests and confirm the partition is unchanged, or that it changed exactly where the issue says it was wrong.
2. Run the README example end to end on a synthetic trace: single-user and multi-user, with and without timestamps, both metrics. Check that labels align with input rows.
3. Test non-obvious edges. For example: a user in a list with no stop events, every point a singleton, a single-row input, repeated timestamps, a tuple of arrays instead of a list, integer-typed input, coordinates near the poles or the antimeridian.

**If the change touches packaging, dependencies or the build:**

1. Build the sdist and wheel (`python -m build`), then install **the built wheel** into a fresh venv on the lowest and the highest supported Python. Do not use an editable install for this.
2. Run a smoke script **from a directory outside the repo**. Editable installs and running from the repo root hide packaging bugs: files missing from the sdist, wrong module paths, a stale `cpputils` build.

**If the change touches plotting:** save the map to HTML (`plot_map(model).m.save(path)`) for each `display_data` option and both model types, open it in a browser and look at it.

**If the change touches docs:** build them (`make -C docs html`) with no new warnings, and run every code example on the changed pages.

**Regardless of what the change is:**

1. **Fix everything you find.** Each bug gets a fix commit on the worktree branch. Push as you go. For each fix, run the **negative control** once: revert the fix (or restore the triggering input), watch the check fail, then re-apply. If adding tests, **break what the test claims to catch, not just the code it covers.**
2. **Tick off the test plan.** As you verify each item in the PR's test plan checklist, update the PR description to check the box (`gh pr edit`).
3. **Stop when confident.** You're done when you can't think of another way to break it.

**Phase 2 routinely takes longer than Phase 1 and produces many fix commits. That's the intention, not a sign something went wrong.** Fast and wrong is worse than slow and right. Don't rush this.

### Phase 2.5: Adversarial subagent review

Launch a cold reviewer subagent before your turn ends.

- **Always apply when the change touches the pipeline or packaging.** Specifically `cpputils/`, `infostop/models.py`, `infostop/utils.py`, `setup.py`, `requirements.txt` or the CI workflow.
- **Skip for docs-only and comment-only changes.**
- **How.** Give it **only the diff and the files it needs to read, not your reasoning, not the issue's framing, not "here's what I built and why."** It must not know the happy path.
- **The subagent reviews; it does not implement.** You triage the findings: fix the real ones as fix commits on the branch (they feed Phase 3), and note in the PR body why you rejected any you didn't act on.

### Phase 3: Document mistakes (only if you found any)

Before asking the human to merge, **if self-testing found real bugs you had to fix**, add a `## Mistakes found during self-testing` section to the PR description listing each issue. Then append to `AGENTS/AGENT_MISTAKES.md` with the date, PR number, and a concise description focusing on the _category_ of error (e.g. "missed edge case in multi-user input", "tested against editable install, not the built wheel"). The goal is to surface patterns that better context or refactors could prevent. **If self-testing found nothing**, add nothing.

### Phase 4: Clean up

- **Leave a clean state.** Kill any processes you started, and delete throwaway venvs and build output you made outside your worktree.

### Phase 5: Post-merge verification

**If you merge a PR in-session, you own Phase 5.** Wait for the CI run on `master` to go green (`gh run list --branch master --limit 3`, then `gh run watch <run-id>`). If it fails, fix forward before anything else.

**If the human approved a release in-session**, the release is not done until you have installed the new version from PyPI into fresh venvs on the lowest and highest supported Python and run the smoke script against it. A green publish step is not a working install.

## Strategies for context management

Be context conscious. Nothing is worth reading twice. Treat context as a resource with a burn rate.

### Don't read large files, grep for structure

For any file over ~300 lines, don't read the whole thing. Run a structural grep to get a table of contents, then read only the sections you need:

```bash
grep -n "^class \|^def \|^    def \|^[A-Z_]* = \|# ([0-9])" path/to/file.py
```

### Keeping context spendable

- **Edit files with the Edit/Write tools, not with `python3`/`perl`/`sed` heredocs.** When a file is written out-of-band the harness re-emits the _entire file_ back into context as a modification notice.
- **Write long prose (PR bodies, issue bodies) to a scratchpad file with Write, then `--body-file` it.**
- **Don't re-read a file you just edited to check the edit landed.** Edit fails loudly if the match missed.
- **Pipe build and test output through `tail`/`grep`.** A failed C++ build prints hundreds of lines; the error is a targeted grep away.

# Collaborating with the human

## About the human

Ulf: Original author of Infostop and of the paper, with Laura Alessandretti. Senior mathematician, data scientist and Python developer; knows the algorithm better than anyone. Not a packaging or devops specialist: when a decision involves build systems, compiler toolchains or dependency resolution, explain the trade-offs concretely in plain language. Strong opinions about code quality and project hygiene. Thinks in systems.

## Signals from the human

- **The 🤘 signal.** When the human ends a request with 🤘, they're explicitly granting wider latitude and want you to finish the job. Lean harder into your own judgment, decide more, ask less. Merging PRs is allowed. Releasing to PyPI is not (§ Development › Releasing). You must of course still follow the workflow rules and stop in case you are genuinely blocked or about to do something very stupid.
- The human often uses speech-to-text. Such messages have characteristic typos like "CLAUDE.md" called "Cloud MD", "Infostop" called "info stop", "Infomap" called "info map". When you see things like that, expect other _near misses_ and adjust your understanding accordingly. If something is genuinely uninterpretable, ask.

## Writing for the human

The last message of your turn should be optimized for the human's understanding. Software engineering jargon is strongly discouraged. Ideally you write ~80% ASD-STE100 compliant. End substantial work with a response that follows this format:

```
<Description of where the work landed. Details from the process are only included if they are relevant to the outcome. Keep this section short.>

⚠️ HEADS UP

<ALL CAPS PIECE OF INFORMATION THAT THE HUMAN MUST ABSOLUTELY TAKE NOTE OF>
<...>

Recap

<Same short recap as generated with /recap>

Next steps

- <...>
- <...>

<status label: Merged/Not merged/Released> · <timestamp: HH:MM:SS (24-hour format)>
```

Guidelines that apply when explaining your work:

- **Lead with the outcome.**
- **Explain decisions, tradeoffs, risks and blockers.**
- **Make the final response self-contained.**
- **Let response depth follow stakes, not effort.**
- **Signal your confidence clearly.** Never conflate what is verified, inferred, and assumed. "Tests pass" is not "it installs on a user's machine"; a green publish step is not a working `pip install`. Never claim success without fresh evidence you actually looked at.
- **Link what you name.** Embed markdown links into anything you name that can carry a link (GitHub issue or PR, source file, docs page), so the human doesn't have to go and find things.

### Status updates during long work

The test-fix-learn cycle can run for an hour with dozens of tool calls. Long silent stretches make a session unreadable, so post a brief update IN ALL CAPS at each **meaningful state change** (a phase boundary, a bug found, an approach abandoned, a blocker hit), insofar as it is worth for the human to know if they scroll through the transcript. Keep it at one-sentence ALL CAPS, with optional short lower-case explanation below.

## Highly valued agent behavior

- **Do it, don't suggest it.** If you can execute a task (build, test, open a PR, file a bug), do it yourself. Never tell the human to run something you have access to run. The exceptions are the public and irreversible actions listed under "Making decisions autonomously".
- **Don't offer to do work the human already asked for.**
- **Match the requested mode.** _Explain, review, diagnose, "what do you think about", "why does X happen"_ are **read-only**: answer them, don't start editing files or opening worktrees. _Change, build, fix, add, ship_ carry implementation **and** the verification that goes with it (the full test-fix-learn cycle, not just the edit). When the mode is genuinely ambiguous, answer first and offer the implementation in one line.
- **Report blockers with evidence.** Exhaust the safe in-scope alternatives before declaring yourself blocked. When you are actually blocked, state three things: the exact condition, the evidence for it, and the specific action needed to continue. "Couldn't get X working" is not a blocker report.
- **Never trigger a skill just because its name appears in the human's text.** "Let's review the clustering step" or "run the example" are English sentences, not skill invocations. Aside from direct invocation, only run a skill when the human directly asks you to.

# State of the repo

Snapshot, verified 2026-10-27. Not DNA. Delete a line when its fix merges.

- **`pip install infostop` fails on current Python.** `infomap==1.0.6` has no wheels for it, and its sdist does not build (it imports `pkg_resources` without declaring it). Issues #16, #20, #22, #23, #24; PR #25 relaxes the pin. Until it is fixed, the § Development › Setup install only resolves with an override: put `infomap>=1` in a file and add `--override <file>`.
- **With the pin relaxed, the code fails on current Infomap.** On Python 3.13 with Infomap 2.16, NumPy 2.5, SciPy 1.18 and scikit-learn 1.9, infostop builds and imports, then `fit_predict` fails with `AttributeError: 'Infomap' object has no attribute 'addNode'`: Infomap 2.x dropped the camelCase API. PR #25 reports no problems with Infomap 2.7.1, so the removal came later.
- **`utils.py` imports `QhullError` from `scipy.spatial.qhull`.** It works with a `DeprecationWarning` on SciPy 1.18, and stops working in SciPy 2.0.
- **Nothing has been published since 0.1.9 (February 2022).** `metadata.py` says 0.1.11 and the git tags say `v1.0.10` and `v1.0.11`. The workflow only triggers on branch pushes and PRs, so its tag-gated build and publish steps never run.
- **The test suite is a placeholder.** CI runs Python 3.9 only, on `actions/*@v2`.
- **Current `black` wants to reformat `infostop/deprecated.py`**, so `black --check .` fails on `master`.
- **The docs describe the removed functional API** (`infostop.detect`, `label_trace`). Issue #27.

# Environment notes

- **macOS build trap.** Some Python builds pass a hardcoded SDK path to the linker (`-isysroot .../MacOSX14.sdk`). If that SDK isn't installed, the extension fails to link with `no such sysroot directory` and `ld: library 'c++' not found`; setting `SDKROOT` doesn't help. On Ulf's Mac (2026-10-27) Homebrew `python3.12` hit this and Homebrew `python3.13` built fine.
