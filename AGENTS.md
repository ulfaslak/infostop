# About

Infostop detects stop locations in mobility data. It takes time-ordered location traces (GPS or otherwise) and labels each point with the stop it belongs to, or `-1` if the point was recorded while moving. The method is published in Aslak & Alessandretti (2020), [arXiv:2003.14370](https://arxiv.org/abs/2003.14370) (see `CITATION.cff`).

It is a small Python library with one C++ extension, published on PyPI as `infostop`. Researchers cite the paper and build on its labels, so **reproducible output is part of the product**.

The core algorithm is sound and settled. Most of what is broken is around it: packaging, dependency pins, the post-processing and plotting helpers, and the docs. **Fix the edges. Do not redesign the middle.**

# Development

## Testing ideology

- Tests build small synthetic traces inside the test: a few Gaussian blobs of points with timestamps, plus moving segments between them. No data files, no network access. Use `np.random.default_rng(<seed>)`, never NumPy's global random state.
- **Every bug fix lands with a test that fails without the fix.** Run the negative control (revert the fix, watch the test fail, re-apply).
- **Characterization tests before touching the pipeline.** Pin the partition the current code produces on a fixed trace, for single-user and multi-user input and both metrics. Compare partitions **up to relabelling**: label integers are arbitrary, and different Infomap versions may number modules differently. `-1` positions must match exactly.
- Test the C++ step directly through `cpputils.get_stationary_events` for its edges: 0 and 1 points, one event covering everything, the last group, the `min_size` boundary, distance exactly `r1` (the check is `<=`), and time gaps around `max_time_between`.
- Don't test plotting pixels. Test that `plot_map` runs for both model types and every `display_data` option, and returns a folium map.
- CI must test the lowest and the highest supported Python.

## Releasing

**Releases are irreversible.** A version number on PyPI can never be reused, and users pin versions in their papers' code. Never release, and never push a tag, without the human's explicit go in the current session.

The intended procedure (from `README.md`): run `python increment_version.py` once, commit, tag the commit `v<version>` with the version from `metadata.py`, then `git push --tags && git push`. CI then builds and publishes.

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

Given a detailed enough prompt, many decisions can be made autonomously, because they derive nicely from the code and context. Therefore you **default to deciding, then surfacing what was decided**. Decide confidently when the cost of a wrong call is a follow-up PR. **Ask only when:**

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

### Phase 3: Clean up

- **Leave a clean state.** Kill any processes you started, and delete throwaway venvs and build output you made outside your worktree.

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

## Signals from the human

- **The 🤘 signal.** When the human ends a request with 🤘, they're explicitly granting wider latitude and want you to finish the job. Lean harder into your own judgment, decide more, ask less. Merging PRs is allowed. Releasing to PyPI is not (§ Development › Releasing). You must of course still follow the workflow rules and stop in case you are genuinely blocked or about to do something very stupid.
- The human may use speech-to-text. Such messages have characteristic typos like "CLAUDE.md" called "Cloud MD", "Infostop" called "info stop", "Infomap" called "info map". When you see things like that, expect other _near misses_ and adjust your understanding accordingly. If something is genuinely uninterpretable, ask.

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

## Highly valued agent behavior

- **Do it, don't suggest it.** If you can execute a task (build, test, open a PR, file a bug), do it yourself. Never tell the human to run something you have access to run. The exceptions are the public and irreversible actions listed under "Making decisions autonomously".
- **Match the requested mode.** _Explain, review, diagnose, "what do you think about", "why does X happen"_ are **read-only**: answer them, don't start editing files or opening worktrees. _Change, build, fix, add, ship_ carry implementation **and** the verification that goes with it (the full test-fix-learn cycle, not just the edit). When the mode is genuinely ambiguous, answer first and offer the implementation in one line.
- **Report blockers with evidence.** Exhaust the safe in-scope alternatives before declaring yourself blocked. When you are actually blocked, state three things: the exact condition, the evidence for it, and the specific action needed to continue. "Couldn't get X working" is not a blocker report.
