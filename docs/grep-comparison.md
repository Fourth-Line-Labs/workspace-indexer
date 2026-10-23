---
title: "Semantic retrieval against grep and ripgrep"
summary: >-
  A measured comparison between this index and the command-line tools an agent
  would otherwise reach for, over one 24-case dataset and one real workspace.
  Records the method, the two corpus confounds that invalidated earlier
  attempts, two predictions the results falsified, and the decision rule the
  numbers support.
created: 2026-10-18
updated: 2026-10-23
tags: [evaluation, retrieval, grep, ripgrep, benchmark]
status: current
audience: "an engineer deciding whether an agent should call this tool or just grep"
source_ref: "feat/grep-comparison @ ace1b3c"
---

# Semantic retrieval against grep and ripgrep

The pitch for this tool is that an agent should call it instead of grepping.
That claim had no measurement behind it. This is the measurement.

## What was predicted beforehand

Written into [issue #103](https://github.com/Fourth-Line-Labs/workspace-indexer/issues/103)
before any run:

> ripgrep should win outright on exact-token queries, and win on latency by two
> to three orders of magnitude.

Latency held. The lexical claim is half right: ripgrep **ties** on recall and
loses on rank. An intermediate version of this document claimed ripgrep lost
outright — that was an artefact of a broken corpus, described below.

## Method

One dataset, 24 cases, three groups of eight:

- **retrieval** — can the index find a thing at all. Written to defeat lexical
  matching: *"described nowhere by that name in the code"*.
- **guidance** — *how am I supposed to build this?* The wrong answers sit in the
  same directory with the same vocabulary.
- **lexical** — added for this comparison. Queries whose answer is a token that
  appears verbatim. **The control group, where grep is expected to win.**
  Without it the comparison could only report a victory, which is not a
  measurement.

Every case carries a hand-written `grep_pattern`, written *before* any run and
written to win, derived from the question and never from the answer file. They
live in `config/eval.yaml` so anyone can argue a pattern was too weak.

Lexical hits rank by **match count descending** — the most favourable honest
ordering, since grep returns an unordered set.

Beyond recall and MRR, two metrics recall cannot express: **breadth** (files
matched before the cutoff) and **flood rate** (share of queries matching more
than 3x the cutoff). Recall scores a 400-file match containing the right file
as a perfect success; for an agent it is a haystack.

**All three tools search the same corpus**, using the index's *effective*
exclusion set — `HARDCODED_EXCLUDES`, the dataset in use, and the lockfiles
`Walker._consider` rejects through `is_lockfile` rather than through any
pattern. Not just the user-editable list. That sentence has been false twice
and is the single most important thing on this page.

## Results

`-n 10`, ~1,700 indexed files.

| group | tool | recall@10 | MRR@10 | median latency | median breadth | flood |
|---|---|---|---|---|---|---|
| **all (24)** | search | **0.875** | **0.715** | 509 ms | — | — |
| | ripgrep | 0.771 | 0.446 | **16 ms** | 23 | 38% |
| | grep | 0.771 | 0.443 | 56 ms | 27 | 42% |
| **retrieval (8)** | search | **1.000** | **0.615** | 494 ms | — | — |
| | ripgrep | 0.625 | 0.281 | **18 ms** | 56 | 62% |
| | grep | 0.625 | 0.281 | 58 ms | 61 | 62% |
| **guidance (8)** | search | 0.625 | **0.625** | 491 ms | — | — |
| | **find_guidance** | **0.938** | 0.768 | 498 ms | — | — |
| | ripgrep | 0.688 | 0.378 | **19 ms** | 44 | 50% |
| | grep | 0.688 | 0.373 | 59 ms | 52 | 62% |
| **lexical (8)** | search | **1.000** | **0.906** | 502 ms | — | — |
| | ripgrep | **1.000** | 0.680 | **13 ms** | 4 | 0% |
| | grep | **1.000** | 0.675 | 42 ms | 4 | 0% |

**grep and ripgrep now agree to within 0.006 on every group.** That is the strongest evidence
the corpus is finally shared: the two binaries differ in gitignore handling and
little else, so once they search the same files they should score the same, and
they do.

## Where the predictions failed

**The index does not beat lexical search on exact tokens.** All three reach
1.000 recall on the lexical group with a median breadth of 4 and *zero* floods.
The index wins rank (0.906 against 0.680) because count-ordering puts
documentation above the one line that sets a value, but an agent reading four
paths does not need rank 1. **On a query naming a token, use ripgrep.**

**The index loses the guidance group to both tools on recall** — 0.625 against
0.688. That was supposed to be the strongest argument for semantic retrieval
and it is the weakest result here. The tool an agent actually calls,
`find_guidance`, reaches 0.938 with its document-type filter, so the capability
is real; plain `search` does not have it.

## Two corpus confounds, both found after the numbers were published

Every claim on this page rests on the three tools searching the same files.
Both times that has been false, the defect was invisible in the output: the
numbers looked ordinary, and only a reviewer reading the exclusion path found
them. They are recorded together because the pattern is the finding.

### The first: the harness searched its own answer key

The first version of this document reported ripgrep at 0.646 overall and grep
at 0.167, and concluded that the index beat ripgrep even on exact tokens. That
was wrong, and the cause is worth recording because the harness is built to
prevent exactly it.

`config/excludes.py` keeps `evals/**`, `logs/**` and the baselines table out of
the index, and its docstring says why: a persisted record quotes every query
verbatim, making it *"a near-perfect lexical and semantic match for the very
queries used to measure retrieval quality — the measurement then scores our own
output instead of the workspace. A correctness rule, not a preference."*

The lexical tools were handed only `index.exclude`, the user-editable list. So
they searched `evals/*.json`, the JSONL logs **and `config/eval.yaml` — the
dataset, which contains every query and every pattern.** `eval.yaml` came back
rank 1 for five of eight lexical queries. The tools were being scored against a
file containing the answer key, while the index never saw any of it.

Two secondary effects of the same bug: grep additionally walked gitignored logs
that ripgrep skips natively, which is why it collapsed to 0.167 and why its
latency was 421 ms rather than 57 ms.

Fixed by translating the full effective exclusion set — including rooted
`logs/**`-shaped globs, previously untranslatable — and by excluding this
document, which quotes the queries too.

### The second: lockfiles the index never indexed

`all_excludes` is a pattern list, and lockfiles are not excluded by a pattern.
`Walker._consider` drops them through `is_lockfile`, so no translation of the
exclusion set could have found them and the lexical tools searched
`poetry.lock`, `Cargo.lock`, `uv.lock` and `flake.lock` while the index had
never seen any of them.

Seven lockfile hits across the 24-case grep run; five of them filled half the
top ten of one query — *"how do we avoid re-embedding a file that was only
reformatted"* — pushing the expected file to rank 8. Unlike the first confound,
this one biased **against** the lexical tools, and correcting it raised their
MRR on every group that contained a lockfile hit: 0.425 to 0.443 overall, 0.234
to 0.281 on retrieval. The eight affected records were withdrawn and the runs
repeated; the `search` rows were never affected, since the index excluded
lockfiles all along.

The exclusions a lexical baseline needs are therefore *not* a list that exists
anywhere in the configuration. They are whatever the walker actually does, and
the only honest way to track that is a test per rule.

## A second finding: the index can be stale

An earlier lexical run missed *"find the WorkspaceRegistry class definition"*
and returned `worktree_registry.py`. The manifest showed `workspace_registry.py`
was not indexed at all — it landed in a PR merged after the last index run.
Reindexing (348 files, 543k tokens) fixed it.

**The index can be stale; grep never is.** A tool answering about the tree as
it was last week is wrong in a way the reader cannot detect from the answer.
`watch` exists for this and was not running.

## What this does not establish

- One workspace, ~1,700 files, 24 cases. Eight per group sees a large effect,
  not a small one.
- The dataset author and the tool author are the same person.
- Patterns are hand-written. An agent-generated pattern is the realistic
  baseline and is [part 2 of #103](https://github.com/Fourth-Line-Labs/workspace-indexer/issues/103),
  unrun.
- Match-count ranking is a choice, and it drives the MRR gap. An agent scanning
  ten paths is not bound by rank 1, so MRR understates the lexical tools.
- No measurement of tokens delivered to the agent.

## The decision rule the numbers support

Not "use the index instead of grep":

1. **Know the token? Use ripgrep.** Identical recall, 4-file breadth, no
   floods, 13 ms, no staleness, no API key. The index's better ranking does not
   justify 38x the latency when the answer is one of four files.
2. **Describing behaviour rather than naming it? Use the index.** This is the
   one unambiguous win: 1.000 against 0.625 recall, 0.615 against 0.281 MRR,
   and the lexical tools flood on 62% of those queries while the index cannot.
3. **Looking for specs or design docs? Use `find_guidance`, not plain search.**
   Plain `search` loses this group to ripgrep on recall; the document-type
   filter is what makes it 0.938.
4. **Grepped and got nothing, or got 60 files? That is the index's cue.** The
   lexical tools flood on 38-42% of all cases and 62% of the retrieval group. An
   empty result and a flood are both detectable at the call site without
   knowing the answer.

Reproduce with `workspace-indexer eval --tool search|ripgrep|grep|find_guidance
--group all|retrieval|guidance|lexical`. Per-case detail is in `evals/*.json`.
