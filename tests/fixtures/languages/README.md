---
title: "Language fixture corpora"
summary: >-
  Authored per-language fixture trees and their exact baselines, measured on
  every pull request with no network and no API key. Explains what the corpus
  is for, what each number means, how to change a fixture, and which buckets it
  deliberately cannot exercise yet.
created: 2026-09-17
updated: 2026-09-17
tags: [testing, fixtures, eval, graph]
status: current
audience: "someone changing an extractor, a resolver, or a fixture"
---

# Language fixture corpora

The retrieval eval needs embeddings, an API key, a vector store, and is
nondeterministic. It cannot gate a pull request. **This can**: import
extraction, namespace extraction, resolution, origin classification and chunk
boundaries are pure functions of the file bytes. Tree-sitter parses locally,
the manifest is a file, resolution is a query. The whole corpus indexes in
about **0.2 seconds** with no credentials.

So this is not a degraded retrieval eval. It measures the half CI was blind to.

## What is asserted

Exact counts, per language, from `baselines.json` — the fixtures are authored,
so the ground truth is known and a band would hide the change it exists to
catch. On top of the counts, three claims that are not just numbers:

- **the first-party gate is whole.** Only a first-party edge can reach a file
  in this corpus, so `first_party_resolved < first_party` is a resolver that
  stopped following something, not a package reference. Asserted separately
  because a baseline would happily record 4 of 5 as a stable number.
- **only the intended files are withheld.** A file the secret scanner keeps out
  leaves no row anywhere — that is what withholding means — so the only way to
  see a false positive is to compare disk against manifest. `CLAUDE.md` calls
  it silent data loss; this is what makes it audible.
- **every fixture file is accounted for.** Indexed plus withheld equals what is
  on disk, so the corpus cannot shrink without a number moving.

## Changing a fixture

Any edit moves the counts, and that is the point — the failure is a claim that
something changed and nobody said so. Re-record deliberately:

1. change the fixture, and say in that language's README what it now exercises
2. run the gate, read the diff it prints (language, field, direction, path)
3. check each moved number against the source by hand — the value of an
   authored corpus is that the truth is known rather than observed
4. re-record `baselines.json` and commit it with the fixture in one change

A baseline regenerated without step 3 is a screenshot of a bug.

## What it was shown to catch

Written by breaking things, not by reasoning about them. Each of these was
applied to a copy of the tree and the gate run against it:

| breakage | what fails |
|---|---|
| `NamespaceResolver` returns nothing | `csharp.first_party: 3 -> 0`, `unclassified: 1 -> 4` |
| Python relative resolution returns `None` | `python.first_party_resolved: 3 -> 1`, **and** the first-party gate |
| `_EXPRESSION` forgets C#'s `?.` and `??` | `csharp/Web/Options.cs` withheld — the exact shape that purged eight files from a real index |
| the all-letters identifier rule is dropped | `Options.cs` **and** `config_values.py` withheld |
| a query string is read as one value again | `config_values.py` withheld |
| the diff stops naming the language, or the field, or the path | the formatter's own tests, in `test_language_fixture_diff.py` |
| the diff reports only the first moved field | the same — three numbers move when the C# resolver breaks, and reporting one sends someone chasing a symptom |

The first row is why the first-party gate is not enough on its own, and the
last three are why `config_values.py` is written the way it is. An earlier
version of it held only plausible-looking constants, and lowering the entropy
threshold from 3.6 to 2.9 did not withhold it — the test asserted the right
thing and the fixture could not make it fail. Every line in it now rests on a
different rejection rule.

## What this corpus cannot exercise yet

- **`declared_dependency` is structurally zero.** Nothing populates the
  declared-dependency set until a manifest reader lands, so every third-party
  import falls to `unclassified`. The bucket is measured so that the day it
  stops being zero is visible.
- **Chunk counts are not measured at all**, and cannot be until #93 is decided.
  They are not a function of the file bytes: the context header carries
  `# repo: name (branch)`, its token cost comes out of the chunk budget, and
  the same file therefore splits differently depending on the branch it is read
  from. Measured on `config_values.py` — one chunk on `main`, two on
  `feat/language-fixture-baselines`, one again on the detached HEAD a CI
  checkout produces. This gate recorded a chunk count before that was noticed,
  and passed CI only because the two environments disagreed in the same
  direction. A number that moves when you rename a branch is the definition of
  a gate that fails for innocent reasons.
- **Cross-unit isolation is not here.** A unit is the first path segment, and
  every fixture sits under its language's directory, so this corpus is five
  units of one language each. Two repositories declaring the same namespace is
  covered in `test_manifest.py` and `test_indexer.py`.
