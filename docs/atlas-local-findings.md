# MongoDB Atlas Local — infrastructure spike findings

This is a report, not a plan: no application code changed. The question was
whether a local MongoDB Atlas deployment on the labbox VM can stand in for
Atlas cloud while testing `$vectorSearch`, server-side `$rerank`, and Atlas
"automated embedding" — relevant here because `rerank/` already defines a
`Reranker` protocol with `NoopReranker` as the off switch, and a real local
`$rerank` would have been a candidate implementation.

## How it's actually running, and why not the "normal" way

The documented path is `atlas deployments setup --type local`, which drives a
Docker (or, per MongoDB's docs, Podman on RHEL) container for you. This VM
has no Docker installed and no passwordless `sudo`, so installing Docker
wasn't a quiet background step — and the Podman that *is* already installed
(4.9.3, Ubuntu 24.04) doesn't clear the version bar MongoDB documents for
that CLI path (Podman v5.0+, and even then the docs only call out RHEL as a
supported distro).

Rather than fight the `atlas` CLI's version/distro checks, we skipped it
entirely and pulled the container image it would have run anyway —
`mongodb/mongodb-atlas-local` — straight from Docker Hub with rootless
Podman, then ran it ourselves with a plain `podman run`.

**This is not a caveat on anything below.** The CLI's only job on that path is
to run this same OCI image; running it directly is the same deployment, not a
lesser one. Nothing here is qualified by having skipped the wizard.

## Findings

### `$vectorSearch` — works, fully offline

Created a `vectorSearch` index on a small test collection, queried it with a
raw vector, got correct cosine-similarity scores back. No network egress
involved. This works on both the `latest` and `preview` image tags.

### `$search` and `$rankFusion` — both work, offline

Added after the original spike, because the report verified `$vectorSearch`
and stopped there — and `$vectorSearch` alone is not what this project needs.
`MongoStore` runs **hybrid** retrieval: `$vectorSearch` over the dense vector
and `$search` over the text, fused with `$rankFusion`. Two of those three were
unverified, so "Atlas Local works" was a weaker claim than it looked.

On the `latest` tag, mongod 8.3.8:

- `createSearchIndex` with a Lucene `mappings` definition succeeded, and the
  index reached `READY`.
- `$search` returned the expected document.
- `$rankFusion` returned rows. (It is a core server stage since 8.1 and works
  even on a plain community `mongod`, which has no mongot at all — so it was
  never in doubt, but it is now measured rather than assumed.)

**Consequence:** Atlas Local can back `MongoStore` and run the cross-backend
contract suite offline, with no Atlas Flex account and no network. That is a
larger result than the spike set out to establish — the Mongo half of the
storage tests stops depending on a hosted tier.

### `$rerank` — not available locally, on any tag

```
MongoServerError: Unrecognized pipeline stage name: '$rerank'
```

Tested on both `latest` and `preview`, both running mongod 8.3.8 — which
clears the 8.3+ version bar MongoDB's docs list for `$rerank`. The stage
isn't disabled behind a flag; the parser doesn't know it exists. It isn't in
the self-managed/local server binary at all. **`$rerank` is Atlas-cloud-only
today — there's no local server-side path to it**, local or Community
Edition, on either image tag. Any reranking here has to happen in
application code (calling Voyage's rerank API directly, or another
reranker), which is exactly what `rerank/`'s `Reranker` protocol already
allows for.

### Automated embedding (`autoEmbed`) — works locally, with two conditions

This took the most digging, and both conditions mattered:

1. **Image tag**: the `latest` tag's `mongot` never attempts a Voyage call at
   all — `createSearchIndex` with an `autoEmbed` field fails immediately
   (`supported models are: []`), no retries, no logs. Only the **`preview`**
   tag's `mongot` implements the automated-embedding path — matching
   MongoDB's own docs, which mark this feature Preview.

2. **Key type**: a personal/dashboard Voyage AI key (`pa-...` prefix), even
   though it authenticates fine for a direct call to Voyage's own
   `/v1/embeddings` endpoint, gets a hard `403 Forbidden` from
   `mongot`'s embedding client specifically — confirmed by watching
   `mongot`'s log retry the call roughly a dozen times, each `403`, before we
   stopped it. A key generated **through the Atlas UI** (`al-...` prefix)
   worked immediately: the index went `BUILDING → INITIAL_SYNC → STEADY`,
   all 3 test documents were embedded, and a `$vectorSearch` query using
   plain query *text* (not a precomputed vector) came back correctly ranked.
   Voyage's own dashboard has no visible scoping/permissions to explain the
   403 — the two key types are evidently authorized differently on Voyage's
   backend regardless.

Even working, this is not an offline feature: every embed and every
text-based query is a live HTTPS call from this VM out to Voyage's API.
`$vectorSearch` with a precomputed vector has no such dependency;
`autoEmbed` always does.

## Scorecard

| Feature | Status locally |
|---|---|
| `$vectorSearch` | Works, fully offline, either image tag |
| `$search` (Lucene) | Works, fully offline; index reaches `READY` |
| `$rankFusion` | Works; a core server stage, not a mongot one |
| `$rerank` | Not implemented in the local server, any tag — Atlas-cloud-only |
| Automated embedding (`autoEmbed`) | Works, but only on the `preview` tag, only with an Atlas-issued Voyage key, and only with internet egress to Voyage on every call |

## What this settles for the project

`rerank/`'s `ServerReranker` path, behind the `DatabaseReranking` flag, is
**unreachable in every environment this project currently has**: Atlas Flex
cannot reach mongod 8.3, and the local server has no `$rerank` at any version.
It can only ever run against a paid Atlas tier at 8.3+.

Two things follow.

The Qdrant-versus-Mongo reranking comparison is **settled rather than
pending**. Qdrant scored 0.875 recall / 0.679 MRR reranked against Atlas's
0.812 / 0.661, with Voyage doing the reranking on both sides. There is no
outstanding "but Mongo's native reranker might win" — it is not obtainable.

And the only reachable code in `atlas_rerank.py` is its **error** path:
`_translated()`, which turns Atlas's generic "`$rerank` is not allowed or the
syntax is incorrect" into a message naming the 8.3 requirement and the project
toggle. That is what a user actually hits, so that is what has to be right.

## Current state of the machine

Two containers exist, both **stopped** (not removed) as of 2026-09-04.
`atlas-local` was started once more to run the `$search`/`$rankFusion`
checks above and stopped again; its data is unchanged apart from a scratch
database that was dropped.

| Container | Image | Port | Notes |
|---|---|---|---|
| `atlas-local` | `mongodb/mongodb-atlas-local:latest` | 27017 | mongod 8.3.8; no working `autoEmbed` |
| `atlas-local-preview` | `mongodb/mongodb-atlas-local:preview` | 27018 | mongod 8.3.8; has a working `autoembed_idx` on `spike.docs` from this spike |

Both images are still cached locally, so `podman start atlas-local` (or
`atlas-local-preview`) brings either back up in a few seconds with all data
intact — no re-pull needed. Connect with
`podman exec -it <name> mongosh` or, from the host,
`mongosh "mongodb://localhost:<port>/?directConnection=true"` once `mongosh`
is installed there (it isn't yet — only inside the containers).

The Atlas-issued Voyage key used for the working `autoEmbed` test lives in
the project's `.env` (repo root, not `config/`) as
`MONGO_ATLAS_VOYAGE_API_KEY`, separate from the pre-existing `VOYAGE_API_KEY`
(the personal key that gets the 403). Neither is wired into any application
code — this was infrastructure testing only.
