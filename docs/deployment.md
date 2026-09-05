# Deployment and operations

`README.md` is a developer quickstart: clone, install, run the tests. This
document is about running it somewhere that is not the machine it was written
on, which is the intended shape — the indexer wants to sit next to the source
it indexes, and that is rarely a laptop.

Everything measured here was measured on the VM this was built on: 8 emulated
QEMU cores, no GPU, dense embeddings via the Voyage API.

---

## 1. What must be co-located with what

The single most useful thing to understand before deploying, because it is
implicit in the code and nowhere else.

| Component | Needs the source files | Needs Qdrant | Needs the manifest | Needs API credentials |
|---|---|---|---|---|
| `index` | **yes** — it reads and chunks them | yes | **yes** | yes, to embed |
| `search` / `eval` | only for staleness checks | yes | no | yes, to embed the query |
| `serve` (MCP) | only for staleness checks | yes | no | yes, to embed the query |
| `status` | no | yes | **yes** | no |

Three consequences worth stating plainly:

**The indexer is the only thing pinned to the source.** It runs where the code
is. Nothing else does.

**Qdrant can live anywhere.** `QDRANT_URL` is just a URL, so "install Qdrant
here" and "run the indexer there" are independent decisions.

**The MCP server does not need the source, with one exception.** Every hit
carries its `source_text` in the payload, so a result is self-contained and
rendering it reads no files. The exception is the staleness check, which reads
the indexed file to confirm the chunk's text is still there — and on a box
without the source, *every* hit comes back flagged stale. A flag on everything
carries no signal, so turn it off there:

```yaml
search:
  check_staleness: false
```

The honest trade: a result may then be out of date without saying so. That is
why it is not the default.

---

## 2. Choosing a storage mode

| Mode | When | Limits |
|---|---|---|
| `embedded` | One process at a time: batch indexing, eval, CI | Single-process lock. **Payload indexes are ignored**, so `doc_type` filtering scans. Python implementation, not Rust |
| `server`, local | Indexer and MCP server on one box | Needs a Qdrant binary or Docker |
| `server`, remote | Indexer near the source, Qdrant central | Network latency per query. Set `QDRANT_API_KEY` — Qdrant has **no authentication by default** |

**Embedded mode is not a toy, and it needs no server at all.** Measured: a full
16-case eval with reranking runs against embedded mode in **21.8 s**. Batch
indexing and CI are entirely happy there.

**Two things force server mode.** The MCP server holds the index open for the
whole session, so embedded mode's exclusive lock would block every reindex
while an agent is connected. And embedded mode ignores payload indexes
completely, which is what `doc_type` filtering depends on.

> **Security note.** Qdrant ships with no authentication. Anything that can
> reach port 6333 can read every chunk, and the payload contains `source_text`
> — the actual contents of every indexed file. Bind to `127.0.0.1` unless you
> have set `QDRANT_API_KEY` *and* firewalled the port. To reach a
> loopback-bound instance from another machine, use an SSH tunnel rather than
> opening it up:
>
> ```bash
> ssh -N -L 6333:127.0.0.1:6333 user@indexer-host
> ```
>
> The tunnel terminates inside the remote box and connects to `127.0.0.1` from
> there, so Qdrant still sees a local connection. Loopback binding and remote
> access are not in conflict.

---

## 3. Installing Qdrant

### Linux — binary under a systemd *user* service

This is what the VM runs, and the choice is deliberate. There is no passwordless
sudo on it, and joining the `docker` group is effectively passwordless root: the
daemon runs as root and will bind-mount any path you name. That is a large
standing privilege grant to avoid one download. A user service needs no root at
all.

```bash
# 1. Fetch the binary. Check the release page for the current version.
VERSION=1.19.0
mkdir -p ~/.local/opt/qdrant ~/.local/share/qdrant/storage
curl -sSL -o /tmp/qdrant.tar.gz \
  "https://github.com/qdrant/qdrant/releases/download/v${VERSION}/qdrant-x86_64-unknown-linux-gnu.tar.gz"
tar -xzf /tmp/qdrant.tar.gz -C ~/.local/opt/qdrant

# 2. The web dashboard ships separately -- the binary release has no web assets
#    at all, so /dashboard returns 404 without this step.
curl -sSL -o /tmp/ui.zip \
  https://github.com/qdrant/qdrant-web-ui/releases/download/v0.2.16/dist-qdrant.zip
unzip -q /tmp/ui.zip -d /tmp/ui
mv /tmp/ui/dist ~/.local/share/qdrant/static
```

Match the web UI version to the Qdrant release by date — v0.2.16 and Qdrant
1.19.0 were published a day apart.

`~/.config/systemd/user/qdrant.service`:

```ini
[Unit]
Description=Qdrant vector database
After=network.target

[Service]
Type=simple
ExecStart=%h/.local/opt/qdrant/qdrant
WorkingDirectory=%h/.local/share/qdrant
Environment=QDRANT__STORAGE__STORAGE_PATH=%h/.local/share/qdrant/storage
Environment=QDRANT__STORAGE__SNAPSHOTS_PATH=%h/.local/share/qdrant/snapshots
Environment=QDRANT__SERVICE__STATIC_CONTENT_DIR=%h/.local/share/qdrant/static
# Bound to localhost: this holds the contents of every indexed file.
Environment=QDRANT__SERVICE__HOST=127.0.0.1
Environment=QDRANT__SERVICE__HTTP_PORT=6333
Environment=QDRANT__SERVICE__GRPC_PORT=6334
Restart=on-failure
RestartSec=5

[Install]
WantedBy=default.target
```

```bash
systemctl --user daemon-reload
systemctl --user enable --now qdrant
curl -s http://127.0.0.1:6333/ | head -c 80    # expect a version banner
```

**The one thing a user service cannot do by itself is survive logout.** It stops
when your last session ends, which is wrong for a machine nobody is logged into.
Fix it once, and this step *does* need root:

```bash
sudo loginctl enable-linger "$USER"
```

Without that, a reboot leaves Qdrant down until someone logs in — a failure that
looks like data loss and is not.

### Linux — Docker

Simpler where the privilege trade is acceptable, and it bundles the dashboard:

```bash
docker run -d --name qdrant --restart unless-stopped \
  -p 127.0.0.1:6333:6333 -p 127.0.0.1:6334:6334 \
  -v "$HOME/.local/share/qdrant/storage:/qdrant/storage" \
  qdrant/qdrant:v1.19.0
```

Keep the `127.0.0.1:` prefixes on `-p`. Without them Docker publishes on all
interfaces and inserts its own iptables rules, which bypass a `ufw` deny.

### Windows

Qdrant does publish a native Windows binary —
`qdrant-x86_64-pc-windows-msvc.zip` on the release page — but no installer and
no service wrapper, so nothing keeps it running across a reboot on its own.
Three routes, in the order worth trying:

- **Docker Desktop** — the command above, unchanged, with `%USERPROFILE%` for
  `$HOME`. `--restart unless-stopped` gives you the restart behaviour the bare
  binary lacks. The simplest option.
- **WSL2** — install the binary inside the distro exactly as in the Linux
  section. Two caveats. Storage must live on the Linux filesystem
  (`~/.local/share`), never under `/mnt/c`: WSL reaches Windows files over 9P,
  which is slow and does not give Qdrant the locking semantics it expects. And
  a WSL service does not start with Windows unless the distro is launched, so
  this is for a machine someone logs into.
- **The native binary** — unzip it and run `qdrant.exe`, configured with the
  same `QDRANT__*` environment variables. For a persistent service you need a
  wrapper such as NSSM or a scheduled task with a boot trigger; Windows has no
  equivalent of a systemd user service.

The indexer and the MCP server both run natively on Windows either way. Only
Qdrant benefits from the container.

---

## 4. Running the MCP server

**It is not a service, and there is no daemon to run.** `serve` speaks MCP over
stdio: the client starts the process, talks to it down a pipe, and the process
exits when the session ends. Nothing to enable, no port, nothing to restart.

Register it in `.mcp.json` at a project root, or with `claude mcp add`:

```json
{
  "mcpServers": {
    "workspace-indexer": {
      "command": "/opt/workspace-indexer/.venv/bin/workspace-indexer",
      "args": ["serve", "--config", "/opt/workspace-indexer/config/workspace.yaml"],
      "env": {
        "QDRANT_MODE": "server",
        "QDRANT_URL": "http://127.0.0.1:6333",
        "EMBEDDING_MODEL": "voyageai:voyage-code-4",
        "EMBEDDING_DIMENSIONS": "1024",
        "VOYAGE_API_KEY": "..."
      }
    }
  }
}
```

**Everything absolute, and the environment explicit.** The client launches this
from *its* working directory, not from the repository. A relative `--config` is
not found, `.env` is never read, and the defaults then resolve to an embedded
Qdrant at a `data/qdrant` that does not exist. That combination starts up
perfectly and serves an empty index, answering every query with "nothing
found" — which an agent believes.

`serve` now checks the collection at startup and refuses rather than serve
nothing:

```
the collection for voyageai_voyage-code-4_2048 is empty (embedded qdrant at
/some/other/cwd/data/qdrant), so every query would return nothing.
```

Note the dimensions in that message. `EMBEDDING_DIMENSIONS` defaults to 2048,
so an unreachable `.env` also changes which collection is addressed — the error
names it for exactly that reason.

**Logs go to stderr and to the rolling JSONL file, never stdout.** stdout is
the protocol channel; a single stray line on it makes the stream unparseable
and every call fails. If you add logging, keep it off stdout.

**Each command writes its own file**, with its name spliced in before the
suffix: `workspace-indexer-serve.jsonl`, `-index.jsonl`, `-watch.jsonl`. Two
processes sharing one rotating file cannot both roll it over — measured on
Linux with three writers through the stdlib handler, 346 of 360 lines survived
— and on Windows the rename fails outright with `WinError 32`. The file sink
locks around appends and rollover for the same reason: every editor session
running an MCP server is another `serve` process on the same file. Sessions
stay tellable apart by the `run_id` on every line.

---

## 5. Keeping the index current

Two options, and the watcher is the better one where it works.

### `watch` — event-driven

```bash
workspace-indexer watch
```

Watches every configured root and reindexes the root a change lands in. A
trigger, not a second indexing path: each change ends up in the same
`index --root <label>` run the CLI performs, so it cannot develop its own
opinion about what counts as changed.

It prints the mode it chose per root before it starts:

```
  src: native
Watching. Ctrl-C to stop.
```

**`native` versus `poll` is not a preference.** On WSL and similar, an inotify
watch on a path under `/mnt/c` *succeeds* and then never delivers an event —
the 9P protocol WSL uses to reach Windows files carries no change
notification, and inotify is a service of the kernel's own filesystem layer.
The same is true of NFS, CIFS and most FUSE mounts. Nothing errors; the watcher
just sits there looking healthy and indexing nothing.

So each root is checked against `/proc/mounts` and polled when its filesystem
cannot deliver events. Unknown filesystems are polled too: guessing "native"
wrong costs a watcher that silently never fires, and guessing "poll" wrong only
costs some CPU. Override with `watch.mode: native|poll` when you know better.

**Watch out for the watch limit.** `/proc/sys/fs/inotify/max_user_watches` caps
watches per *user* — often 65536, sometimes 8192 — and it is shared with every
editor and language server already running. A workspace containing
`node_modules` will exhaust it, and the failure reads
`OSError: [Errno 28] No space left on device`, which has nothing to do with
disk space.

**The watch is placed on exactly the directories the index looks in**, and
nowhere else. inotify watches directories, so the cost is per directory either
way — watching a root recursively just meant watching every directory under it,
including the ones the index ignores. Measured on a 1,154-file workspace:
**1,517 directories watched before, 215 after**. `watch.scoped` logs the count
at startup and the budget check counts the real scope.

`node_modules`, `.venv`, `target` and friends are skipped whatever your index
excludes say, because this is about the watch budget rather than index quality.
Index one of those trees without excluding it and the index will hold it while
the watcher will not see it change — accepted, because exhausting the limit
stops the watcher working at all.

This is the other reason the ignore rules matter. They are not only about index
quality — they are what keeps the watcher functional, and now they are also
what keeps it from walking into a path it cannot read.

Editing `workspace.yaml` reloads it live. Settings and ignore rules take effect
immediately; **a newly added root still needs `watch` restarted**, because the
underlying watch cannot be handed a new path mid-flight. The watcher says so
when it reloads rather than leaving you to find out.

### A timer — for a machine nobody is logged into

`watch` is a foreground process. Where you want unattended reindexing, or where
every root is polled anyway, a timer is simpler and survives reboots.

`~/.config/systemd/user/workspace-indexer.service`:

```ini
[Unit]
Description=Reindex the workspace
After=qdrant.service

[Service]
Type=oneshot
WorkingDirectory=/opt/workspace-indexer
ExecStart=/opt/workspace-indexer/.venv/bin/workspace-indexer index
```

`~/.config/systemd/user/workspace-indexer.timer`:

```ini
[Unit]
Description=Reindex every 15 minutes

[Timer]
OnBootSec=2min
OnUnitActiveSec=15min
# Without this, a laptop that was asleep runs every missed interval at once.
Persistent=true

[Install]
WantedBy=timers.target
```

```bash
systemctl --user enable --now workspace-indexer.timer
systemctl --user list-timers workspace-indexer.timer
```

`Type=oneshot` because the command exits. `After=qdrant.service` orders startup
but does not wait for Qdrant to be *ready*; a run that starts a second too
early fails and the next one succeeds, which is why `Restart` is absent — a
oneshot that retries forever hides a real outage.

A 15-minute interval is cheap because unchanged files cost one `stat()` each.
See the timings below.

---

## 6. Per-machine configuration

The `workspace.yaml` / `.env` split is *what to index* versus *how to index*.
The deployment consequence:

**`workspace.yaml` is committable but still machine-specific**, because `roots`
are paths. Two ways to handle it, and the first is better:

- Use `~`-relative paths (`~/src`), which are expanded at load time. A single
  committed file then works on any box with the same layout under `$HOME`.
- Or keep a per-host file and select it: `workspace-indexer index --config
  config/hosts/build-01.yaml`.

**`.env` is per-machine and holds credentials.** Never committed. Real
environment variables win over it, which is what makes the `.mcp.json` `env`
block above work.

**`.mcp.json` holds API tokens and must never be committed.** It is gitignored
now, after a `git add -A` swept one into a commit and GitHub push protection
rejected the push — the token had to be rotated. Check `git status --porcelain`
before staging, and prefer `git add <paths>` to `-A`.

---

## 7. Recovery

**`data/` is entirely derived state.** Qdrant collections plus the SQLite
manifest. The backup story is *do not back it up, re-index*.

What that costs, measured on 1,028 files / 10,788 chunks:

| | Time | Notes |
|---|---|---|
| Full index, Voyage API | **72 s** | 3.19M tokens embedded |
| Full index, local CPU model | **~2h40m** | `bge-small` ONNX, 8 emulated cores, no GPU |
| Incremental, nothing changed | **5.8 s** | 1,050 files, one `stat()` each |
| Full 16-case eval with reranking | **21.8 s** | embedded mode, no server needed |

The gap between rows 1 and 3 is the entire argument for the manifest, and the
gap between rows 1 and 2 is the argument for an API over local inference on a
box without a GPU.

Cost reporting tells three answers apart, because they are three different
answers:

| Display | Meaning |
|---|---|
| `$0.1234` | The provider reported this price |
| `~$0.1234` | Estimated from `EMBEDDING_PRICE_PER_MTOK`, because the provider did not |
| `unpriced (7)` | Neither the provider nor config could price 7 requests |

`genai-prices`, which pydantic-ai delegates to, has no entry for
`voyage-code-4`, so set `EMBEDDING_PRICE_PER_MTOK=0.12` (its rate at the time
of writing) to get a real estimate. Without it, runs are recorded as *unknown*
rather than as free — `$0.0000` for an unpriced API is the wrong answer in the
expensive direction.

`status` also reports cumulative tokens against `EMBEDDING_FREE_TIER_TOKENS`.
Read it as a floor, not a measurement: it counts what this manifest embedded,
while the allowance belongs to the account and is drawn down by everything
using the key.

**Recovering by hand:**

```bash
rm -rf data/qdrant data/manifest.sqlite3    # embedded mode
workspace-indexer index --force
```

`--force` re-embeds everything, ignoring the manifest. Without it, a run against
a *fresh* store and a *surviving* manifest skips nearly every file, because the
manifest records which embedding space a file is complete for and not which
store holds it. `index` warns when it sees that mismatch, and `status` reports
it as `inconsistent`.

**If you switched from embedded to server mode**, the old `data/qdrant` is a
stale duplicate of the whole index — 137 MB here — and can be deleted.

---

## 8. Disk footprint

| | Size |
|---|---|
| Qdrant storage, 10,788 chunks at 1024 dimensions | 134 MB |
| SQLite manifest | 15 MB |
| Qdrant binary | 85 MB |
| Qdrant web UI assets | 18 MB |

Roughly 12 KB per chunk of vector storage. At 2048 dimensions expect double the
vector portion; `reproject` derives a narrower collection from vectors already
paid for, with no re-embedding.

---

## 9. A worked split deployment

The shape the project was designed for: source on one box, Qdrant central,
agent sessions on a third.

```
┌──── build-01 ────────────┐      ┌──── vectors-01 ─────────┐
│ /srv/src        (source) │      │ qdrant :6333            │
│ workspace-indexer index  │─────▶│ 134 MB storage          │
│ data/manifest.sqlite3    │      │                         │
│ timer, every 15 min      │      └─────────────────────────┘
└──────────────────────────┘                  ▲
                                              │ QDRANT_URL
                                   ┌──── laptop ──────────────┐
                                   │ Claude Code              │
                                   │ serve  (stdio, no source)│
                                   │ check_staleness: false   │
                                   └──────────────────────────┘
```

**On `build-01`** — the only box that needs the source:

```bash
# .env
QDRANT_MODE=server
QDRANT_URL=http://vectors-01:6333
QDRANT_API_KEY=...
EMBEDDING_MODEL=voyageai:voyage-code-4
EMBEDDING_DIMENSIONS=1024
VOYAGE_API_KEY=...
STATE_DB=/srv/workspace-indexer/data/manifest.sqlite3
```

The manifest stays here, with the indexer. It is the indexer's bookkeeping and
nothing else reads it.

**On `vectors-01`** — Qdrant with an API key, since it is no longer reachable
only over loopback:

```ini
Environment=QDRANT__SERVICE__HOST=0.0.0.0
Environment=QDRANT__SERVICE__API_KEY=...
```

Firewall 6333 to `build-01` and the client hosts. `0.0.0.0` without both the
key and the firewall publishes your source code on an unauthenticated endpoint.

**On the laptop** — the `.mcp.json` from §4, with `QDRANT_URL` pointing at
`vectors-01`, plus:

```yaml
search:
  check_staleness: false
```

No source here, so the check would flag every hit. This is the one setting a
split deployment must change.

**What breaks if you get it wrong:**

| Symptom | Cause |
|---|---|
| `serve` refuses to start, "collection is empty" | `QDRANT_URL` or `EMBEDDING_DIMENSIONS` disagree with what the indexer wrote |
| Every result flagged stale | `check_staleness` left on, on a box without the source |
| `index` reports 0 changed on a fresh store | Manifest survived a store rebuild — run `--force` |
| `status` says `inconsistent` | Same cause, caught before the run |
| `doc_type` filters slow | Embedded mode, which ignores payload indexes |
