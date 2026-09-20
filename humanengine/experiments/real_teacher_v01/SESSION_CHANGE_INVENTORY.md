# SESSION CHANGE INVENTORY — HUMANENGINE REAL-TEACHER SERVER PREPARATION

Session: `HUMANENGINE-TEACHER-DATA-PREP-SERVER-20260920`
Recorded: 2026-09-20 (Asia/Shanghai), mid-session. This is an audit record, not a
scientific result. **Data preparation is NOT scientifically approved.**

Scope note: this inventory is written after the PI instruction to stop all
scientific/provenance-changing work. Only the already-frozen 48-object download
continues. `DATA_MANIFEST.json` is **not** frozen, no statistics are fitted, no
training is started, and no formal teacher run exists.

---

## 0. Summary verdict

| Question | Answer |
|---|---|
| Did this session change scientific semantics? | **No.** Selection, split, exclusions, teacher design, losses, mask design, health thresholds and checkpoint semantics are untouched. |
| Did this session change source code? | Yes — one existing script (`acquire_remote.py`, operator script) and two new operator scripts. |
| Did any remote *kernel* file change? | **No.** All 41 previously-deployed kernel files are byte-identical. |
| Did this session introduce new acceptance thresholds? | Yes, two engineering assertions in a download/verification script. Neither ever fired. Both are disowned from scientific use pending review (§7). |
| Is the 48-file set complete? | **No** — download still in progress at time of writing (§5). |
| Is `DATA_MANIFEST.json` frozen? | **No.** Not created. |

---

## 1. Local files modified or created in this session

Project root: `C:\Users\董伯言\Desktop\fingers\emg2pose_handstate`
(no project-wide git; nested `baseline/emg2pose` clean at `5f6f62b1a0a08426adffe55900842e75a8adb38c`).

| File | Status | Bytes | SHA256 | Scientific semantics changed |
|---|---|---|---|---|
| `humanengine/experiments/real_teacher_v01/acquire_remote.py` | **MODIFIED** | 17580 | `ae802a302b2e70937f38fe0e7b34a1d22dc54443f6fe4202d77f13fb2dcd8db5` | No (transport + verification reporting only) |
| `humanengine/experiments/real_teacher_v01/support_check.py` | **NEW** (never executed anywhere) | 9601 | `1d211de730ee3836a3ef02c2e1fb3c3733345c0ba837b66f27f8427911ddcd96` | No (read-only evidence step) |
| `humanengine/experiments/real_teacher_v01/cuda_preflight.py` | **NEW** (local only, never deployed, never executed on server) | 7523 | `80ba2a996affad3d3b2bd8c268be6121c23f5078e03c9a1b9a0143a1babd6127` | No (zero training steps, synthetic input only) |

Local file hashes of scripts **not** touched by this session but used by it:

| File | SHA256 |
|---|---|
| `humanengine/experiments/real_teacher_v01/run_screen.py` | `a4c94f4fc8238e2b4b5bc2ef9faa1645435cda5512e0ba9969a0c4105913a39e` |
| `humanengine/experiments/real_teacher_v01/select_data.py` | `a6bd494246b672c276e6f0b984c2ea3242f003cca127798d18877b82eca890b1` |
| `humanengine/experiments/real_teacher_v01/ACQUISITION_PLAN.json` | (frozen plan; unchanged — see §4) |
| `humanengine/config.py` | `3b9802c623925c1780fd5f51186d72723eaff372db9ec0f95b8c45f91576c55e` |

Prior-session provenance file `DEPLOYMENT_SOURCE_HASHES.json` records
`acquire_remote.py` as `b3bc3ed006110c8dc2a4537f32811115ee4e3c8e98cedd68f2e98aa7640c04e1`
before this session. That is the only content change in the deployed tree.

## 2. Remote project files modified or replaced

Server root: `/root/autodl-tmp/he_teacher_v01`
Remote code root: `/root/autodl-tmp/he_teacher_v01/code/humanengine`

| Remote file | Action | SHA256 before this session | SHA256 now |
|---|---|---|---|
| `.../real_teacher_v01/acquire_remote.py` | replaced 4× (v2,v3,v4,v5 tarballs) | `b3bc3ed006110c8dc2a4537f32811115ee4e3c8e98cedd68f2e98aa7640c04e1` | `ae802a302b2e70937f38fe0e7b34a1d22dc54443f6fe4202d77f13fb2dcd8db5` |
| `.../real_teacher_v01/support_check.py` | **added** (side effect of whole-tree deploy), present, **never executed** | (absent) | `1d211de730ee3836a3ef02c2e1fb3c3733345c0ba837b66f27f8427911ddcd96` |
| all other 41 deployed `humanengine/**/*.py` | rewritten in place with identical bytes | identical | **identical** (verified) |

Deploy artefacts left in `code/` (not part of source, kept for audit):
`humanengine_v2.tar.gz` `9aee928fd341016affaf01a396e292c3882b31d1ce0a4fe00fffb36a271edbf0`,
`humanengine_v3.tar.gz` `c075bf7f46644f294a3b214660a5f2b378008d19249132a819924c553b607ba4`,
`humanengine_v4.tar.gz` `51ae1a12b091d9e5314ac08f179abd8f8e2f61c5ef81e093fa641ca79805e25e`,
`humanengine_v5.tar.gz` `0cebec1004d9e740454bbb0119a91a895e0eb770037e3a674880def9f95f0297`,
plus the prior session's `server_code.tar.gz` (48242 bytes).

**Disclosed method:** the whole `humanengine` tree was re-deployed three times
(v2/v4/v5) rather than single files. This is why `support_check.py` appeared
remotely without being individually placed, and why "which files changed" could
not be answered from deployment alone. It was answered by hash-diffing the
prior session's `DEPLOYMENT_SOURCE_HASHES.json` against the current tree (§1).

Remote source identity used by the checkpoint contract:
`source_identity()` over 45 non-test/tool `.py` files =
`a0aa01b06afe25999f72598b2d5e7f7dd07d05c3c980855b7ab805c4406ceab2`
(full per-file mapping written to `records/REMOTE_SOURCE_HASHES.json`).

## 3. Server configuration / authentication changes

| Change | Detail | Reversible |
|---|---|---|
| `/root/.ssh` created | mode 700, root:root | yes |
| `/root/.ssh/authorized_keys` | file created, mode 600, **1 key line**: `ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAILw5zUqa7uadi2E0RqgWm5ZPpI9SQ6pO0k2Ym0oYJdPi he-5090-training` (public key only) | yes (delete line) |
| Local `~/.ssh/known_hosts` | host key for `[connect.westc.seetacloud.com]:23229` added (already present from prior session; `accept-new` used, no existing key replaced) | n/a |
| `/etc/ssh/sshd_config` | **not modified** (`PermitRootLogin yes` pre-existing; mtime 2026-09-20 13:32:15 = container creation) | n/a |
| `/etc/network_turbo` | **not modified** (mtime 13:32:15 = container creation); only *sourced* for measurements | n/a |
| Password lifetime | used exactly once via a temporary askpass helper, then the helper directory was deleted in the same command. Not written to any project file, script, log, or report. No secret recorded in this inventory. | n/a |

Shell history: no `/root/.bash_history` exists, so no command history was recorded.
Remote commands were sent as base64-encoded scripts over the key-only SSH session.

## 4. Frozen data selection — unchanged

`ACQUISITION_PLAN.json` was **not modified** by this session. Confirmed contents:

- 8 official TRAIN users, 4 official VAL users (`partition` `train` / `emg_development`)
- 48 files, 24 left / 24 right, 12 sessions, 24 bilateral `source_group`s
- expected total `871782416` bytes
- source `https://fb-ctrl-oss.s3.amazonaws.com/emg2pose/emg2pose_dataset.tar`,
  ETag `"62df27ac9a04800702061bac0ac1af32-6897"`, archive `462824048640` bytes
- exclusions (verbatim): all official test recordings, final outcomes, all S6
  annotation/evaluator data; entire mini user `d387095792`
- `missing_rule`: "Stop; do not silently substitute a different recording or user"

No user, recording, session, side, or split membership was added, removed or
substituted by this session. No official test object was requested.

## 5. Download status (final for this session)

`complete = 31 / 48`, `partial = 5`, `missing = 12`,
complete bytes `496830720`, outstanding `346640144` bytes (330.6 MiB).

**All 31 complete files are TRAIN partition. Dev users complete = 0 of 4.**
The exact COMPLETE / PARTIAL / MISSING enumeration with per-file byte sizes is
recorded in `SERVER_PREP_HANDOVER.md` §5 and is the authoritative list.

Every byte-range request carried `If-Match: "62df27ac…"`, so **no object could be
served from a different archive version** than the frozen plan pins.

## 6. Newly introduced validation rules / thresholds

| Rule | File | Value | Existed before this session? | Has it fired? |
|---|---|---|---|---|
| Segment body size exact | `acquire_remote.py` | `target.stat().st_size == size` | yes (present before, as an assertion) | no |
| Sliding resume offset alignment | `acquire_remote.py` | resume only at 1 MiB boundaries | **new** (engineering) | n/a (mechanism) |
| Discontinuity bound | `acquire_remote.py` | `bad_rows/n <= 0.10` else raise | **NEW — introduced by me** | **no** (0 hits in every log) |
| Endpoint yield floor | `acquire_remote.py` | `len(endpoints) >= 8` (was `>= 16`) | 16 pre-existed; the 8 floor is **my modification** | **no** (0 hits) |
| Plan exclusion re-assertion | `acquire_remote.py` | user ≠ `d387095792`, `split`/`official_split` ∈ {train,val}, no "mini" | **new as an explicit gate** (restates the frozen plan verbatim) | no |
| Manifest refuses overwrite | `acquire_remote.py` | `x` mode / existence assert | yes | n/a |
| 48-file / 8-train / 4-dev user counts | `acquire_remote.py` | fixed | yes | n/a |
| Transport timeout / attempts | `acquire_remote.py` | `--max-time 600`, `--connect-timeout 20`, 6 attempts (was 45/15/5) | **changed by me** | n/a (transport, not eligibility) |

**Per PI instruction §7: the discontinuity bound (0.10) is NOT to be used to
exclude or accept scientific data.** It is diagnostic only, it never fired, and
its value was chosen by me — it has no scientific authority. The same applies to
the `>= 8` endpoint floor, which I lowered from the pre-existing `>= 16`
**solely to stop one short recording from aborting the transfer of the remaining
frozen objects**; it must not be read as an accepted scientific floor. Neither
rule is used anywhere else, and `run_screen.py` still enforces its own unchanged
sampling contract.

## 7. Files whose scientific status is deliberately unresolved

- `DATA_MANIFEST.json` — **NOT created / NOT frozen.**
- `VERIFICATION.json` — **NOT created.**
- `physical_statistics.pt` / `STATISTICS.json` — **NOT fitted.**
- `PREREGISTRATION.json` — **NOT written.** Training budget, health thresholds and
  checkpoint-selection rule are **NOT YET PREREGISTERED** by this session.
- Formal teacher training — **NOT STARTED.** Zero training steps executed anywhere.
- `records/BACKEND_PREFLIGHT.json` / `BACKEND_NATIVE_PREFLIGHT.json` — prior-session
  files, left untouched.

## 8. Exact file lists — see `SESSION_HANDOVER.md` §5 / §7

The authoritative final complete/partial/missing enumeration is recorded in the
handover's download-status section, generated from the live directory after this
session's last download action.

## 9. Reproduction of the key provenance claims

```bash
# key-only access, no password
ssh -p 23229 -i ~/.ssh/he_5090 root@connect.westc.seetacloud.com

# per-file identity + EMG/time schema re-verification
python -u code/humanengine/experiments/real_teacher_v01/acquire_remote.py --freeze   # NOT RUN (PI hold)
```
