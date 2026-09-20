#!/usr/bin/env python3
"""One-command acquisition + extraction + verification of the full emg2pose dataset.

Designed to be copied to a remote server with enough disk and run there. Nothing
in this file is local-machine specific except the default paths.

Why not ``curl`` + ``tar -xvf``?
--------------------------------
The official tar is ~431 GiB and its contents are ~431 GiB. The naive workflow
needs ~862 GiB free simultaneously and is not resumable in any useful way. This
tool instead uses a **bounded sliding window**:

    repeat until done:
        download the next window of the remote object directly to its final
        offset in the local tar file (HTTP Range, resumable)
        extract every complete tar member inside that window
        delete the consumed prefix of the tar

Peak disk usage is therefore ``extracted_so_far + window_size + headroom``
instead of ``tar + extracted``. With a default 32 GiB window that is roughly
``431 + 32`` GiB instead of ``862`` GiB. Every step is recorded in a state file,
so an interruption (or a reboot) resumes without re-downloading completed bytes
and without re-extracting completed files.

Phases
------
``plan``     report sizes, window plan, and whether the target volume suffices
``run``      do the acquisition (idempotent; safe to re-run any number of times)
``verify``   integrity-check the resulting tree (delegates to the verifier)
``all``      ``run`` then ``verify``

Usage
-----
    # 1. see the plan (touches nothing)
    python acquire_emg2pose_full.py plan --root /data/emg2pose

    # 2. do it (re-run as often as you like; it resumes)
    python acquire_emg2pose_full.py run --root /data/emg2pose

    # 3. verify every file
    python acquire_emg2pose_full.py verify --root /data/emg2pose
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import tarfile
import time
import urllib.error
import urllib.request
from pathlib import Path

TAR_URL = "https://fb-ctrl-oss.s3.amazonaws.com/emg2pose/emg2pose_dataset.tar"
TAR_BYTES = 462_824_048_640
META_URL = "https://fb-ctrl-oss.s3.amazonaws.com/emg2pose/emg2pose_metadata.csv"
META_BYTES = 5_674_462
META_MD5 = "c3f48fd51576fa7752d455f2751c8b99"
EXPECTED_FILES = 25_253
DEFAULT_WINDOW = 32 * 1024 ** 3      # 32 GiB
DEFAULT_HEADROOM = 8 * 1024 ** 3     # 8 GiB kept free at all times
SEG = 64 * 1024 ** 2                 # 64 MiB ranged requests
COPY_BUF = 8 * 1024 ** 2
TAR_BLOCK = 512                      # tar is a 512-byte-block format


def block_align_up(n: int) -> int:
    """Round up to the next 512-byte tar block boundary.

    Critical for pruning: a member's payload ends at ``offset_data + size``, but
    tar pads that payload to a 512-byte boundary. Pruning at the raw member end
    leaves the tar file starting mid-block, so the next parse reads a header
    split across the boundary and fails with ``bad checksum``.
    """
    return (n + TAR_BLOCK - 1) // TAR_BLOCK * TAR_BLOCK


def human(n: float) -> str:
    for u in ("B", "KiB", "MiB", "GiB", "TiB"):
        if abs(n) < 1024.0 or u == "TiB":
            return f"{int(n)} B" if u == "B" else f"{n:.2f} {u}"
        n /= 1024.0
    return f"{n:.2f} TiB"


def free_bytes(path: Path) -> int:
    p = path
    while not p.exists() and p.parent != p:
        p = p.parent
    return shutil.disk_usage(p).free


# --------------------------------------------------------------------------- #
# state
# --------------------------------------------------------------------------- #
class State:
    """Acquisition state. Written atomically after every durable step."""

    def __init__(self, path: Path):
        self.path = path
        self.d: dict = {}
        if path.exists():
            try:
                self.d = json.loads(path.read_text("utf-8"))
            except Exception:
                self.d = {}
        self.d.setdefault("remote_bytes_total", TAR_BYTES)
        self.d.setdefault("segments", {})       # offset(str) -> nbytes
        self.d.setdefault("extracted_through", 0)  # tar offset consumed
        self.d.setdefault("files_extracted", 0)
        self.d.setdefault("created", time.time())

    def save(self) -> None:
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.d, indent=1), "utf-8")
        os.replace(tmp, self.path)

    # convenience
    @property
    def downloaded(self) -> int:
        return sum(self.d["segments"].values())

    @property
    def consumed(self) -> int:
        return int(self.d["extracted_through"])


# --------------------------------------------------------------------------- #
# network
# --------------------------------------------------------------------------- #
def fetch(offset: int, length: int, retries: int = 10, url: str = TAR_URL,
          seg: int = SEG) -> bytes:
    end = offset + length - 1
    last = None
    for a in range(1, retries + 1):
        try:
            req = urllib.request.Request(
                url, headers={"Range": f"bytes={offset}-{end}"})
            with urllib.request.urlopen(req, timeout=180) as r:
                buf = r.read()
            if len(buf) != length:
                raise IOError(f"short read {len(buf)} != {length} @ {offset}")
            return buf
        except Exception as e:  # noqa: BLE001
            last = e
            wait = min(60, 2 ** a)
            print(f"    ! {offset} attempt {a}/{retries}: {type(e).__name__}: {e}"
                  f" (retry {wait}s)", flush=True)
            time.sleep(wait)
    raise SystemExit(f"segment at {offset} failed permanently: {last}")


def head_size(url: str, fallback: int) -> int:
    try:
        req = urllib.request.Request(url, method="HEAD")
        with urllib.request.urlopen(req, timeout=60) as r:
            return int(r.headers["Content-Length"])
    except Exception as e:  # noqa: BLE001
        print(f"  ! HEAD failed ({type(e).__name__}); using declared size "
              f"{fallback} B (UNVERIFIED)")
        return fallback


def local_size(path: Path) -> int:
    """Size of a local tar, for self-testing the orchestrator offline."""
    return path.stat().st_size


class Source:
    """Where the tar comes from: either the official URL or a local file.

    Supporting a local ``file://``-style source exists so the whole sliding-window
    orchestrator can be exercised end-to-end before anyone commits to a 431 GiB
    remote run. It is not an alternative acquisition route.
    """

    def __init__(self, local: Path | None, url: str, seg: int = SEG):
        self.local = local
        self.url = url
        self.seg = seg

    def size(self) -> int:
        if self.local:
            return local_size(self.local)
        return head_size(self.url, TAR_BYTES)

    def fetch(self, offset: int, length: int) -> bytes:
        if self.local:
            with open(self.local, "rb") as fh:
                fh.seek(offset)
                b = fh.read(length)
            if len(b) != length:
                raise IOError(f"short local read {len(b)} != {length} @ {offset}")
            return b
        return fetch(offset, length, url=self.url, seg=self.seg)


# --------------------------------------------------------------------------- #
# tar window extraction
# --------------------------------------------------------------------------- #
def _safe(dest: Path, name: str) -> Path:
    n = name.replace("\\", "/").lstrip("/")
    if n.startswith("../") or "/../" in n or n in ("..", "."):
        raise ValueError(f"unsafe member path: {name!r}")
    if os.path.isabs(name) or (len(n) > 1 and n[1] == ":"):
        raise ValueError(f"absolute member path: {name!r}")
    return dest / n


def extract_window(tar_path: Path, dest: Path, start_remote: int, end_remote: int,
                   base: int, headroom: int, state: State) -> int:
    """Extract members whose remote offset lies in ``[start_remote, end_remote)``.

    Coordinate spaces — the subtlety that makes this correct:

    * ``remote`` offset = byte position inside the original 431 GiB object.
    * ``local``  offset = byte position inside the on-disk tar file. Because a
      consumed prefix is pruned after every cycle, ``local = remote - base``
      where ``base`` is the remote offset the current tar file starts at.

    tar has no random access, so we always re-parse from the file's beginning;
    members ending at or before ``start_remote`` are skipped with a ``seek`` and
    their payload is never read.

    Returns the remote offset through which extraction is now complete.
    """
    dest.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    done_remote = start_remote
    n_new = 0
    n_skip = 0

    with open(tar_path, "rb") as fh:

        class Bounded:
            """Reads at most ``limit`` bytes from ``fh``, tracking its own offset.

            ``seek`` is exposed *and* keeps ``pos`` in sync because tarfile seeks
            internally (e.g. when skipping a member payload). An earlier version
            seeked ``fh`` directly, which left ``pos`` stale; the wrapper then
            mis-computed the remaining window and truncated a subsequent member
            read into a bogus end-of-archive. All access must go through here.
            """

            def __init__(self, f, limit):
                self.f, self.limit, self.pos = f, limit, 0

            def read(self, n=-1):
                if n is None or n < 0:
                    n = self.limit - self.pos
                n = min(n, self.limit - self.pos)
                if n <= 0:
                    return b""
                b = self.f.read(n)
                self.pos += len(b)
                return b

            def seek(self, off, whence=0):
                if whence == 0:
                    new = off
                elif whence == 1:
                    new = self.pos + off
                else:
                    raise ValueError("whence=2 not supported")
                self.f.seek(new)
                self.pos = new
                return new

            def tell(self):
                return self.pos

        bounded = Bounded(fh, end_remote - base)

        try:
            tf = tarfile.open(fileobj=bounded, mode="r|")
        except tarfile.ReadError as e:
            print(f"  ! window is smaller than one tar header, cannot parse yet "
                  f"({e}); will retry with a larger window.")
            return done_remote

        try:
            for m in tf:
                m_remote_start = m.offset_data + base
                m_remote_end = m_remote_start + m.size
                # reject anything that starts beyond this window
                if m_remote_start >= end_remote:
                    break

                if not m.isfile():
                    if m.isdir():
                        try:
                            _safe(dest, m.name).mkdir(parents=True, exist_ok=True)
                        except ValueError as e:
                            print(f"  ! {e}")
                    done_remote = max(done_remote, m_remote_end)
                    continue

                # already consumed in a previous cycle -> skip payload
                if m_remote_end <= start_remote:
                    bounded.seek(m.offset_data + m.size)
                    n_skip += 1
                    done_remote = max(done_remote, m_remote_end)
                    continue

                try:
                    out = _safe(dest, m.name)
                except ValueError as e:
                    print(f"  ! {e}")
                    done_remote = max(done_remote, m_remote_end)
                    continue

                if out.exists() and out.stat().st_size == m.size:
                    bounded.seek(m.offset_data + m.size)
                    n_skip += 1
                    done_remote = max(done_remote, m_remote_end)
                    continue

                fr = free_bytes(out.parent)
                if fr < m.size + headroom:
                    print(f"  ABORT: free {human(fr)} < member {human(m.size)} "
                          f"+ headroom. Tar intact; resume after freeing space.")
                    state.d["extracted_through"] = done_remote
                    state.save()
                    raise SystemExit(3)

                out.parent.mkdir(parents=True, exist_ok=True)
                src = tf.extractfile(m)
                if src is None:
                    print(f"  ! unreadable {m.name}")
                    done_remote = max(done_remote, m_remote_end)
                    continue
                part = out.with_suffix(out.suffix + ".part")
                try:
                    with open(part, "wb") as w:
                        while True:
                            b = src.read(COPY_BUF)
                            if not b:
                                break
                            w.write(b)
                    if part.stat().st_size != m.size:
                        raise IOError(f"size mismatch {m.name}")
                    os.replace(part, out)
                except Exception:
                    part.unlink(missing_ok=True)
                    raise
                n_new += 1
                done_remote = max(done_remote, m_remote_end)
                if n_new % 500 == 0:
                    print(f"    extracted {n_new} files up to "
                          f"{human(done_remote)} skipped={n_skip}", flush=True)
        except tarfile.ReadError as e:
            # A truncated window is the normal case until the whole object has
            # arrived; everything fully contained in the window is already out.
            print(f"  ! window parse stopped (truncated/edge): {e}")

    el = time.time() - t0
    # Block-align the completion point so the tar can be pruned cleanly.
    done_remote = min(block_align_up(done_remote), end_remote)
    print(f"  window extract: {n_new} new, {n_skip} skipped, "
          f"complete through remote offset {done_remote:,} ({human(done_remote)}) "
          f"in {el/60:.1f} min")
    state.d["extracted_through"] = max(state.consumed, done_remote)
    state.d["files_extracted"] += n_new
    state.save()
    return state.consumed


def prune_prefix(tar_path: Path, keep_from: int) -> None:
    """Drop tar[0:keep_from] (LOCAL offsets), shifting the remainder to offset 0.

    Uses a streaming forward copy rather than reading the tail into RAM. This is
    the price of a bounded sliding window, and it is far cheaper than holding the
    full 431 GiB tar alongside the extracted tree.
    """
    if keep_from <= 0:
        return
    size = tar_path.stat().st_size
    if keep_from >= size:
        tar_path.unlink()
        print("  prune: tar fully consumed and removed.")
        return
    tail = size - keep_from
    t0 = time.time()
    tmp = tar_path.with_suffix(".prune")
    with open(tar_path, "rb") as src, open(tmp, "wb") as dst:
        src.seek(keep_from)
        remaining = tail
        while remaining > 0:
            b = src.read(min(COPY_BUF, remaining))
            if not b:
                break
            dst.write(b)
            remaining -= len(b)
    os.replace(tmp, tar_path)
    print(f"  prune: dropped local prefix {human(keep_from)}, kept {human(tail)} "
          f"in {time.time()-t0:.0f}s")


# --------------------------------------------------------------------------- #
# phases
# --------------------------------------------------------------------------- #
def layout(root: Path) -> dict:
    return {
        "root": root,
        "raw": root / "raw",
        "tar": root / "raw" / "emg2pose_dataset.tar",
        "dataset": root / "dataset",
        "meta": root / "raw" / "emg2pose_metadata.csv",
        "state": root / "raw" / "acquire_state.json",
        "logs": root / "logs",
    }


def phase_plan(L: dict, window: int, src: "Source") -> int:
    total = src.size()
    free = free_bytes(L["root"])
    meta = head_size(META_URL, META_BYTES)
    print("=" * 76)
    print("emg2pose FULL ACQUISITION PLAN")
    print("=" * 76)
    print(f"source            : {src.local or src.url}")
    print(f"remote tar        : {total:,} B  ({human(total)})")
    print(f"metadata.csv      : {meta:,} B  ({human(meta)})  md5 {META_MD5}")
    print(f"expected files    : {EXPECTED_FILES:,}")
    print()
    print(f"root              : {L['root']}")
    print(f"volume free       : {human(free)}")
    print(f"window            : {human(window)}")
    print(f"headroom          : {human(DEFAULT_HEADROOM)}")
    print()
    print("disk requirement")
    print(f"  naive (tar+extract)      : {human(2*total)}")
    print(f"  this tool (bounded)      : ~{human(total + window + DEFAULT_HEADROOM)}")
    need = total + window + DEFAULT_HEADROOM
    print(f"  fits here                : {free >= need}"
          f"{'' if free >= need else f'  (short {human(need-free)})'}")
    print()
    cyc = max(1, (total + window - 1) // window)
    print(f"planned cycles    : {cyc} x ~{human(window)}")
    print()
    print("layout that will be produced")
    print(f"  {L['dataset']}/            <- 25,253 .hdf5 + metadata.csv")
    print(f"  {L['meta']}")
    print(f"  {L['state']}")
    return 0 if free >= need else 1


def get_metadata(L: dict) -> None:
    """Fetch metadata.csv and verify it against the official ETag MD5."""
    dest = L["meta"]
    dest.parent.mkdir(parents=True, exist_ok=True)
    import hashlib

    if dest.exists() and dest.stat().st_size == META_BYTES:
        h = hashlib.md5(dest.read_bytes()).hexdigest()
        if h == META_MD5:
            print(f"metadata.csv already present and MD5-verified.")
            # also make sure a copy sits inside the dataset dir (official layout)
            in_tree = L["dataset"] / "metadata.csv"
            if not in_tree.exists():
                L["dataset"].mkdir(parents=True, exist_ok=True)
                shutil.copy2(dest, in_tree)
            return
    print("fetching metadata.csv ...")
    for a in range(1, 11):
        try:
            req = urllib.request.Request(META_URL)
            with urllib.request.urlopen(req, timeout=180) as r:
                data = r.read()
            if len(data) != META_BYTES:
                raise IOError(f"short read {len(data)}")
            dest.write_bytes(data)
            break
        except Exception as e:  # noqa: BLE001
            print(f"  ! attempt {a}: {type(e).__name__}: {e}")
            time.sleep(min(30, 2 ** a))
    else:
        raise SystemExit("could not fetch metadata.csv")

    h = hashlib.md5(dest.read_bytes()).hexdigest()
    status = "MATCHES official ETag (bit-identical)" if h == META_MD5 else \
             f"MISMATCH (got {h})"
    print(f"metadata.csv MD5 {h} -> {status}")
    if h != META_MD5:
        raise SystemExit("metadata.csv failed integrity check; refusing to continue")
    L["dataset"].mkdir(parents=True, exist_ok=True)
    shutil.copy2(dest, L["dataset"] / "metadata.csv")


def phase_run(L: dict, window: int, max_cycles: int | None, src: "Source") -> int:
    for d in (L["raw"], L["dataset"], L["logs"]):
        d.mkdir(parents=True, exist_ok=True)
    st = State(L["state"])

    total = src.size()
    st.d["remote_bytes_total"] = total
    print(f"remote tar {human(total)}; resuming at "
          f"downloaded={human(st.downloaded)}, consumed={human(st.consumed)}")

    get_metadata(L)

    tar = L["tar"]
    cycle = 0
    t_start = time.time()
    while st.consumed < total:
        cycle += 1
        if max_cycles is not None and cycle > max_cycles:
            print(f"cycle budget reached ({max_cycles}); stopping cleanly. "
                  f"Re-run to continue.")
            break

        base = st.consumed                       # remote offset the tar starts at
        win_start = base
        local_cur = st.downloaded - base         # bytes already on disk
        win_end = min(total, win_start + window)
        if win_end <= st.downloaded:
            # already downloaded past this window; just extract + prune more
            win_end = min(total, st.downloaded)

        print(f"\n=== cycle {cycle}: remote window "
              f"[{human(win_start)} .. {human(win_end)}) = "
              f"{human(win_end-win_start)}; on disk {human(local_cur)}; "
              f"{len(st.d['segments'])} segments tracked ===")

        # ---- free-space guard ----
        free = free_bytes(tar.parent)
        need = (win_end - st.downloaded) + DEFAULT_HEADROOM
        if free < need:
            print(f"ABORT: free {human(free)} < needed {human(need)}. "
                  f"Nothing lost; free space and re-run.")
            return 3

        # ---- download into the tar at LOCAL offsets ----
        fh = open(tar, "r+b") if tar.exists() else open(tar, "wb")
        try:
            fh.truncate(max(0, win_end - base))
            for off_remote in range(st.downloaded, win_end, SEG):
                ln = min(SEG, win_end - off_remote)
                buf = src.fetch(off_remote, ln)
                fh.seek(off_remote - base)       # remote -> local
                fh.write(buf)
                fh.flush()
                os.fsync(fh.fileno())
                st.d["segments"][str(off_remote)] = len(buf)
                st.save()
            el = time.time() - t_start
            print(f"    downloaded {human(st.downloaded)}/{human(total)} "
                  f"({100*st.downloaded/total:.3f}%) "
                  f"avg {human(st.downloaded/el)}/s", flush=True)
        finally:
            fh.close()

        got = tar.stat().st_size
        exp = max(0, st.downloaded - base)
        if got != exp:
            print(f"  ! WARNING: tar on disk is {human(got)} but expected "
                  f"{human(exp)}; state/disk disagree.")

        # ---- extract this window (remote coords) ----
        consumed = extract_window(tar, L["dataset"], win_start, win_end,
                                  base, DEFAULT_HEADROOM, st)

        # ---- prune the consumed prefix (LOCAL offsets) ----
        prune_len = consumed - base
        if prune_len > 0:
            prune_prefix(tar, prune_len)
            # base moves to `consumed`; segments recorded in remote coords stay
            # valid, only their LOCAL projection changed. Nothing to rewrite.
            st.d["pruned_through"] = consumed
            st.save()

        print(f"  progress: conserved {human(st.consumed)}/{human(total)} "
              f"({100*st.consumed/total:.3f}%)  files={st.d['files_extracted']}")

        if st.consumed <= win_start and st.downloaded <= win_end:
            print("  ! no progress this cycle (window smaller than one tar "
                  "member, or nothing left to advance). Increase --window-gib "
                  "and re-run; stopping to avoid a spin.")
            break

    print(f"\ndownload+extract loop finished after {cycle} cycle(s), "
          f"{(time.time()-t_start)/3600:.2f} h")
    return phase_verify(L, quick=True)


def phase_verify(L: dict, quick: bool = False) -> int:
    here = Path(__file__).resolve().parent
    verifier = here / "verify_emg2pose_dataset.py"
    if not verifier.exists():
        print(f"! verifier not found at {verifier}; skipping verification")
        return 0
    cmd = [sys.executable, str(verifier),
           "--data", str(L["dataset"]),
           "--metadata", str(L["dataset"] / "metadata.csv"),
           "--report", str(L["logs"] / "verify_report.json")]
    if quick:
        cmd += ["--limit", "200"]
    print("\n=== running integrity verification ===")
    return os.spawnv(os.P_WAIT, sys.executable, cmd)


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("phase", choices=["plan", "run", "verify", "all"])
    ap.add_argument("--root", type=Path, default=Path("/data/emg2pose"),
                    help="directory to create/fill on the server")
    ap.add_argument("--window-gib", type=float, default=DEFAULT_WINDOW / 1024**3)
    ap.add_argument("--max-cycles", type=int, default=None,
                    help="stop after N windows (for budgeted cron-style runs)")
    ap.add_argument("--tar-url", default=TAR_URL)
    ap.add_argument("--seg-mib", type=int, default=SEG // 1024**2,
                    help="HTTP range request size; larger is faster on high-"
                         "latency links (measured: 1 MiB requests were ~35x "
                         "faster than tiny ones on the dev machine)")
    ap.add_argument("--source-tar", type=Path, default=None,
                    help="read the tar from a local file instead of the network; "
                         "intended for self-testing the pipeline only")
    args = ap.parse_args()

    L = layout(args.root)
    window = int(args.window_gib * 1024 ** 3)
    seg = args.seg_mib * 1024 ** 2
    src = Source(args.source_tar, args.tar_url, seg=seg)

    if args.phase == "plan":
        return phase_plan(L, window, src)
    if args.phase == "run":
        return phase_run(L, window, args.max_cycles, src)
    if args.phase == "verify":
        return phase_verify(L)
    rc = phase_run(L, window, args.max_cycles, src)
    return rc


if __name__ == "__main__":
    sys.exit(main())
