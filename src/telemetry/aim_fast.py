"""Runtime batching patch for Aim remote client."""

import threading
from typing import Any, Optional

try:
    from aim import Run
    from aim.ext.transport.client import Client
    from aim.storage.treeutils import encode_tree
    _AIM_AVAILABLE = True
except ImportError:
    _AIM_AVAILABLE = False
    Client = None
    encode_tree = None
    Run = None

_ORIG_START = Client.start_instructions_batch if Client else None
_ORIG_FLUSH = Client.flush_instructions_batch if Client else None
_ORIG_WAIT = Client.get_queue if Client else None
_PATCHED = False


def is_patched() -> bool:
    """Return whether the Aim client batching patch is active."""
    return _PATCHED


class BatchTracker:
    """Thread-safe batch accumulator for Aim write instructions."""

    def __init__(self, client: Any, batch_size: int = 50) -> None:
        """Initialize tracker with target batch capacity."""
        self.client = client
        self.batch_size = batch_size
        self._lock = threading.Lock()
        self._buffers = {}

    def add(self, hash_: str, instructions: list) -> None:
        """Add instructions to buffer and flush when threshold exceeded."""
        with self._lock:
            if hash_ not in self._buffers:
                self._buffers[hash_] = []
            self._buffers[hash_].extend(instructions)
            if len(self._buffers[hash_]) >= self.batch_size * 7:
                self._flush_locked(hash_)

    def flush(self, hash_: Optional[str] = None) -> None:
        """Flush buffered instructions for given hash or all runs."""
        with self._lock:
            if hash_ is not None:
                self._flush_locked(hash_)
            else:
                for h in list(self._buffers.keys()):
                    self._flush_locked(h)

    def _flush_locked(self, hash_: str) -> None:
        """Flush buffer while lock is acquired."""
        buf = self._buffers.pop(hash_, None)
        if buf and encode_tree:
            encoded = list(encode_tree(buf, strict=False))
            self.client.get_queue().register_task(
                self.client, self.client._run_write_instructions, encoded
            )


_client_trackers = {}


def patch(batch_size: int = 50) -> None:
    """Apply monkey-patch to Aim client for batched remote transport."""
    global _PATCHED
    if _PATCHED or not _AIM_AVAILABLE:
        return

    def start_instructions_batch(self, hash_):
        if getattr(self._thread_local, "atomic_instructions", None) is None:
            self._thread_local.atomic_instructions = {}
        if hash_ not in self._thread_local.atomic_instructions:
            self._thread_local.atomic_instructions[hash_] = []

    def flush_instructions_batch(self, hash_, force=False):
        cid = id(self)
        if cid not in _client_trackers:
            _client_trackers[cid] = BatchTracker(self, batch_size=batch_size)
        tracker = _client_trackers[cid]
        instructions = getattr(self._thread_local, "atomic_instructions", {}).pop(hash_, None)
        if instructions:
            tracker.add(hash_, instructions)
        if force:
            tracker.flush(hash_)

    def get_queue(self):
        q = _ORIG_WAIT(self)
        cid = id(self)
        if cid in _client_trackers:
            orig_wait = q.wait_for_finish

            def hooked_wait():
                _client_trackers[cid].flush()
                return orig_wait()

            q.wait_for_finish = hooked_wait
        return q

    Client.start_instructions_batch = start_instructions_batch
    Client.flush_instructions_batch = flush_instructions_batch
    Client.get_queue = get_queue

    orig_run_close = Run.close

    def hooked_run_close(self, *args, **kwargs):
        if getattr(self, "repo", None) and getattr(self.repo, "is_remote_repo", False):
            cid = id(self.repo._client)
            if cid in _client_trackers:
                _client_trackers[cid].flush(self.hash)
        return orig_run_close(self, *args, **kwargs)

    Run.close = hooked_run_close
    _PATCHED = True
