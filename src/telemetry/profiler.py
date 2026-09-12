import time


def _sync_target(obj):
    """Recursively synchronize any object supporting block_until_ready."""
    if obj is None:
        return
    if hasattr(obj, "block_until_ready") and callable(getattr(obj, "block_until_ready")):
        try:
            obj.block_until_ready()
        except Exception:
            pass
    elif isinstance(obj, dict):
        for v in obj.values():
            _sync_target(v)
    elif isinstance(obj, (list, tuple)):
        for item in obj:
            _sync_target(item)


class ExecutionProfiler:
    """Context manager for accurate inference latency profiling with JAX array synchronization."""

    def __init__(self, sync_target=None):
        """Initialize profiler state and cumulative timing stats."""
        self.sync_targets = [sync_target] if sync_target is not None else []
        self._start_time = 0.0
        self.elapsed_sec = 0.0
        self.elapsed_ms = 0.0
        self.total_ms = 0.0
        self.call_count = 0
        self.mean_ms = 0.0

    def start(self):
        """Start measurement clock."""
        self._start_time = time.perf_counter()

    def stop(self):
        """Stop measurement clock and update statistics."""
        end_time = time.perf_counter()
        self.elapsed_sec = max(0.0, end_time - self._start_time)
        self.elapsed_ms = self.elapsed_sec * 1000.0
        self.total_ms += self.elapsed_ms
        self.call_count += 1
        self.mean_ms = self.total_ms / self.call_count
        return self.elapsed_ms

    @property
    def last_elapsed_ms(self):
        """Return the most recent execution latency in milliseconds."""
        return self.elapsed_ms

    def sync(self, *targets):
        """Synchronize target tensors and compute devices."""
        for target in targets:
            _sync_target(target)

    def register_sync(self, *targets):
        """Register targets to synchronize automatically at exit."""
        self.sync_targets.extend(targets)

    def reset(self):
        """Reset cumulative statistics."""
        self.elapsed_sec = 0.0
        self.elapsed_ms = 0.0
        self.total_ms = 0.0
        self.call_count = 0
        self.mean_ms = 0.0
        self.sync_targets.clear()

    def __enter__(self):
        """Start timer upon entering context."""
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Sync registered targets and stop timer on context exit."""
        if self.sync_targets:
            self.sync(*self.sync_targets)
        self.stop()
        return False
