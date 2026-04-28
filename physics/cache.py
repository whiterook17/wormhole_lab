"""Intra-computation array cache for expensive physics functions.

Streamlit's @st.cache_data handles UI-level caching.
ArrayCache handles within-a-single-computation caching (e.g. repeated
calls to b_prime or ricci_scalar_MT during the same f(R) solve).
"""

import hashlib
import numpy as np


def _arr_key(*arrays_and_scalars):
    """Build a hashable cache key from arrays and scalar values."""
    h = hashlib.sha256()
    for item in arrays_and_scalars:
        if isinstance(item, np.ndarray):
            h.update(item.tobytes())
            h.update(str(item.shape).encode())
            h.update(str(item.dtype).encode())
        else:
            h.update(str(item).encode())
    return h.hexdigest()[:24]


class ArrayCache:
    """LRU-like manual cache for numpy-array-valued computations.

    Arrays are not hashable so we hash their bytes instead.
    Eviction is FIFO once max_entries is reached.
    """

    def __init__(self, max_entries: int = 64):
        self._cache: dict = {}
        self._max = max_entries

    def get_or_compute(self, key: str, compute_fn):
        if key in self._cache:
            return self._cache[key]
        if len(self._cache) >= self._max:
            oldest = next(iter(self._cache))
            del self._cache[oldest]
        result = compute_fn()
        self._cache[key] = result
        return result

    def clear(self):
        self._cache.clear()

    def __len__(self):
        return len(self._cache)


# Module-level caches used by physics functions
_ricci_cache = ArrayCache(max_entries=64)
_bprime_cache = ArrayCache(max_entries=128)
_sigma_cache = ArrayCache(max_entries=64)
_delta_cache = ArrayCache(max_entries=64)
