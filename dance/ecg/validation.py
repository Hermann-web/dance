from __future__ import annotations

import importlib.util
from typing import Any, Callable

if importlib.util.find_spec("check_shapes") is not None:
    from check_shapes import check_shapes as _check_shapes
else:

    def _check_shapes(*args: Any, **kwargs: Any) -> Callable:
        def _decorator(fn: Callable) -> Callable:
            return fn

        return _decorator


check_shapes = _check_shapes

