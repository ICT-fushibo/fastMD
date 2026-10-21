"""Local transport hooks for MatRIS interaction layers.

LOCAL_FORWARD_RUNTIME is the layer-level EngineRuntime singleton. Feature
rows remain on one device and no inter-device transport is inserted."""
from __future__ import annotations

from fastmd.engine.runtime import LOCAL_RUNTIME


LOCAL_FORWARD_RUNTIME = LOCAL_RUNTIME
