"""Host facts: memory and GPU.

These are deterministic inputs to routing, not judgments. Two of them matter on
this machine:

* **Free system RAM.** The 27B models are only partially offloaded, so their
  CPU-resident layers are read from RAM on every token. The stack's own tuning
  log recorded a drop from ~9.6 tok/s to 1.25 tok/s when RAM ran out, so a
  switch that would not fit is worth warning about before it happens.
* **Free VRAM.** Tells us whether a model is genuinely resident and how much
  headroom a load would have.

Both are best-effort: if `nvidia-smi` is missing or a call fails, the fact is
reported as unknown and the policy carries on without it.
"""

from __future__ import annotations

import asyncio
import ctypes
import shutil
import subprocess
import time
from dataclasses import dataclass
from typing import Any


@dataclass(slots=True)
class MemoryStatus:
    total_gb: float | None = None
    free_gb: float | None = None
    error: str | None = None

    @property
    def known(self) -> bool:
        return self.free_gb is not None


@dataclass(slots=True)
class GpuStatus:
    name: str | None = None
    total_mb: int | None = None
    used_mb: int | None = None
    free_mb: int | None = None
    driver: str | None = None
    error: str | None = None

    @property
    def known(self) -> bool:
        return self.free_mb is not None

    @property
    def free_gb(self) -> float | None:
        return None if self.free_mb is None else self.free_mb / 1024


class _MemoryStatusEx(ctypes.Structure):
    _fields_ = [
        ("dwLength", ctypes.c_ulong),
        ("dwMemoryLoad", ctypes.c_ulong),
        ("ullTotalPhys", ctypes.c_ulonglong),
        ("ullAvailPhys", ctypes.c_ulonglong),
        ("ullTotalPageFile", ctypes.c_ulonglong),
        ("ullAvailPageFile", ctypes.c_ulonglong),
        ("ullTotalVirtual", ctypes.c_ulonglong),
        ("ullAvailVirtual", ctypes.c_ulonglong),
        ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
    ]


def memory_status() -> MemoryStatus:
    """Physical memory, straight from the Windows API (no psutil dependency)."""
    try:
        status = _MemoryStatusEx()
        status.dwLength = ctypes.sizeof(_MemoryStatusEx)
        if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):  # type: ignore[attr-defined]
            return MemoryStatus(error="GlobalMemoryStatusEx failed")
        return MemoryStatus(
            total_gb=status.ullTotalPhys / 1024**3,
            free_gb=status.ullAvailPhys / 1024**3,
        )
    except (AttributeError, OSError) as error:
        # Not Windows, or the call is unavailable.
        return MemoryStatus(error=f"{type(error).__name__}: {error}")


_NVIDIA_QUERY = "name,memory.total,memory.used,memory.free,driver_version"


def gpu_status(timeout_s: float = 4.0) -> GpuStatus:
    """First NVIDIA GPU, via nvidia-smi. Unknown rather than fatal when absent."""
    executable = shutil.which("nvidia-smi")
    if not executable:
        return GpuStatus(error="nvidia-smi not found on PATH")
    try:
        completed = subprocess.run(
            [executable, f"--query-gpu={_NVIDIA_QUERY}", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=timeout_s,
            check=False,
        )
    except (OSError, subprocess.SubprocessError) as error:
        return GpuStatus(error=f"{type(error).__name__}: {error}")
    if completed.returncode != 0:
        return GpuStatus(error=f"nvidia-smi exited {completed.returncode}")
    line = completed.stdout.strip().splitlines()
    if not line:
        return GpuStatus(error="nvidia-smi returned no rows")
    parts = [part.strip() for part in line[0].split(",")]
    if len(parts) < 5:
        return GpuStatus(error=f"unexpected nvidia-smi output: {line[0]!r}")
    try:
        return GpuStatus(
            name=parts[0],
            total_mb=int(float(parts[1])),
            used_mb=int(float(parts[2])),
            free_mb=int(float(parts[3])),
            driver=parts[4],
        )
    except ValueError as error:
        return GpuStatus(error=f"could not parse nvidia-smi output: {error}")


class HostFacts:
    """Cached host facts.

    `nvidia-smi` costs tens of milliseconds, which is too much to pay on every
    routed request, so results are cached briefly. Memory is cheap but cached
    too, so one request sees one consistent view.
    """

    def __init__(self, *, ttl_s: float = 5.0) -> None:
        self._ttl_s = ttl_s
        self._memory: tuple[float, MemoryStatus] | None = None
        self._gpu: tuple[float, GpuStatus] | None = None
        self._gpu_lock = asyncio.Lock()

    def memory(self) -> MemoryStatus:
        now = time.monotonic()
        if self._memory and now - self._memory[0] < self._ttl_s:
            return self._memory[1]
        status = memory_status()
        self._memory = (now, status)
        return status

    async def gpu(self) -> GpuStatus:
        now = time.monotonic()
        if self._gpu and now - self._gpu[0] < self._ttl_s:
            return self._gpu[1]
        async with self._gpu_lock:
            # Another waiter may have refreshed it while we queued.
            if self._gpu and time.monotonic() - self._gpu[0] < self._ttl_s:
                return self._gpu[1]
            status = await asyncio.to_thread(gpu_status)
            self._gpu = (time.monotonic(), status)
            return status

    async def describe(self) -> dict[str, Any]:
        memory = self.memory()
        gpu = await self.gpu()
        return {
            "ram_total_gb": None if memory.total_gb is None else round(memory.total_gb, 2),
            "ram_free_gb": None if memory.free_gb is None else round(memory.free_gb, 2),
            "ram_error": memory.error,
            "gpu_name": gpu.name,
            "vram_total_mb": gpu.total_mb,
            "vram_free_mb": gpu.free_mb,
            "vram_used_mb": gpu.used_mb,
            "driver": gpu.driver,
            "gpu_error": gpu.error,
        }
