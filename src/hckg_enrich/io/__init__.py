"""I/O utilities: file safety and KG adapter."""

from hckg_enrich.io.file_safety import GraphFileLock, LockTimeoutError, atomic_write_json
from hckg_enrich.io.kg_adapter import KGAdapter

__all__ = ["GraphFileLock", "KGAdapter", "LockTimeoutError", "atomic_write_json"]
