from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

import psycopg
from psycopg.rows import dict_row

try:
    from pgvector.psycopg import register_vector
except ImportError:  # pragma: no cover - optional at runtime
    register_vector = None  # type: ignore[assignment]


class NeonStore:
    def __init__(self, dsn: Optional[str]) -> None:
        self.dsn = dsn

    @property
    def enabled(self) -> bool:
        return bool(self.dsn)

    @contextmanager
    def connection(self):
        if not self.dsn:
            raise RuntimeError("Neon database URL is not configured.")
        conn = psycopg.connect(self.dsn, autocommit=True)
        if register_vector is not None:
            register_vector(conn)
        try:
            yield conn
        finally:
            conn.close()

    def run_sql_file(self, path: Path) -> None:
        if not self.enabled:
            return
        sql = path.read_text(encoding="utf-8")
        with self.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(sql)

    def execute(self, sql: str, params: Optional[Sequence[Any]] = None) -> None:
        with self.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(sql, params)

    def executemany(self, sql: str, params_seq: Iterable[Sequence[Any]]) -> None:
        with self.connection() as conn:
            with conn.cursor() as cur:
                cur.executemany(sql, params_seq)

    def fetch_all(self, sql: str, params: Optional[Sequence[Any]] = None) -> List[Dict[str, Any]]:
        with self.connection() as conn:
            with conn.cursor(row_factory=dict_row) as cur:
                cur.execute(sql, params)
                return list(cur.fetchall())

