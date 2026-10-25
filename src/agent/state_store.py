"""Validated session state persistence for resumable agent workflows."""
import json
from pathlib import Path
from threading import RLock
from typing import Protocol
from ..contracts import SessionState

class StateStore(Protocol):
    def get(self, session_id: str) -> SessionState | None: ...
    def put(self, state: SessionState) -> SessionState: ...
    def delete(self, session_id: str) -> None: ...

class MemoryStateStore:
    """Process-local store used by unit tests and local development."""
    def __init__(self):
        self._states: dict[str, SessionState] = {}
        self._lock = RLock()

    def get(self, session_id: str) -> SessionState | None:
        with self._lock:
            state = self._states.get(session_id)
            return SessionState.model_validate(state.model_dump()) if state else None

    def put(self, state: SessionState) -> SessionState:
        with self._lock:
            validated = SessionState.model_validate(state.model_dump())
            self._states[validated.session_id] = validated
            return SessionState.model_validate(validated.model_dump())

    def delete(self, session_id: str) -> None:
        with self._lock:
            self._states.pop(session_id, None)

class JsonStateStore:
    """Directory-backed store; each session is one validated JSON document."""
    def __init__(self, root: Path):
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)
        self._lock = RLock()

    def _path(self, session_id: str) -> Path:
        safe = ''.join(c for c in session_id if c.isalnum() or c in '-_')
        if not safe or safe != session_id:
            raise ValueError('invalid session ID')
        return self.root / f'{safe}.json'

    def get(self, session_id: str) -> SessionState | None:
        with self._lock:
            path = self._path(session_id)
            if not path.exists():
                return None
            return SessionState.model_validate_json(path.read_text())

    def put(self, state: SessionState) -> SessionState:
        validated = SessionState.model_validate(state.model_dump())
        with self._lock:
            path = self._path(validated.session_id)
            temp = path.with_suffix('.tmp')
            temp.write_text(json.dumps(validated.model_dump(), indent=2) + '\n')
            temp.replace(path)
            return SessionState.model_validate(validated.model_dump())

    def delete(self, session_id: str) -> None:
        with self._lock:
            path = self._path(session_id)
            if path.exists():
                path.unlink()
