"""Cliente mínimo de la API REST de Airflow 3 para disparar y consultar el reentrenamiento."""

from __future__ import annotations

import time

import httpx

DAG_ID = "reentrenamiento"
ESTADOS = {
    "queued": "en_curso",
    "running": "en_curso",
    "scheduled": "en_curso",
    "up_for_retry": "en_curso",
    "success": "exitoso",
    "failed": "fallido",
}


class ErrorAirflow(RuntimeError):
    pass


class ClienteAirflow:
    def __init__(self, url: str, usuario: str, clave: str, timeout: float = 5.0):
        self.url, self.usuario, self.clave, self.timeout = url.rstrip("/"), usuario, clave, timeout
        self._token: str | None = None
        self._vence = 0.0

    def _cabeceras(self) -> dict[str, str]:
        if self._token is None or time.monotonic() > self._vence:
            r = httpx.post(f"{self.url}/auth/token", timeout=self.timeout,
                           json={"username": self.usuario, "password": self.clave})
            r.raise_for_status()
            self._token, self._vence = r.json()["access_token"], time.monotonic() + 600
        return {"Authorization": f"Bearer {self._token}"}

    def disparar(self) -> str:
        try:
            r = httpx.post(f"{self.url}/api/v2/dags/{DAG_ID}/dagRuns", headers=self._cabeceras(),
                           json={"logical_date": None, "conf": {}}, timeout=self.timeout)
        except httpx.HTTPError as e:
            raise ErrorAirflow(f"Airflow no responde: {e}") from e
        if r.status_code >= 300:
            raise ErrorAirflow(f"Airflow rechazó la ejecución ({r.status_code}): {r.text[:200]}")
        return r.json()["dag_run_id"]

    def estado(self, run_id: str) -> str | None:
        """Estado del contrato (en_curso, exitoso, fallido) o None si la ejecución no existe."""
        try:
            r = httpx.get(f"{self.url}/api/v2/dags/{DAG_ID}/dagRuns/{run_id}",
                          headers=self._cabeceras(), timeout=self.timeout)
        except httpx.HTTPError as e:
            raise ErrorAirflow(f"Airflow no responde: {e}") from e
        if r.status_code == 404:
            return None
        r.raise_for_status()
        return ESTADOS.get(r.json()["state"], "en_curso")

    def sano(self) -> bool:
        try:
            r = httpx.get(f"{self.url}/api/v2/monitor/health", timeout=self.timeout)
            return r.status_code == 200
        except httpx.HTTPError:
            return False
