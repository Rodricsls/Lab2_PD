"""Estado del servicio: modelo vigente (recarga en caliente) y salud de las dependencias.

- `GestorModelo` consulta cada `segundos_recarga_modelo` qué versión tiene el alias en el
  Registry; si cambió, la descarga y la cambia sin reiniciar. El modelo cargado se guarda
  también en disco: si MLflow no está disponible (incluso al arrancar) la API sigue
  respondiendo con el último modelo que conoce.
- `MonitorSalud` revisa base de datos, MLflow y Airflow cada `segundos_chequeo_salud` en
  segundo plano, de modo que `GET /health` responde al instante y refleja una caída en
  segundos.
"""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable
from pathlib import Path

import httpx
import joblib

from novamercado.modelo import ModeloVentas

log = logging.getLogger(__name__)


class GestorModelo:
    def __init__(self, nombre: str, alias: str, cache: Path,
                 consultar_version: Callable[[], str | None],
                 descargar: Callable[[str], ModeloVentas]):
        self.nombre, self.alias, self.cache = nombre, alias, cache
        self._consultar, self._descargar = consultar_version, descargar
        self._lock = threading.Lock()
        self.modelo: ModeloVentas | None = None
        self.version: str | None = None

    def actual(self) -> tuple[ModeloVentas | None, str | None]:
        with self._lock:
            return self.modelo, self.version

    def cargar_cache(self) -> bool:
        archivo = self.cache / "modelo.joblib"
        if not archivo.is_file():
            return False
        try:
            modelo, version = joblib.load(archivo)
        except Exception:  # cache corrupta: se ignora
            log.exception("No se pudo leer la cache del modelo")
            return False
        with self._lock:
            self.modelo, self.version = modelo, version
        log.info("Modelo versión %s cargado desde la cache", version)
        return True

    def sincronizar(self) -> bool:
        """Devuelve True si cambió el modelo. Lanza excepción si MLflow no responde."""
        version = self._consultar()
        if version is None or version == self.version:
            return False
        modelo = self._descargar(version)
        with self._lock:
            self.modelo, self.version = modelo, version
        self.cache.mkdir(parents=True, exist_ok=True)
        temporal = self.cache / "modelo.joblib.tmp"
        joblib.dump((modelo, version), temporal)
        temporal.replace(self.cache / "modelo.joblib")
        log.info("Modelo vigente actualizado a la versión %s", version)
        return True

    def iniciar(self, cada: float) -> None:
        def ciclo():
            while True:
                try:
                    self.sincronizar()
                except Exception as e:  # MLflow caído: se conserva el modelo actual
                    log.warning("No se pudo consultar el Registry: %s", e)
                time.sleep(cada)

        threading.Thread(target=ciclo, daemon=True, name="recarga-modelo").start()


class MonitorSalud:
    def __init__(self, chequeos: dict[str, Callable[[], bool]]):
        self.chequeos = chequeos
        self.estado: dict[str, bool] = dict.fromkeys(chequeos, False)

    def revisar(self) -> None:
        for nombre, chequeo in self.chequeos.items():
            try:
                self.estado[nombre] = bool(chequeo())
            except Exception:
                self.estado[nombre] = False

    def caidas(self) -> list[str]:
        return [n for n, ok in self.estado.items() if not ok]

    def iniciar(self, cada: float) -> None:
        def ciclo():
            while True:
                self.revisar()
                time.sleep(cada)

        threading.Thread(target=ciclo, daemon=True, name="monitor-salud").start()


def http_ok(url: str, timeout: float) -> Callable[[], bool]:
    def chequeo() -> bool:
        return httpx.get(url, timeout=timeout).status_code == 200

    return chequeo
