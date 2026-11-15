"""Parámetros (config/params.yaml) y entorno (variables de Docker Compose)."""

from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

RAIZ = Path(__file__).resolve().parents[2]


@lru_cache
def params() -> dict[str, Any]:
    ruta = Path(os.environ.get("PARAMS_PATH", RAIZ / "config" / "params.yaml"))
    with open(ruta, encoding="utf-8") as f:
        return yaml.safe_load(f)


@dataclass(frozen=True)
class Entorno:
    datos: Path
    database_url: str
    mlflow_uri: str
    airflow_url: str
    airflow_usuario: str
    airflow_clave: str
    cache_modelo: Path


def entorno() -> Entorno:
    return Entorno(
        datos=Path(os.environ.get("DATOS_DIR", RAIZ / "datos" / "publicados")),
        database_url=os.environ.get(
            "DATABASE_URL", "postgresql://novamercado:novamercado@localhost:5433/novamercado"
        ),
        mlflow_uri=os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5000"),
        airflow_url=os.environ.get("AIRFLOW_URL", "http://localhost:8080"),
        airflow_usuario=os.environ.get("AIRFLOW_USUARIO", "airflow"),
        airflow_clave=os.environ.get("AIRFLOW_CLAVE", "airflow"),
        cache_modelo=Path(os.environ.get("CACHE_MODELO", RAIZ / ".cache" / "modelo")),
    )
