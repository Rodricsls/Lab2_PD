"""Catálogos y datos sintéticos: las pruebas no dependen de los datos publicados, de la
base de datos ni de un servidor de MLflow."""

from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from novamercado.calidad import ResolutorFamilias
from novamercado.catalogos import cargar_catalogos

DATOS = Path(__file__).parent / "datos"
FECHA = date(2025, 3, 10)


@pytest.fixture(scope="session")
def cat():
    return cargar_catalogos(DATOS)


@pytest.fixture(scope="session")
def familias(cat):
    return ResolutorFamilias(cat, 0.88, 0.75)


def historia_sintetica(cat, desde=date(2024, 11, 1), hasta=date(2025, 3, 9), semilla=0):
    """Ventas con estacionalidad semanal y efecto de promoción."""
    rng = np.random.default_rng(semilla)
    filas = []
    for d in pd.date_range(desde, hasta):
        for t in cat.ids_tienda:
            if d.date() < cat.apertura(t):
                continue
            for k, f in enumerate(cat.ids_familia):
                promo = int(rng.integers(0, 5))
                nivel = 50 + 20 * k + 10 * t
                semanal = 1.3 if d.dayofweek >= 5 else 1.0
                ventas = rng.poisson(nivel * semanal * (1 + 0.05 * promo))
                filas.append((d.date(), t, f, float(ventas), promo))
    return pd.DataFrame(filas, columns=["fecha", "tienda_id", "familia_id", "ventas", "en_promocion"])


@pytest.fixture(scope="session")
def historia(cat):
    return historia_sintetica(cat)


def cuerpo_prediccion(cat, tienda=1, fecha=FECHA):
    return {
        "tienda_id": tienda,
        "fecha": fecha.isoformat(),
        "familias": [{"familia_id": f, "en_promocion": 0} for f in cat.ids_familia],
    }


def cuerpo_observacion(cat, tienda=1, fecha=FECHA - timedelta(days=1), ventas=100):
    return {
        "tienda_id": tienda,
        "fecha": fecha.isoformat(),
        "ventas": [{"familia_id": f, "ventas": ventas} for f in cat.ids_familia],
    }
