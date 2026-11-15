"""Catálogos publicados: tiendas, familias, costos y feriados."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from pathlib import Path

import pandas as pd


@dataclass(frozen=True)
class Catalogos:
    tiendas: pd.DataFrame  # índice tienda_id
    familias: pd.DataFrame  # índice familia_id
    costos: pd.DataFrame  # índice familia_id: costo_faltante, costo_sobrante
    feriados: pd.DataFrame  # fecha (date), nombre, tipo, ambito

    @property
    def ids_tienda(self) -> list[int]:
        return [int(t) for t in self.tiendas.index]

    @property
    def ids_familia(self) -> list[str]:
        return list(self.familias.index)

    def apertura(self, tienda_id: int) -> date:
        return self.tiendas.loc[tienda_id, "fecha_apertura"]

    @property
    def ultimo_feriado(self) -> date:
        return max(self.feriados["fecha"])

    def cuantil_critico(self, familia_id: str) -> float:
        c = self.costos.loc[familia_id]
        return c["costo_faltante"] / (c["costo_faltante"] + c["costo_sobrante"])


def cargar_catalogos(carpeta: Path) -> Catalogos:
    tiendas = pd.read_csv(carpeta / "tiendas.csv")
    tiendas["fecha_apertura"] = pd.to_datetime(tiendas["fecha_apertura"]).dt.date
    feriados = pd.read_csv(carpeta / "feriados.csv")
    feriados["fecha"] = pd.to_datetime(feriados["fecha"]).dt.date
    return Catalogos(
        tiendas=tiendas.set_index("tienda_id"),
        familias=pd.read_csv(carpeta / "familias.csv").set_index("familia_id"),
        costos=pd.read_csv(carpeta / "costos.csv").set_index("familia_id"),
        feriados=feriados,
    )
