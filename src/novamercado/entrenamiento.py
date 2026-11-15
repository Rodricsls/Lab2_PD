"""Entrenamiento, evaluación y criterio de reemplazo (usado por el arranque y por el DAG).

Procedimiento de cada entrenamiento con datos hasta H:
1. Ajustar el modelo con ventas hasta H − dias_calibracion.
2. Pronosticar los últimos `dias_calibracion` días y elegir el factor de cada familia que
   minimiza el costo.
3. Reajustar con todo hasta H, conservando esos factores.

Reentrenamiento (criterio de reemplazo, D5). La comparación se hace en una ventana W que
ningún modelo comparado vio: los días con ventas conocidas posteriores a los datos del
vigente (como máximo los últimos `dias_ventana`). El challenger y el baseline se entrenan
con datos hasta antes de W. Se promueve si (1) el challenger le gana al baseline en W y
(2) su costo en W no supera al del vigente en más de `tolerancia`. Si W tiene menos de
`dias_minimos` días no hay evidencia suficiente y el vigente se conserva. Lo que se
registra y se promueve es el challenger reentrenado con todos los datos.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Any

import pandas as pd

from novamercado.catalogos import Catalogos
from novamercado.features import construir
from novamercado.modelo import Baseline, ModeloLGBM, ModeloVentas, elegir_factores, metricas


@dataclass
class Tabla:
    """Features de todas las filas con venta conocida (se calcula una vez por ciclo)."""

    datos: pd.DataFrame  # features + ventas

    @classmethod
    def desde_historia(cls, historia: pd.DataFrame, cat: Catalogos) -> Tabla:
        inicio = historia["fecha"].min() + timedelta(days=28)
        filas = historia[historia["fecha"] >= inicio].reset_index(drop=True)
        X = construir(historia, filas[["fecha", "tienda_id", "familia_id", "en_promocion"]], cat)
        return cls(X.assign(ventas=filas["ventas"].to_numpy()))

    def rango(self, desde: date | None, hasta: date) -> pd.DataFrame:
        f = self.datos["fecha"]
        mascara = f <= hasta if desde is None else (f >= desde) & (f <= hasta)
        return self.datos[mascara]

    @property
    def ultima_fecha(self) -> date:
        return self.datos["fecha"].max()


def nuevo_modelo(tipo: str, params: dict[str, Any]) -> ModeloVentas:
    if tipo == "baseline":
        return Baseline()
    cfg = params["entrenamiento"]
    return ModeloLGBM(cfg["lightgbm"], cfg["semilla"])


def entrenar(tipo: str, tabla: Tabla, cat: Catalogos, params: dict[str, Any],
             hasta: date) -> ModeloVentas:
    cfg = params["entrenamiento"]
    corte = hasta - timedelta(days=cfg["dias_calibracion"])
    interno = tabla.rango(None, corte)
    calibracion = tabla.rango(corte + timedelta(days=1), hasta)
    previo = nuevo_modelo(tipo, params).ajustar(interno, interno["ventas"])
    factores = elegir_factores(
        calibracion.assign(venta_esperada=previo.venta_esperada(calibracion)), cat, cfg
    )
    todo = tabla.rango(None, hasta)
    modelo = nuevo_modelo(tipo, params).ajustar(todo, todo["ventas"])
    modelo.factores = factores
    modelo.info = {
        "tipo": tipo,
        "datos_desde": str(todo["fecha"].min()),
        "datos_hasta": str(hasta),
        "filas": int(len(todo)),
    }
    return modelo


def evaluar(modelo: ModeloVentas, tabla: Tabla, cat: Catalogos, desde: date,
            hasta: date) -> dict[str, float]:
    ventana = tabla.rango(desde, hasta)
    pred = modelo.predecir(ventana)
    return metricas(ventana.assign(**pred), cat)


def decidir(costo_challenger: float, costo_baseline: float, costo_vigente: float | None,
            tolerancia: float) -> tuple[bool, str]:
    """Criterio de reemplazo (función pura)."""
    def n(x):
        return f"{x:,.0f}".replace(",", " ")

    if costo_challenger >= costo_baseline:
        return False, (f"Costo en validación {n(costo_challenger)} no mejora al baseline "
                       f"({n(costo_baseline)})")
    if costo_vigente is None:
        return True, (f"Primer modelo: costo en validación {n(costo_challenger)} vs "
                      f"{n(costo_baseline)} del baseline")
    if costo_challenger <= costo_vigente * (1 + tolerancia):
        return True, (f"Costo en validación {n(costo_challenger)} vs {n(costo_vigente)} del "
                      f"vigente (tolerancia {tolerancia:.0%}; mejora al baseline "
                      f"{n(costo_baseline)})")
    return False, (f"Costo en validación {n(costo_challenger)} vs {n(costo_vigente)} del "
                   f"vigente: supera la tolerancia de {tolerancia:.0%}")


@dataclass
class Ciclo:
    challenger: ModeloVentas
    promovido: bool
    motivo: str
    ventana: tuple[date, date]
    metricas: dict[str, dict[str, float]] = field(default_factory=dict)
    candidatos_holdout: dict[str, ModeloVentas] = field(default_factory=dict)


def ventana(fin: date, vigente: ModeloVentas | None, cfg: dict[str, Any]) -> tuple[date, date]:
    """Días de comparación: los posteriores a los datos del vigente, como máximo `dias_ventana`."""
    inicio = fin - timedelta(days=cfg["dias_ventana"] - 1)
    if vigente is not None and "datos_hasta" in vigente.info:
        visto = date.fromisoformat(vigente.info["datos_hasta"])
        inicio = max(inicio, visto + timedelta(days=1))
    return inicio, fin


def ciclo(historia: pd.DataFrame, cat: Catalogos, params: dict[str, Any],
          vigente: ModeloVentas | None) -> Ciclo:
    tabla = Tabla.desde_historia(historia, cat)
    fin = tabla.ultima_fecha
    cfg = params["promocion"]
    inicio, _ = ventana(fin, vigente, cfg)
    dias_nuevos = (fin - inicio).days + 1
    if vigente is not None and dias_nuevos < cfg["dias_minimos"]:
        challenger = entrenar("lightgbm", tabla, cat, params, fin)
        motivo = (f"Solo hay {max(dias_nuevos, 0)} días con ventas posteriores a los datos del "
                  f"vigente (mínimo {cfg['dias_minimos']}): no hay evidencia para reemplazarlo")
        return Ciclo(challenger, False, motivo, (inicio, fin))
    antes = inicio - timedelta(days=1)

    holdout = {t: entrenar(t, tabla, cat, params, antes) for t in ("baseline", "lightgbm")}
    medidas = {f"{t}_validacion": evaluar(m, tabla, cat, inicio, fin) for t, m in holdout.items()}
    costo_vigente = None
    if vigente is not None:
        medidas["vigente_validacion"] = evaluar(vigente, tabla, cat, inicio, fin)
        costo_vigente = medidas["vigente_validacion"]["costo_total"]

    promovido, motivo = decidir(
        medidas["lightgbm_validacion"]["costo_total"],
        medidas["baseline_validacion"]["costo_total"],
        costo_vigente,
        params["promocion"]["tolerancia"],
    )
    challenger = entrenar("lightgbm", tabla, cat, params, fin)
    return Ciclo(challenger, promovido, motivo, (inicio, fin), medidas, holdout)
