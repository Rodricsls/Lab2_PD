"""Modelos, cantidad sugerida y métricas de negocio.

Ambos modelos predicen la venta esperada de un día; la cantidad sugerida es
`round(venta_esperada × factor_familia)`, con el factor de cada familia elegido para
minimizar el costo de `costos.csv` en los últimos días antes del corte (newsvendor: en
familias donde faltar cuesta más que sobrar el factor queda > 1, y al revés).
"""

from __future__ import annotations

from typing import Any

import lightgbm as lgb
import numpy as np
import pandas as pd

from novamercado.catalogos import Catalogos
from novamercado.features import CATEGORICAS, FEATURES


def costo(real, sugerido, c_faltante, c_sobrante) -> np.ndarray:
    real, sugerido = np.asarray(real, float), np.asarray(sugerido, float)
    return c_faltante * np.maximum(real - sugerido, 0) + c_sobrante * np.maximum(sugerido - real, 0)


def costo_tabla(df: pd.DataFrame, cat: Catalogos) -> np.ndarray:
    c = cat.costos.loc[df["familia_id"]]
    return costo(df["ventas"], df["cantidad_sugerida"], c["costo_faltante"].to_numpy(),
                 c["costo_sobrante"].to_numpy())


def metricas(df: pd.DataFrame, cat: Catalogos) -> dict[str, float]:
    """df con ventas, venta_esperada y cantidad_sugerida."""
    y, p = df["ventas"].to_numpy(float), df["venta_esperada"].to_numpy(float)
    c = costo_tabla(df, cat)
    return {
        "filas": float(len(df)),
        "costo_total": float(c.sum()),
        "costo_por_unidad": float(c.sum() / max(y.sum(), 1)),
        "wape": float(np.abs(y - p).sum() / max(y.sum(), 1)),
        "mae": float(np.abs(y - p).mean()),
        "sesgo": float((p - y).sum() / max(y.sum(), 1)),
    }


def elegir_factores(df: pd.DataFrame, cat: Catalogos, cfg: dict[str, Any]) -> dict[str, float]:
    """Por familia, el factor de la grilla que minimiza el costo (df con ventas y venta_esperada)."""
    grilla = np.round(np.arange(cfg["factor_min"], cfg["factor_max"] + 1e-9, cfg["factor_paso"]), 4)
    factores = {}
    for familia in cat.ids_familia:
        sub = df[df["familia_id"] == familia]
        cf, cs = cat.costos.loc[familia, ["costo_faltante", "costo_sobrante"]]
        if sub.empty:
            factores[familia] = 1.0
            continue
        y, p = sub["ventas"].to_numpy(float), sub["venta_esperada"].to_numpy(float)
        costos = [costo(y, np.round(p * f), cf, cs).sum() for f in grilla]
        factores[familia] = float(grilla[int(np.argmin(costos))])
    return factores


class ModeloVentas:
    """Interfaz común: `predecir(features)` → venta_esperada y cantidad_sugerida."""

    tipo = "base"

    def __init__(self, factores: dict[str, float] | None = None, info: dict | None = None):
        self.factores = factores or {}
        self.info = info or {}

    def venta_esperada(self, X: pd.DataFrame) -> np.ndarray:
        raise NotImplementedError

    def predecir(self, X: pd.DataFrame) -> pd.DataFrame:
        esperada = np.clip(np.nan_to_num(self.venta_esperada(X), nan=0.0), 0, None)
        factor = X["familia_id"].map(self.factores).fillna(1.0).to_numpy()
        return pd.DataFrame(
            {
                "venta_esperada": np.round(esperada, 1),
                "cantidad_sugerida": np.maximum(np.round(esperada * factor), 0).astype(int),
            },
            index=X.index,
        )


class Baseline(ModeloVentas):
    """Media de las últimas 4 ventas del mismo día de la semana (o de 28 días si faltan)."""

    tipo = "baseline"

    def ajustar(self, X: pd.DataFrame, y: pd.Series) -> Baseline:
        return self

    def venta_esperada(self, X: pd.DataFrame) -> np.ndarray:
        return X["mismo_dia_4"].fillna(X["base"]).to_numpy(float)


class ModeloLGBM(ModeloVentas):
    """LightGBM global sobre log(venta / base): captura promociones, calendario y tienda."""

    tipo = "lightgbm"

    def __init__(self, cfg: dict[str, Any], semilla: int, **kw):
        super().__init__(**kw)
        self.cfg, self.semilla = cfg, semilla

    def _x(self, X: pd.DataFrame) -> pd.DataFrame:
        x = X[FEATURES].copy()
        for c in CATEGORICAS:
            x[c] = pd.Categorical(x[c].astype(str), categories=self.categorias[c])
        return x

    def ajustar(self, X: pd.DataFrame, y: pd.Series) -> ModeloLGBM:
        ok = X["base"].notna() & (X["base"] > 0) & y.notna()
        X, y = X[ok], y[ok]
        self.categorias = {c: sorted(X[c].astype(str).unique()) for c in CATEGORICAS}
        objetivo = np.log1p(y.to_numpy(float)) - np.log1p(X["base"].to_numpy(float))
        params = {k: v for k, v in self.cfg.items() if k != "num_boost_round"}
        params |= {"verbose": -1, "seed": self.semilla, "num_threads": 2}
        self.booster = lgb.train(
            params,
            lgb.Dataset(self._x(X), objetivo, categorical_feature=CATEGORICAS),
            num_boost_round=self.cfg["num_boost_round"],
        )
        return self

    def venta_esperada(self, X: pd.DataFrame) -> np.ndarray:
        relativo = self.booster.predict(self._x(X))
        base = X["base"].to_numpy(float)
        esperada = np.expm1(relativo + np.log1p(np.nan_to_num(base, nan=0.0)))
        # Sin historia reciente no hay base: se usa el mismo día de semanas anteriores.
        return np.where(np.isfinite(base) & (base > 0), esperada, X["mismo_dia_4"].to_numpy(float))
