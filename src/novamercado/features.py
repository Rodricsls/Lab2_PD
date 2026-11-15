"""Features de un día usando solo ventas de días anteriores.

Para la fecha D de una serie (tienda, familia) solo se usan ventas con fecha < D: las
observaciones del día anterior llegan antes que la solicitud del día. La promoción de D
viene en la solicitud. El objetivo y las features de nivel se expresan relativos a la
media de los últimos 28 días (`base`), de modo que un cambio de nivel en la demanda se
traslada al pronóstico en lugar de quedar fuera del rango que vio el modelo.
"""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd

from novamercado.catalogos import Catalogos

CATEGORICAS = ["tienda_id", "familia_id", "tipo_tienda"]
NUMERICAS = [
    "en_promocion",
    "dia_semana",
    "dia_mes",
    "feriado",
    "comercial",
    "medio_dia",
    "pago",
    "quincena",
    "log_base",
    "r_lag7",
    "r_lag14",
    "r_mismo_dia_4",
    "r_media_7",
    "r_media_14",
    "cv_28",
    "ceros_28",
]
FEATURES = CATEGORICAS + NUMERICAS


def panel(historia: pd.DataFrame) -> pd.DataFrame:
    """Ventas como matriz fecha × (tienda, familia); días sin dato quedan como NaN."""
    m = historia.pivot_table(
        index="fecha", columns=["tienda_id", "familia_id"], values="ventas", aggfunc="last"
    )
    m.index = pd.to_datetime(m.index)
    return m.asfreq("D").astype(float)


def calendario(fechas: pd.Series, ciudades: pd.Series, cat: Catalogos) -> pd.DataFrame:
    """Indicadores por tipo de fecha especial; un feriado local solo aplica a su ciudad."""
    f = cat.feriados.copy()
    f["fecha"] = pd.to_datetime(f["fecha"])
    salida = {}
    for tipo in ["feriado", "comercial", "medio_dia", "pago"]:
        sub = f[f["tipo"] == tipo]
        nacional = set(sub.loc[sub["ambito"] == "nacional", "fecha"])
        local = set(zip(sub.loc[sub["ambito"] != "nacional", "fecha"],
                        sub.loc[sub["ambito"] != "nacional", "ambito"], strict=True))
        salida[tipo] = np.array(
            [float(d in nacional or (d, c) in local) for d, c in zip(fechas, ciudades, strict=True)]
        )
    return pd.DataFrame(salida, index=fechas.index)


def construir(
    historia: pd.DataFrame, pedidos: pd.DataFrame, cat: Catalogos
) -> pd.DataFrame:
    """Features para cada fila de `pedidos` (fecha, tienda_id, familia_id, en_promocion).

    `historia` trae ventas (fecha, tienda_id, familia_id, ventas); para cada fecha pedida
    solo se usan las ventas de días anteriores.
    """
    m = panel(historia) if len(historia) else pd.DataFrame(index=pd.DatetimeIndex([]))
    pedidos = pedidos.copy()
    pedidos["fecha"] = pd.to_datetime(pedidos["fecha"])
    inicio = pedidos["fecha"].min() - pd.Timedelta(days=29)
    fin = pedidos["fecha"].max()
    if len(m.index):
        inicio, fin = min(m.index.min(), inicio), max(m.index.max(), fin)
    m = m.reindex(pd.date_range(inicio, fin, freq="D"))
    valores = m.to_numpy()
    fila_de_fecha = {d: i for i, d in enumerate(m.index)}
    col_de_serie = {s: j for j, s in enumerate(m.columns)}

    n = len(pedidos)
    ventanas = {"lag7": 7, "lag14": 14}
    salida = {k: np.full(n, np.nan) for k in
              ["base", "lag7", "lag14", "mismo_dia_4", "media_7", "media_14", "std_28", "ceros_28"]}
    for k, (fecha, tienda, familia) in enumerate(
        zip(pedidos["fecha"], pedidos["tienda_id"], pedidos["familia_id"], strict=True)
    ):
        j = col_de_serie.get((tienda, familia))
        if j is None:
            continue
        i = fila_de_fecha[fecha]
        serie = valores[:, j]
        previos = serie[max(0, i - 28):i]  # solo días < fecha
        if np.isfinite(previos).any():
            salida["base"][k] = np.nanmean(previos)
            salida["std_28"][k] = np.nanstd(previos)
            salida["ceros_28"][k] = np.nanmean(previos == 0)
        for nombre, dias in ventanas.items():
            if i - dias >= 0:
                salida[nombre][k] = serie[i - dias]
        mismos = [serie[i - 7 * s] for s in range(1, 5) if i - 7 * s >= 0]
        if np.isfinite(mismos).any():
            salida["mismo_dia_4"][k] = np.nanmean(mismos)
        for nombre, dias in [("media_7", 7), ("media_14", 14)]:
            tramo = serie[max(0, i - dias):i]
            if np.isfinite(tramo).any():
                salida[nombre][k] = np.nanmean(tramo)

    base = salida["base"]
    denom = np.where(np.isfinite(base) & (base > 0), base, np.nan)
    tiendas = cat.tiendas.loc[pedidos["tienda_id"]]
    cal = calendario(pedidos["fecha"], pd.Series(tiendas["ciudad"].to_numpy(), index=pedidos.index), cat)
    df = pd.DataFrame(
        {
            "fecha": pedidos["fecha"].dt.date,
            "tienda_id": pedidos["tienda_id"].astype(int),
            "familia_id": pedidos["familia_id"],
            "tipo_tienda": tiendas["tipo"].to_numpy(),
            "en_promocion": pd.to_numeric(pedidos["en_promocion"], errors="coerce"),
            "dia_semana": pedidos["fecha"].dt.dayofweek,
            "dia_mes": pedidos["fecha"].dt.day,
            "quincena": (
                (pedidos["fecha"].dt.day == 15) | pedidos["fecha"].dt.is_month_end
            ).astype(float),
            "base": base,
            "log_base": np.log1p(base),
            "r_lag7": salida["lag7"] / denom,
            "r_lag14": salida["lag14"] / denom,
            "r_mismo_dia_4": salida["mismo_dia_4"] / denom,
            "r_media_7": salida["media_7"] / denom,
            "r_media_14": salida["media_14"] / denom,
            "cv_28": salida["std_28"] / denom,
            "ceros_28": salida["ceros_28"],
            "mismo_dia_4": salida["mismo_dia_4"],
        },
        index=pedidos.index,
    )
    return pd.concat([df, cal], axis=1)


def filas_entrenamiento(historia: pd.DataFrame, desde: date, hasta: date) -> pd.DataFrame:
    """Pedidos 'históricos': cada (fecha, tienda, familia) con venta conocida en el rango."""
    h = historia[(historia["fecha"] >= desde) & (historia["fecha"] <= hasta)]
    return h[["fecha", "tienda_id", "familia_id", "en_promocion", "ventas"]].reset_index(drop=True)
