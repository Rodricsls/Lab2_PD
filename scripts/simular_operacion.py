"""Simula la operación diaria contra la API levantada (evidencia y prueba de punta a punta).

Para cada día posterior al histórico: envía las observaciones del día anterior y luego la
solicitud de pronóstico de cada tienda (algunas con errores tomados del catálogo y otros
nuevos). Las "ventas reales" son sintéticas: las del mismo día de la semana un año antes
con un cambio de nivel, para comprobar que el reentrenamiento las incorpora. Al final
dispara un reentrenamiento y espera su resultado.

    python scripts/simular_operacion.py --datos datos/publicados --dias 10 --salida docs/evidencia
"""

from __future__ import annotations

import argparse
import copy
import json
import random
import time
from datetime import date, timedelta
from pathlib import Path

import httpx
import pandas as pd

API = "http://localhost:8000"


def defectuosa(cuerpo: dict, rng: random.Random) -> tuple[str, dict]:
    """Aplica al azar un problema; varios no están en el catálogo publicado."""
    c = copy.deepcopy(cuerpo)
    tipo = rng.choice(["tienda_prefijo", "familia_mayusculas", "promo_texto", "fecha_dmy",
                       "faltan_familias", "campo_renombrado", "fecha_iso_hora",
                       "familia_con_espacios", "promo_decimal_coma", "campo_extra"])
    if tipo == "tienda_prefijo":
        c["tienda_id"] = f"T{c['tienda_id']:02d}"
    elif tipo == "familia_mayusculas":
        c["familias"][2]["familia_id"] = c["familias"][2]["familia_id"].upper()
    elif tipo == "promo_texto":
        c["familias"][1]["en_promocion"] = str(c["familias"][1]["en_promocion"])
    elif tipo == "fecha_dmy":
        d = date.fromisoformat(c["fecha"])
        c["fecha"] = d.strftime("%d/%m/%Y")
    elif tipo == "faltan_familias":
        c["familias"] = c["familias"][:-2]
    elif tipo == "campo_renombrado":
        c["store_id"] = c.pop("tienda_id")
    elif tipo == "fecha_iso_hora":
        c["fecha"] = c["fecha"] + "T00:00:00Z"
    elif tipo == "familia_con_espacios":
        c["familias"][0]["familia_id"] = "  " + c["familias"][0]["familia_id"].title() + " "
    elif tipo == "promo_decimal_coma":
        c["familias"][3]["en_promocion"] = "2,0"
    elif tipo == "campo_extra":
        c["canal"] = "pos"
    return tipo, c


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--datos", type=Path, default=Path("datos/publicados"))
    ap.add_argument("--dias", type=int, default=10)
    ap.add_argument("--cambio", type=float, default=1.15, help="multiplicador de la demanda")
    ap.add_argument("--salida", type=Path, default=Path("docs/evidencia"))
    args = ap.parse_args()
    rng = random.Random(7)
    args.salida.mkdir(parents=True, exist_ok=True)

    hist = pd.read_csv(args.datos / "historico_ventas.csv", parse_dates=["fecha"])
    tiendas = pd.read_csv(args.datos / "tiendas.csv")["tienda_id"].tolist()
    inicio = hist["fecha"].max().date() + timedelta(days=1)
    cliente = httpx.Client(base_url=API, timeout=30)
    ejemplos, resumen = {}, {"dias": [], "problemas": {}}

    for k in range(args.dias):
        hoy = inicio + timedelta(days=k)
        ayer = hoy - timedelta(days=1)
        if ayer >= inicio:  # ventas "reales" del día anterior (sintéticas)
            base = hist[hist["fecha"] == pd.Timestamp(ayer - timedelta(days=364))]
            for t in tiendas:
                filas = base[base["tienda_id"] == t]
                if filas.empty:  # tienda que hace un año no existía: semana más reciente
                    mismo = hist[(hist["tienda_id"] == t) & (hist["fecha"].dt.dayofweek == ayer.weekday())]
                    filas = mismo[mismo["fecha"] == mismo["fecha"].max()]
                cuerpo = {"tienda_id": t, "fecha": ayer.isoformat(), "ventas": [
                    {"familia_id": f, "ventas": int(round(v * args.cambio))}
                    for f, v in zip(filas["familia_id"], filas["ventas"], strict=True)]}
                r = cliente.post("/v1/observaciones", json=cuerpo)
                r.raise_for_status()
                if k == 1 and t == 1:  # reenvío idéntico: debe guardar 0
                    ejemplos["observacion_valida"] = (cuerpo, r.json())
                    ejemplos["observacion_reenviada"] = (cuerpo, cliente.post("/v1/observaciones", json=cuerpo).json())
        costo_dia = 0
        for t in tiendas:
            familias = sorted(hist["familia_id"].unique())
            cuerpo = {"tienda_id": t, "fecha": hoy.isoformat(), "familias": [
                {"familia_id": f, "en_promocion": rng.choice([0, 0, 0, 2, 5])} for f in familias]}
            tipo = "ninguno"
            if rng.random() < 0.4:
                tipo, cuerpo = defectuosa(cuerpo, rng)
            r = cliente.post("/v1/prediccion", json=cuerpo)
            resumen["problemas"].setdefault(tipo, {}).setdefault(str(r.status_code), 0)
            resumen["problemas"][tipo][str(r.status_code)] += 1
            if tipo not in ejemplos:
                ejemplos[f"prediccion_{tipo}"] = (cuerpo, r.json())
            if r.status_code == 200:
                costo_dia += len(r.json()["predicciones"])
        resumen["dias"].append({"fecha": hoy.isoformat(), "predicciones": costo_dia})
        print(f"{hoy}: {costo_dia} predicciones")

    # Rechazo claro
    malo = {"tienda_id": 99, "fecha": inicio.isoformat(), "familias": [{"familia_id": "abarrotes", "en_promocion": 0}]}
    r = cliente.post("/v1/prediccion", json=malo)
    ejemplos["prediccion_rechazada"] = (malo, {"status": r.status_code, **r.json()})

    r = cliente.post("/admin/reentrenar")
    run_id = r.json()["id"]
    print("reentrenamiento", r.status_code, run_id)
    t0 = time.monotonic()
    while True:
        estado = cliente.get(f"/admin/reentrenar/{run_id}").json()
        if estado["estado"] != "en_curso":
            break
        time.sleep(5)
    estado["segundos"] = round(time.monotonic() - t0)
    print("resultado", estado)
    resumen["reentrenamiento"] = {"id": run_id, **estado}

    with open(args.salida / "ejemplos_api.json", "w", encoding="utf-8") as f:
        json.dump({k: {"solicitud": s, "respuesta": rr} for k, (s, rr) in ejemplos.items()}, f,
                  indent=2, ensure_ascii=False)
    with open(args.salida / "simulacion.json", "w", encoding="utf-8") as f:
        json.dump(resumen, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
