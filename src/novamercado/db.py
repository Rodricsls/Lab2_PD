"""Base de datos del producto (PostgreSQL): esquema, carga inicial y consultas."""

from __future__ import annotations

import io
import json
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import pandas as pd
import psycopg2
import psycopg2.extras

ESQUEMA = """
CREATE TABLE IF NOT EXISTS ventas_historicas (
    fecha date NOT NULL, tienda_id int NOT NULL, familia_id text NOT NULL,
    ventas int NOT NULL, en_promocion int,
    PRIMARY KEY (fecha, tienda_id, familia_id)
);
CREATE TABLE IF NOT EXISTS solicitudes (
    id bigserial PRIMARY KEY, recibida_en timestamptz NOT NULL DEFAULT now(),
    tipo text NOT NULL, cuerpo jsonb, codigo int, estado_calidad text, respuesta jsonb
);
CREATE TABLE IF NOT EXISTS correcciones (
    id bigserial PRIMARY KEY, solicitud_id bigint REFERENCES solicitudes(id),
    campo text, accion text, detalle text
);
CREATE TABLE IF NOT EXISTS predicciones (
    id bigserial PRIMARY KEY, solicitud_id bigint REFERENCES solicitudes(id),
    tienda_id int, fecha date, familia_id text, en_promocion int,
    venta_esperada double precision, cantidad_sugerida int, version_modelo text
);
CREATE INDEX IF NOT EXISTS predicciones_clave ON predicciones (tienda_id, fecha, familia_id);
CREATE TABLE IF NOT EXISTS observaciones (
    fecha date NOT NULL, tienda_id int NOT NULL, familia_id text NOT NULL,
    ventas int NOT NULL, solicitud_id bigint REFERENCES solicitudes(id),
    actualizada_en timestamptz NOT NULL DEFAULT now(),
    PRIMARY KEY (fecha, tienda_id, familia_id)
);
CREATE TABLE IF NOT EXISTS reentrenamientos (
    id text PRIMARY KEY, solicitado_en timestamptz NOT NULL DEFAULT now(),
    estado text NOT NULL, promovido boolean, version text, motivo text,
    terminado_en timestamptz
);
"""


@contextmanager
def conexion(url: str, timeout: int = 5) -> Iterator[Any]:
    conn = psycopg2.connect(url, connect_timeout=timeout)
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def crear_esquema(conn) -> None:
    with conn.cursor() as cur:
        cur.execute("SELECT pg_advisory_xact_lock(42)")  # evita carreras entre servicios
        cur.execute(ESQUEMA)


def cargar_historico(conn, archivo: Path) -> int:
    """Carga historico_ventas.csv si la tabla está vacía (idempotente)."""
    with conn.cursor() as cur:
        cur.execute("SELECT pg_advisory_xact_lock(43)")
        cur.execute("SELECT count(*) FROM ventas_historicas")
        if cur.fetchone()[0] > 0:
            return 0
        df = pd.read_csv(archivo)[["fecha", "tienda_id", "familia_id", "ventas", "en_promocion"]]
        buffer = io.StringIO()
        df.to_csv(buffer, index=False, header=False)
        buffer.seek(0)
        cur.copy_expert(
            "COPY ventas_historicas (fecha, tienda_id, familia_id, ventas, en_promocion) "
            "FROM STDIN WITH CSV",
            buffer,
        )
        return len(df)


def _df(cur, sql: str, args=()) -> pd.DataFrame:
    cur.execute(sql, args)
    columnas = [c.name for c in cur.description]
    return pd.DataFrame(cur.fetchall(), columns=columnas)


def historia(conn, tienda_id: int | None = None, desde: date | None = None,
             hasta: date | None = None) -> pd.DataFrame:
    """Ventas conocidas: histórico publicado + observaciones recibidas (estas prevalecen).

    La promoción de un día observado se toma de la última solicitud de pronóstico de ese
    día, porque las observaciones no la incluyen.
    """
    filtros, args = [], []
    if tienda_id is not None:
        filtros.append("tienda_id = %s")
        args.append(tienda_id)
    if desde is not None:
        filtros.append("fecha >= %s")
        args.append(desde)
    if hasta is not None:
        filtros.append("fecha <= %s")
        args.append(hasta)
    donde = ("WHERE " + " AND ".join(filtros)) if filtros else ""
    with conn.cursor() as cur:
        hist = _df(cur, f"SELECT fecha, tienda_id, familia_id, ventas, en_promocion "
                        f"FROM ventas_historicas {donde}", args)
        obs = _df(cur, f"""
            SELECT o.fecha, o.tienda_id, o.familia_id, o.ventas, p.en_promocion
            FROM observaciones o
            LEFT JOIN LATERAL (
                SELECT en_promocion FROM predicciones p
                WHERE p.tienda_id = o.tienda_id AND p.fecha = o.fecha
                  AND p.familia_id = o.familia_id
                ORDER BY p.id DESC LIMIT 1
            ) p ON true
            {donde.replace('tienda_id', 'o.tienda_id').replace('fecha', 'o.fecha')}""", args)
    todo = pd.concat([hist.assign(_p=0), obs.assign(_p=1)], ignore_index=True)
    todo = todo.sort_values("_p").drop_duplicates(["fecha", "tienda_id", "familia_id"], keep="last")
    todo["ventas"] = todo["ventas"].astype(float)
    todo["en_promocion"] = pd.to_numeric(todo["en_promocion"], errors="coerce")
    return todo.drop(columns="_p").sort_values(["fecha", "tienda_id", "familia_id"]).reset_index(
        drop=True
    )


def historia_para_prediccion(conn, tienda_id: int, fecha: date) -> pd.DataFrame:
    return historia(conn, tienda_id, fecha - timedelta(days=35), fecha - timedelta(days=1))


def registrar_solicitud(conn, tipo: str, cuerpo: Any, codigo: int, calidad: dict | None,
                        respuesta: Any) -> int:
    with conn.cursor() as cur:
        cur.execute(
            "INSERT INTO solicitudes (tipo, cuerpo, codigo, estado_calidad, respuesta) "
            "VALUES (%s, %s, %s, %s, %s) RETURNING id",
            (tipo, json.dumps(cuerpo, default=str), codigo,
             calidad["estado"] if calidad else None, json.dumps(respuesta, default=str)),
        )
        solicitud_id = cur.fetchone()[0]
        if calidad:
            psycopg2.extras.execute_values(
                cur,
                "INSERT INTO correcciones (solicitud_id, campo, accion, detalle) VALUES %s",
                [(solicitud_id, o["campo"], o["accion"], o["detalle"])
                 for o in calidad["observaciones"]],
            )
    return solicitud_id


def registrar_predicciones(conn, solicitud_id: int, tienda_id: int, fecha: date,
                           filas: list[dict], version: str) -> None:
    with conn.cursor() as cur:
        psycopg2.extras.execute_values(
            cur,
            "INSERT INTO predicciones (solicitud_id, tienda_id, fecha, familia_id, en_promocion, "
            "venta_esperada, cantidad_sugerida, version_modelo) VALUES %s",
            [(solicitud_id, tienda_id, fecha, f["familia_id"], f["en_promocion"],
              f["venta_esperada"], f["cantidad_sugerida"], version) for f in filas],
        )


def ventas_previas(conn, tienda_id: int, fecha: date) -> dict[str, int]:
    with conn.cursor() as cur:
        cur.execute("SELECT familia_id, ventas FROM observaciones WHERE tienda_id=%s AND fecha=%s",
                    (tienda_id, fecha))
        return dict(cur.fetchall())


def guardar_observaciones(conn, solicitud_id: int, tienda_id: int, fecha: date,
                          ventas: dict[str, int]) -> int:
    """Upsert idempotente: devuelve cuántos registros quedaron nuevos o modificados."""
    if not ventas:
        return 0
    with conn.cursor() as cur:
        filas = psycopg2.extras.execute_values(
            cur,
            """INSERT INTO observaciones (fecha, tienda_id, familia_id, ventas, solicitud_id)
               VALUES %s
               ON CONFLICT (fecha, tienda_id, familia_id) DO UPDATE
                 SET ventas = EXCLUDED.ventas, solicitud_id = EXCLUDED.solicitud_id,
                     actualizada_en = now()
                 WHERE observaciones.ventas IS DISTINCT FROM EXCLUDED.ventas
               RETURNING familia_id""",
            [(fecha, tienda_id, f, v, solicitud_id) for f, v in ventas.items()],
            fetch=True,
        )
    return len(filas)


def crear_reentrenamiento(conn, id_: str) -> None:
    with conn.cursor() as cur:
        cur.execute("INSERT INTO reentrenamientos (id, estado) VALUES (%s, 'en_curso') "
                    "ON CONFLICT (id) DO NOTHING", (id_,))


def terminar_reentrenamiento(conn, id_: str, estado: str, promovido: bool, version: str | None,
                             motivo: str) -> None:
    with conn.cursor() as cur:
        cur.execute(
            """INSERT INTO reentrenamientos (id, estado, promovido, version, motivo, terminado_en)
               VALUES (%s, %s, %s, %s, %s, now())
               ON CONFLICT (id) DO UPDATE SET estado=EXCLUDED.estado,
                 promovido=EXCLUDED.promovido, version=EXCLUDED.version,
                 motivo=EXCLUDED.motivo, terminado_en=now()""",
            (id_, estado, promovido, version, motivo),
        )


def leer_reentrenamiento(conn, id_: str) -> dict | None:
    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute("SELECT * FROM reentrenamientos WHERE id = %s", (id_,))
        fila = cur.fetchone()
    return dict(fila) if fila else None
