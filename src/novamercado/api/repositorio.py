"""Acceso a datos que usa la API (PostgreSQL); las pruebas lo reemplazan por uno en memoria."""

from __future__ import annotations

import json
from datetime import date
from typing import Any

import pandas as pd

from novamercado import db


class RepositorioPostgres:
    def __init__(self, url: str):
        self.url = url

    def preparar(self) -> None:
        with db.conexion(self.url) as conn:
            db.crear_esquema(conn)

    def ping(self) -> bool:
        with db.conexion(self.url, timeout=3) as conn, conn.cursor() as cur:
            cur.execute("SELECT 1")
            return True

    def historia(self, tienda_id: int, fecha: date) -> pd.DataFrame:
        with db.conexion(self.url) as conn:
            return db.historia_para_prediccion(conn, tienda_id, fecha)

    def registrar_prediccion(self, cuerpo: Any, respuesta: dict, filas: list[dict]) -> None:
        with db.conexion(self.url) as conn:
            sid = db.registrar_solicitud(conn, "prediccion", cuerpo, 200, respuesta["calidad"],
                                         respuesta)
            db.registrar_predicciones(conn, sid, respuesta["tienda_id"], respuesta["fecha"],
                                      filas, respuesta["version_modelo"])

    def registrar_rechazo(self, tipo: str, cuerpo: Any, detalle: str) -> None:
        with db.conexion(self.url) as conn:
            db.registrar_solicitud(conn, tipo, cuerpo, 422, None, {"detail": detalle})

    def ventas_previas(self, tienda_id: int, fecha: date) -> dict[str, int]:
        with db.conexion(self.url) as conn:
            return db.ventas_previas(conn, tienda_id, fecha)

    def guardar_observaciones(self, cuerpo: Any, tienda_id: int, fecha: date,
                              ventas: dict[str, int], calidad: dict) -> int:
        with db.conexion(self.url) as conn:
            sid = db.registrar_solicitud(conn, "observaciones", cuerpo, 200, calidad, None)
            guardados = db.guardar_observaciones(conn, sid, tienda_id, fecha, ventas)
            with conn.cursor() as cur:
                cur.execute("UPDATE solicitudes SET respuesta = %s WHERE id = %s",
                            (json.dumps({"registros_guardados": guardados,
                                               "calidad": calidad}), sid))
            return guardados

    def crear_reentrenamiento(self, id_: str) -> None:
        with db.conexion(self.url) as conn:
            db.crear_reentrenamiento(conn, id_)

    def leer_reentrenamiento(self, id_: str) -> dict | None:
        with db.conexion(self.url) as conn:
            return db.leer_reentrenamiento(conn, id_)
