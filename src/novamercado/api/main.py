"""API de NovaMercado (FastAPI). Los endpoints solo orquestan: la limpieza vive en
`novamercado.calidad`, las features en `novamercado.features` y los modelos en
`novamercado.modelo`."""

from __future__ import annotations

import json
import logging
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any

import pandas as pd
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from novamercado.airflow_cliente import ClienteAirflow, ErrorAirflow
from novamercado.api.esquemas import (
    EstadoReentrenamiento,
    RespuestaModelo,
    RespuestaObservaciones,
    RespuestaPrediccion,
    RespuestaReentrenar,
    RespuestaSalud,
    SolicitudPrediccion,
)
from novamercado.api.servicio import GestorModelo, MonitorSalud
from novamercado.calidad import (
    Rechazo,
    ResolutorFamilias,
    normalizar_observaciones,
    normalizar_prediccion,
)
from novamercado.catalogos import Catalogos
from novamercado.features import construir

log = logging.getLogger("novamercado.api")


@dataclass
class Contexto:
    catalogos: Catalogos
    familias: ResolutorFamilias
    repositorio: Any
    gestor: GestorModelo
    monitor: MonitorSalud
    airflow: Any


def rechazo(detalle: str) -> JSONResponse:
    return JSONResponse(status_code=422, content={"detail": detalle})


def crear_app(fabrica_contexto) -> FastAPI:
    """`fabrica_contexto()` arma las dependencias (reales en producción, falsas en pruebas)."""

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.ctx = fabrica_contexto()
        yield

    app = FastAPI(title="NovaMercado · pronóstico diario de ventas", version="1.0",
                  lifespan=lifespan)

    def ctx(request: Request) -> Contexto:
        return request.app.state.ctx

    async def cuerpo_json(request: Request) -> Any:
        try:
            return json.loads(await request.body() or b"null")
        except (json.JSONDecodeError, UnicodeDecodeError) as e:
            raise Rechazo(f"El cuerpo no es JSON válido: {e}") from e

    @app.exception_handler(Rechazo)
    async def _rechazo(request: Request, exc: Rechazo):
        return rechazo(str(exc))

    @app.post("/v1/prediccion", response_model=RespuestaPrediccion)
    async def prediccion(request: Request):
        c = ctx(request)
        cuerpo = await cuerpo_json(request)
        try:
            pedido, reg = normalizar_prediccion(cuerpo, c.catalogos, c.familias)
        except Rechazo as e:
            _registrar_rechazo(c, "prediccion", cuerpo, str(e))
            raise
        solicitud = SolicitudPrediccion(
            tienda_id=pedido.tienda_id, fecha=pedido.fecha,
            familias=[{"familia_id": f, "en_promocion": p} for f, p in pedido.familias.items()],
        )
        modelo, version = c.gestor.actual()
        if modelo is None:
            return JSONResponse(status_code=503, content={"detail": "Aún no hay un modelo cargado."})
        try:
            historia = c.repositorio.historia(solicitud.tienda_id, solicitud.fecha)
        except Exception as e:
            log.exception("Base de datos no disponible")
            return JSONResponse(status_code=503, content={"detail": f"Base de datos no disponible: {e}"})
        pedidos = pd.DataFrame(
            [{"fecha": solicitud.fecha, "tienda_id": solicitud.tienda_id,
              "familia_id": f.familia_id, "en_promocion": f.en_promocion}
             for f in solicitud.familias]
        )
        salida = modelo.predecir(construir(historia, pedidos, c.catalogos))
        filas = [
            {"familia_id": f.familia_id, "en_promocion": f.en_promocion,
             "venta_esperada": float(e), "cantidad_sugerida": int(s)}
            for f, e, s in zip(solicitud.familias, salida["venta_esperada"],
                               salida["cantidad_sugerida"], strict=True)
        ]
        respuesta = RespuestaPrediccion(
            tienda_id=solicitud.tienda_id, fecha=solicitud.fecha, version_modelo=version,
            predicciones=[{k: f[k] for k in ("familia_id", "venta_esperada", "cantidad_sugerida")}
                          for f in filas],
            calidad=reg.como_dict(),
        )
        try:
            c.repositorio.registrar_prediccion(cuerpo, respuesta.model_dump(), filas)
        except Exception:
            log.exception("No se pudo registrar la predicción")
        return respuesta

    @app.post("/v1/observaciones", response_model=RespuestaObservaciones)
    async def observaciones(request: Request):
        c = ctx(request)
        cuerpo = await cuerpo_json(request)
        try:
            obs, reg = normalizar_observaciones(cuerpo, c.catalogos, c.familias)
        except Rechazo as e:
            _registrar_rechazo(c, "observaciones", cuerpo, str(e))
            raise
        try:
            previas = c.repositorio.ventas_previas(obs.tienda_id, obs.fecha)
            for familia, ventas in obs.ventas.items():
                if familia in previas and previas[familia] != ventas:
                    reg.advertir(
                        f"ventas.{familia}",
                        f"ya había {previas[familia]} registradas para {obs.fecha.isoformat()}; "
                        f"se reemplazó por {ventas} (se asume que el último envío corrige)",
                    )
            calidad = reg.como_dict()
            guardados = c.repositorio.guardar_observaciones(
                cuerpo, obs.tienda_id, obs.fecha, obs.ventas, calidad
            )
        except Exception as e:
            log.exception("Base de datos no disponible")
            return JSONResponse(status_code=503, content={"detail": f"Base de datos no disponible: {e}"})
        return RespuestaObservaciones(registros_guardados=guardados, calidad=calidad)

    @app.post("/admin/reentrenar", status_code=202, response_model=RespuestaReentrenar)
    async def reentrenar(request: Request):
        c = ctx(request)
        try:
            run_id = c.airflow.disparar()
            c.repositorio.crear_reentrenamiento(run_id)
        except Exception as e:
            return JSONResponse(status_code=503,
                                content={"detail": f"No se pudo iniciar el reentrenamiento: {e}"})
        return RespuestaReentrenar(id=run_id)

    @app.get("/admin/reentrenar/{id_}", response_model=EstadoReentrenamiento,
             response_model_exclude_none=True)
    async def estado_reentrenamiento(id_: str, request: Request):
        c = ctx(request)
        fila = c.repositorio.leer_reentrenamiento(id_)
        try:
            estado = c.airflow.estado(id_)
        except (ErrorAirflow, Exception):
            estado = fila["estado"] if fila else None
        if estado is None and fila is None:
            return JSONResponse(status_code=404, content={"detail": f"Ejecución '{id_}' desconocida."})
        terminado = fila is not None and fila.get("terminado_en") is not None
        if estado == "en_curso" or (estado is None and not terminado):
            return EstadoReentrenamiento(estado="en_curso")
        if not terminado:  # Airflow ya terminó pero el resultado aún no se escribió
            _, version = c.gestor.actual()
            if estado == "exitoso":
                return EstadoReentrenamiento(estado="en_curso")
            return EstadoReentrenamiento(estado="fallido", promovido=False, version=version,
                                         motivo="La ejecución falló antes de registrar un resultado.")
        return EstadoReentrenamiento(
            estado=fila["estado"], promovido=bool(fila["promovido"]),
            version=fila["version"], motivo=fila["motivo"],
        )

    @app.get("/modelo", response_model=RespuestaModelo)
    async def modelo(request: Request):
        c = ctx(request)
        _, version = c.gestor.actual()
        if version is None:
            return JSONResponse(status_code=503, content={"detail": "Aún no hay un modelo cargado."})
        return RespuestaModelo(nombre=c.gestor.nombre, alias=c.gestor.alias, version=version)

    @app.get("/health", response_model=RespuestaSalud, response_model_exclude_none=True)
    async def health(request: Request):
        c = ctx(request)
        modelo_actual, version = c.gestor.actual()
        dependencias = dict(c.monitor.estado) | {"modelo": modelo_actual is not None}
        if modelo_actual is None:
            return JSONResponse(
                status_code=503,
                content=RespuestaSalud(estado="iniciando", motivo="Aún no hay un modelo cargado.",
                                       dependencias=dependencias).model_dump(exclude_none=True),
            )
        caidas = [n for n, ok in dependencias.items() if not ok]
        if caidas:
            return RespuestaSalud(estado="degradado", version_modelo=version,
                                  motivo="No disponible: " + ", ".join(caidas),
                                  dependencias=dependencias)
        return RespuestaSalud(estado="ok", version_modelo=version, dependencias=dependencias)

    return app


def _registrar_rechazo(c: Contexto, tipo: str, cuerpo: Any, detalle: str) -> None:
    try:
        c.repositorio.registrar_rechazo(tipo, cuerpo, detalle)
    except Exception:
        log.exception("No se pudo registrar el rechazo")


def contexto_produccion() -> Contexto:
    """Dependencias reales, a partir de las variables de entorno de Docker Compose."""
    import mlflow
    from mlflow.tracking import MlflowClient

    from novamercado.api.repositorio import RepositorioPostgres
    from novamercado.api.servicio import http_ok
    from novamercado.catalogos import cargar_catalogos
    from novamercado.config import entorno, params
    from novamercado.registro import cargar_version, version_vigente

    env, p = entorno(), params()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    mlflow.set_tracking_uri(env.mlflow_uri)
    cat = cargar_catalogos(env.datos)
    s = p["servicio"]
    repo = RepositorioPostgres(env.database_url)
    try:
        repo.preparar()
    except Exception:
        log.exception("La base de datos aún no está disponible")
    nombre, alias = p["modelo"]["nombre"], p["modelo"]["alias"]
    gestor = GestorModelo(
        nombre, alias, env.cache_modelo,
        consultar_version=lambda: version_vigente(MlflowClient(), nombre, alias),
        descargar=lambda v: cargar_version(nombre, v),
    )
    gestor.cargar_cache()
    gestor.iniciar(s["segundos_recarga_modelo"])
    airflow = ClienteAirflow(env.airflow_url, env.airflow_usuario, env.airflow_clave,
                             timeout=s["segundos_timeout_dependencias"])
    monitor = MonitorSalud({
        "base_datos": repo.ping,
        "mlflow": http_ok(f"{env.mlflow_uri}/health", s["segundos_timeout_dependencias"]),
        "airflow": airflow.sano,
    })
    monitor.revisar()
    monitor.iniciar(s["segundos_chequeo_salud"])
    return Contexto(cat, ResolutorFamilias(cat, s["similitud_reparar"], s["similitud_advertir"]),
                    repo, gestor, monitor, airflow)


app = crear_app(contexto_produccion)
