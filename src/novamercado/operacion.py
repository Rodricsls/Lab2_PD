"""Operaciones de punta a punta: arranque desde cero y reentrenamiento.

`inicializar` corre en el servicio `inicializador` del compose cada vez que se levanta el
producto: crea el esquema, carga el histórico si la base está vacía y, si el Registry no
tiene modelo vigente, entrena, registra y promueve el primero. Al volver a levantar el
producto no repite ese trabajo.

`reentrenar` es lo que ejecuta el DAG: entrena con el histórico y las observaciones
registradas, aplica el criterio de reemplazo y solo entonces mueve el alias.
"""

from __future__ import annotations

import logging
import time
from typing import Any

import mlflow
from mlflow.tracking import MlflowClient

from novamercado import db, registro
from novamercado.catalogos import cargar_catalogos
from novamercado.config import entorno, params
from novamercado.entrenamiento import Tabla, ciclo, entrenar

log = logging.getLogger("novamercado.operacion")


def _plano(medidas: dict[str, dict[str, float]]) -> dict[str, float]:
    return {f"{grupo}_{k}": v for grupo, m in medidas.items() for k, v in m.items()}


def _esperar(nombre: str, funcion, intentos: int = 60, pausa: float = 5) -> None:
    for i in range(intentos):
        try:
            funcion()
            return
        except Exception as e:
            log.info("Esperando %s (%d/%d): %s", nombre, i + 1, intentos, e)
            time.sleep(pausa)
    raise RuntimeError(f"{nombre} no estuvo disponible a tiempo")


def registrar_ciclo(resultado, historia, cat, p, etiquetas: dict[str, str]) -> str:
    """Registra el challenger como versión nueva, con métricas de la comparación."""
    m = p["modelo"]
    inicio, fin = resultado.ventana
    parametros = {
        "modelo.tipo": resultado.challenger.tipo,
        "ventana_validacion": f"{inicio}..{fin}",
        "datos_hasta": resultado.challenger.info["datos_hasta"],
        "dias_calibracion": p["entrenamiento"]["dias_calibracion"],
        "tolerancia": p["promocion"]["tolerancia"],
        **{f"lgbm.{k}": v for k, v in p["entrenamiento"]["lightgbm"].items()},
    }
    tags = {
        "huella_datos": registro.huella(historia),
        "datos_hasta": resultado.challenger.info["datos_hasta"],
        "filas_entrenamiento": str(resultado.challenger.info["filas"]),
        "decision_propuesta": "promover" if resultado.promovido else "descartar",
        "motivo": resultado.motivo,
        **etiquetas,
    }
    # El baseline queda como run propio: es el piso de comparación de cada ciclo.
    registro.registrar_run(
        resultado.candidatos_holdout["baseline"], "baseline", m["experimento"],
        {"modelo.tipo": "baseline", "ventana_validacion": f"{inicio}..{fin}"},
        {f"validacion_{k}": v for k, v in resultado.metricas["baseline_validacion"].items()},
        {"huella_datos": tags["huella_datos"], **etiquetas},
    )
    _, version = registro.registrar_run(
        resultado.challenger, f"lightgbm-{fin}", m["experimento"], parametros,
        _plano(resultado.metricas), tags, nombre_modelo=m["nombre"],
    )
    return version


def inicializar() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    env, p = entorno(), params()
    mlflow.set_tracking_uri(env.mlflow_uri)
    cliente = MlflowClient()

    def preparar_db():
        with db.conexion(env.database_url) as conn:
            db.crear_esquema(conn)
            n = db.cargar_historico(conn, env.datos / "historico_ventas.csv")
            log.info("Histórico: %s filas cargadas", n if n else "ya estaba cargado, 0")

    _esperar("la base de datos", preparar_db)
    _esperar("MLflow", lambda: cliente.search_registered_models(max_results=1))

    m = p["modelo"]
    vigente = registro.version_vigente(cliente, m["nombre"], m["alias"])
    if vigente:
        log.info("Ya existe el modelo vigente (versión %s): no se entrena de nuevo", vigente)
        return

    cat = cargar_catalogos(env.datos)
    with db.conexion(env.database_url) as conn:
        historia = db.historia(conn)
    resultado = ciclo(historia, cat, p, vigente=None)
    if not resultado.promovido:  # el ML no superó al baseline: el baseline es el primer vigente
        tabla = Tabla.desde_historia(historia, cat)
        resultado.challenger = entrenar("baseline", tabla, cat, p, tabla.ultima_fecha)
    version = registrar_ciclo(resultado, historia, cat, p, {"origen": "arranque"})
    registro.promover(cliente, m["nombre"], m["alias"], version, resultado.motivo)
    log.info("Primer modelo registrado y promovido: versión %s (%s)", version, resultado.motivo)


def reentrenar(run_id: str) -> dict[str, Any]:
    """Entrena el challenger y lo registra; devuelve la propuesta de decisión."""
    env, p = entorno(), params()
    mlflow.set_tracking_uri(env.mlflow_uri)
    cliente = MlflowClient()
    m = p["modelo"]
    cat = cargar_catalogos(env.datos)
    with db.conexion(env.database_url) as conn:
        historia = db.historia(conn)
    version_actual = registro.version_vigente(cliente, m["nombre"], m["alias"])
    vigente = registro.cargar_version(m["nombre"], version_actual) if version_actual else None
    resultado = ciclo(historia, cat, p, vigente)
    version = registrar_ciclo(resultado, historia, cat, p,
                              {"origen": "reentrenamiento", "airflow_run_id": run_id,
                               "vigente_al_comparar": version_actual or "ninguno"})
    return {"version_challenger": version, "promovido": resultado.promovido,
            "motivo": resultado.motivo, "vigente_anterior": version_actual}


def aplicar_decision(propuesta: dict[str, Any]) -> dict[str, Any]:
    env, p = entorno(), params()
    mlflow.set_tracking_uri(env.mlflow_uri)
    cliente = MlflowClient()
    m = p["modelo"]
    version = propuesta["version_challenger"]
    if propuesta["promovido"]:
        registro.promover(cliente, m["nombre"], m["alias"], version, propuesta["motivo"])
    else:
        registro.descartar(cliente, m["nombre"], version, propuesta["motivo"])
    vigente = registro.version_vigente(cliente, m["nombre"], m["alias"])
    return {"promovido": propuesta["promovido"], "version": vigente, "motivo": propuesta["motivo"]}


def registrar_resultado(run_id: str, resultado: dict[str, Any] | None, error: str | None) -> None:
    env, p = entorno(), params()
    vigente = None
    try:
        mlflow.set_tracking_uri(env.mlflow_uri)
        vigente = registro.version_vigente(MlflowClient(), p["modelo"]["nombre"], p["modelo"]["alias"])
    except Exception:
        log.exception("No se pudo consultar el modelo vigente")
    with db.conexion(env.database_url) as conn:
        if resultado is None:
            db.terminar_reentrenamiento(conn, run_id, "fallido", False, vigente,
                                        f"El reentrenamiento falló; el modelo vigente no cambió. {error or ''}".strip())
        else:
            db.terminar_reentrenamiento(conn, run_id, "exitoso", resultado["promovido"],
                                        resultado["version"], resultado["motivo"])


if __name__ == "__main__":
    inicializar()
