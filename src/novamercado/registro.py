"""MLflow: runs con linaje, Model Registry y alias del modelo vigente."""

from __future__ import annotations

import hashlib
import os
from typing import Any

import mlflow
import mlflow.pyfunc
import pandas as pd
from mlflow.exceptions import MlflowException
from mlflow.tracking import MlflowClient

from novamercado.modelo import ModeloVentas


class ModeloPyfunc(mlflow.pyfunc.PythonModel):
    """Envoltura pyfunc: el Registry guarda el modelo con sus factores por familia."""

    def __init__(self, modelo: ModeloVentas):
        self.modelo = modelo

    def predict(self, context, model_input: pd.DataFrame, params=None) -> pd.DataFrame:
        return self.modelo.predecir(model_input)


def huella(historia: pd.DataFrame) -> str:
    texto = historia[["fecha", "tienda_id", "familia_id", "ventas"]].to_csv(index=False)
    return hashlib.md5(texto.encode()).hexdigest()


def registrar_run(
    modelo: ModeloVentas,
    nombre_run: str,
    experimento: str,
    parametros: dict[str, Any],
    metricas: dict[str, float],
    etiquetas: dict[str, str],
    nombre_modelo: str | None = None,
) -> tuple[str, str | None]:
    """Crea un run con parámetros, métricas y linaje; si `nombre_modelo`, registra una versión."""
    mlflow.set_experiment(experimento)
    with mlflow.start_run(run_name=nombre_run) as run:
        mlflow.log_params({k: str(v)[:500] for k, v in parametros.items()})
        mlflow.log_metrics(metricas)
        mlflow.set_tags(etiquetas | {"git_commit": os.environ.get("GIT_COMMIT", "desconocido")})
        mlflow.log_dict({"factores": modelo.factores, **modelo.info}, "modelo_info.json")
        info = mlflow.pyfunc.log_model(
            name="modelo",
            python_model=ModeloPyfunc(modelo),
            pip_requirements=[],
            registered_model_name=nombre_modelo,
        )
    version = str(info.registered_model_version) if nombre_modelo else None
    return run.info.run_id, version


def version_vigente(cliente: MlflowClient, nombre: str, alias: str) -> str | None:
    try:
        return str(cliente.get_model_version_by_alias(nombre, alias).version)
    except MlflowException:
        return None


def cargar_version(nombre: str, version: str) -> ModeloVentas:
    pyfunc = mlflow.pyfunc.load_model(f"models:/{nombre}/{version}")
    return pyfunc.unwrap_python_model().modelo


def promover(cliente: MlflowClient, nombre: str, alias: str, version: str, motivo: str) -> None:
    anterior = version_vigente(cliente, nombre, alias)
    cliente.set_model_version_tag(nombre, version, "decision", "promovido")
    cliente.set_model_version_tag(nombre, version, "motivo", motivo[:4000])
    if anterior and anterior != version:
        cliente.set_registered_model_alias(nombre, "anterior", anterior)
    cliente.set_registered_model_alias(nombre, alias, version)


def descartar(cliente: MlflowClient, nombre: str, version: str, motivo: str) -> None:
    cliente.set_model_version_tag(nombre, version, "decision", "descartado")
    cliente.set_model_version_tag(nombre, version, "motivo", motivo[:4000])
