"""Contrato de la API (Anexo) sin servidor de MLflow, base de datos ni Airflow."""

from __future__ import annotations

from datetime import date

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from novamercado.api.main import Contexto, crear_app
from novamercado.api.servicio import GestorModelo, MonitorSalud
from novamercado.modelo import Baseline
from tests.conftest import cuerpo_observacion, cuerpo_prediccion


class RepositorioMemoria:
    def __init__(self, historia: pd.DataFrame):
        self._historia = historia
        self.observaciones: dict[tuple, int] = {}
        self.predicciones: list[dict] = []
        self.rechazos: list[str] = []
        self.reentrenamientos: dict[str, dict] = {}

    def historia(self, tienda_id: int, fecha: date) -> pd.DataFrame:
        h = self._historia
        return h[(h["tienda_id"] == tienda_id) & (h["fecha"] < fecha)]

    def registrar_prediccion(self, cuerpo, respuesta, filas):
        self.predicciones.append(respuesta)

    def registrar_rechazo(self, tipo, cuerpo, detalle):
        self.rechazos.append(detalle)

    def ventas_previas(self, tienda_id, fecha):
        return {f: v for (t, d, f), v in self.observaciones.items() if t == tienda_id and d == fecha}

    def guardar_observaciones(self, cuerpo, tienda_id, fecha, ventas, calidad):
        cambios = 0
        for f, v in ventas.items():
            if self.observaciones.get((tienda_id, fecha, f)) != v:
                self.observaciones[(tienda_id, fecha, f)] = v
                cambios += 1
        return cambios

    def crear_reentrenamiento(self, id_):
        self.reentrenamientos[id_] = {"estado": "en_curso", "terminado_en": None}

    def leer_reentrenamiento(self, id_):
        return self.reentrenamientos.get(id_)


class AirflowFalso:
    def __init__(self, disponible=True):
        self.disponible, self.estados = disponible, {}

    def disparar(self):
        if not self.disponible:
            raise RuntimeError("Airflow no responde")
        run_id = f"manual__{len(self.estados)}"
        self.estados[run_id] = "en_curso"
        return run_id

    def estado(self, run_id):
        return self.estados.get(run_id)


def gestor_con_modelo(version="1"):
    modelo = Baseline(factores={"abarrotes": 1.5, "panaderia": 0.8})
    gestor = GestorModelo("novamercado-ventas", "champion", cache=None,
                          consultar_version=lambda: version, descargar=lambda v: modelo)
    gestor.modelo, gestor.version = modelo, version
    return gestor


@pytest.fixture
def entorno(cat, familias, historia):
    repo, airflow = RepositorioMemoria(historia), AirflowFalso()
    monitor = MonitorSalud({"base_datos": lambda: True, "mlflow": lambda: True, "airflow": lambda: True})
    monitor.revisar()
    ctx = Contexto(cat, familias, repo, gestor_con_modelo(), monitor, airflow)
    with TestClient(crear_app(lambda: ctx)) as cliente:
        yield cliente, ctx


# ------------------------------------------------------------------ POST /v1/prediccion


def test_prediccion_valida_cumple_el_contrato(entorno, cat):
    cliente, ctx = entorno
    r = cliente.post("/v1/prediccion", json=cuerpo_prediccion(cat))
    assert r.status_code == 200
    cuerpo = r.json()
    assert set(cuerpo) >= {"tienda_id", "fecha", "version_modelo", "predicciones", "calidad"}
    assert cuerpo["tienda_id"] == 1 and cuerpo["fecha"] == "2025-03-10"
    assert cuerpo["version_modelo"] == "1" and isinstance(cuerpo["version_modelo"], str)
    assert cuerpo["calidad"] == {"estado": "ok", "observaciones": []}
    assert len(cuerpo["predicciones"]) == 12
    for p in cuerpo["predicciones"]:
        assert set(p) == {"familia_id", "venta_esperada", "cantidad_sugerida"}
        assert isinstance(p["cantidad_sugerida"], int) and p["cantidad_sugerida"] >= 0
        assert p["venta_esperada"] > 0
    assert len(ctx.repositorio.predicciones) == 1


def test_la_cantidad_sugerida_usa_el_factor_de_la_familia(entorno, cat):
    cliente, _ = entorno
    respuesta = cliente.post("/v1/prediccion", json=cuerpo_prediccion(cat)).json()
    pred = {p["familia_id"]: p for p in respuesta["predicciones"]}
    assert pred["abarrotes"]["cantidad_sugerida"] == round(pred["abarrotes"]["venta_esperada"] * 1.5)
    assert pred["panaderia"]["cantidad_sugerida"] == round(pred["panaderia"]["venta_esperada"] * 0.8)


def test_prediccion_reparada_informa_las_correcciones(entorno, cat):
    cliente, _ = entorno
    cuerpo = cuerpo_prediccion(cat) | {"tienda_id": "T01", "fecha": "2025-03-10T00:00:00"}
    cuerpo["familias"][2]["familia_id"] = "LÁCTEOS"
    r = cliente.post("/v1/prediccion", json=cuerpo).json()
    assert r["calidad"]["estado"] == "corregido"
    campos = {o["campo"] for o in r["calidad"]["observaciones"]}
    assert {"tienda_id", "fecha", "familias[2].familia_id"} <= campos
    assert all(o["accion"] == "reparar" for o in r["calidad"]["observaciones"])


def test_prediccion_con_supuestos_responde_advertencia(entorno, cat):
    cliente, _ = entorno
    cuerpo = cuerpo_prediccion(cat)
    cuerpo["familias"] = cuerpo["familias"][:10]
    r = cliente.post("/v1/prediccion", json=cuerpo).json()
    assert r["calidad"]["estado"] == "advertencia" and len(r["predicciones"]) == 12


@pytest.mark.parametrize("cuerpo, codigo", [
    ({"tienda_id": 99, "fecha": "2025-03-10", "familias": [{"familia_id": "abarrotes", "en_promocion": 0}]}, 422),
    ({"fecha": "2025-03-10", "familias": [{"familia_id": "abarrotes", "en_promocion": 0}]}, 422),
    ({"tienda_id": 1, "fecha": "no-es-fecha", "familias": [{"familia_id": "abarrotes"}]}, 422),
])
def test_rechazo_responde_422_con_detail(entorno, cuerpo, codigo):
    cliente, ctx = entorno
    r = cliente.post("/v1/prediccion", json=cuerpo)
    assert r.status_code == codigo and isinstance(r.json()["detail"], str) and r.json()["detail"]
    assert ctx.repositorio.rechazos


def test_cuerpo_que_no_es_json_responde_422(entorno):
    cliente, _ = entorno
    r = cliente.post("/v1/prediccion", content=b"{no es json", headers={"content-type": "application/json"})
    assert r.status_code == 422 and "JSON" in r.json()["detail"]


def test_misma_fecha_se_puede_pedir_dos_veces(entorno, cat):
    cliente, ctx = entorno
    for _ in range(2):
        assert cliente.post("/v1/prediccion", json=cuerpo_prediccion(cat)).status_code == 200
    assert len(ctx.repositorio.predicciones) == 2


def test_sin_modelo_cargado_prediccion_responde_503(entorno, cat):
    cliente, ctx = entorno
    ctx.gestor.modelo, ctx.gestor.version = None, None
    assert cliente.post("/v1/prediccion", json=cuerpo_prediccion(cat)).status_code == 503


# ------------------------------------------------------------------ POST /v1/observaciones


def test_observaciones_son_idempotentes(entorno, cat):
    cliente, _ = entorno
    cuerpo = cuerpo_observacion(cat)
    primera = cliente.post("/v1/observaciones", json=cuerpo).json()
    assert primera == {"registros_guardados": 12, "calidad": {"estado": "ok", "observaciones": []}}
    segunda = cliente.post("/v1/observaciones", json=cuerpo).json()
    assert segunda["registros_guardados"] == 0 and segunda["calidad"]["estado"] == "ok"


def test_observacion_modificada_cuenta_y_advierte(entorno, cat):
    cliente, _ = entorno
    cliente.post("/v1/observaciones", json=cuerpo_observacion(cat))
    cambiada = cuerpo_observacion(cat)
    cambiada["ventas"][0]["ventas"] = 999
    r = cliente.post("/v1/observaciones", json=cambiada).json()
    assert r["registros_guardados"] == 1 and r["calidad"]["estado"] == "advertencia"


def test_observacion_con_venta_negativa_guarda_el_resto(entorno, cat):
    cliente, _ = entorno
    cuerpo = cuerpo_observacion(cat)
    cuerpo["ventas"][0]["ventas"] = -4
    cuerpo["ventas"][1]["ventas"] = "1,234"
    r = cliente.post("/v1/observaciones", json=cuerpo).json()
    assert r["registros_guardados"] == 11 and r["calidad"]["estado"] == "advertencia"
    acciones = {o["accion"] for o in r["calidad"]["observaciones"]}
    assert acciones == {"reparar", "advertir"}


# ------------------------------------------------------------------ administración y salud


def test_reentrenar_responde_202_con_id_y_estado_en_curso(entorno):
    cliente, _ = entorno
    r = cliente.post("/admin/reentrenar")
    assert r.status_code == 202 and isinstance(r.json()["id"], str)
    estado = cliente.get(f"/admin/reentrenar/{r.json()['id']}").json()
    assert estado == {"estado": "en_curso"}


def test_reentrenamiento_terminado_trae_promovido_version_y_motivo(entorno):
    cliente, ctx = entorno
    run_id = cliente.post("/admin/reentrenar").json()["id"]
    ctx.airflow.estados[run_id] = "exitoso"
    ctx.repositorio.reentrenamientos[run_id] = {
        "estado": "exitoso", "promovido": True, "version": "2", "motivo": "mejor",
        "terminado_en": "2025-03-10",
    }
    assert cliente.get(f"/admin/reentrenar/{run_id}").json() == {
        "estado": "exitoso", "promovido": True, "version": "2", "motivo": "mejor"}


def test_reentrenamiento_desconocido_404_y_airflow_caido_503(entorno):
    cliente, ctx = entorno
    assert cliente.get("/admin/reentrenar/no-existe").status_code == 404
    ctx.airflow.disponible = False
    assert cliente.post("/admin/reentrenar").status_code == 503


def test_modelo_vigente(entorno):
    cliente, _ = entorno
    assert cliente.get("/modelo").json() == {"nombre": "novamercado-ventas", "alias": "champion", "version": "1"}


def test_health_ok_degradado_y_sin_modelo(entorno):
    cliente, ctx = entorno
    assert cliente.get("/health").json()["estado"] == "ok"
    ctx.monitor.estado["mlflow"] = False
    r = cliente.get("/health")
    assert r.status_code == 200 and r.json()["estado"] == "degradado" and "mlflow" in r.json()["motivo"]
    ctx.gestor.modelo, ctx.gestor.version = None, None
    assert cliente.get("/health").status_code == 503


def test_gestor_cambia_de_modelo_sin_reiniciar_y_sobrevive_a_mlflow_caido(tmp_path):
    versiones = iter(["1", "2"])
    gestor = GestorModelo("m", "champion", tmp_path, consultar_version=lambda: next(versiones),
                          descargar=lambda v: Baseline(info={"v": v}))
    assert gestor.sincronizar() and gestor.version == "1"
    assert gestor.sincronizar() and gestor.version == "2"

    def caido():
        raise ConnectionError("MLflow no responde")

    gestor._consultar = caido
    with pytest.raises(ConnectionError):
        gestor.sincronizar()
    assert gestor.actual()[1] == "2"  # sigue respondiendo con el modelo cargado

    nuevo = GestorModelo("m", "champion", tmp_path, caido, lambda v: None)
    assert nuevo.cargar_cache() and nuevo.version == "2"  # arranque con MLflow caído
