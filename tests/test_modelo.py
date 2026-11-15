"""Features sin fuga, cantidad sugerida coherente con los costos y criterio de reemplazo."""

from datetime import date, timedelta

import numpy as np
import pandas as pd
import pytest

from novamercado.entrenamiento import Tabla, ciclo, decidir, entrenar
from novamercado.features import construir
from novamercado.modelo import Baseline, costo, elegir_factores

CFG = {"factor_min": 0.5, "factor_max": 2.0, "factor_paso": 0.01}
PARAMS = {
    "entrenamiento": {
        "dias_calibracion": 14, "semilla": 1,
        "lightgbm": {"objective": "regression_l1", "learning_rate": 0.1, "num_leaves": 15,
                     "min_data_in_leaf": 20, "num_boost_round": 60, "deterministic": True,
                     "force_row_wise": True},
        **CFG,
    },
    "promocion": {"dias_ventana": 14, "dias_minimos": 7, "tolerancia": 0.02},
}


def test_costo_del_enunciado():
    assert costo(10, 7, 1.5, 2.0) == pytest.approx(4.5)
    assert costo(10, 12, 1.5, 2.0) == pytest.approx(4.0)


def test_features_solo_usan_dias_anteriores(cat, historia):
    pedido = pd.DataFrame([{"fecha": date(2025, 3, 1), "tienda_id": 1, "familia_id": "abarrotes",
                            "en_promocion": 2}])
    antes = construir(historia, pedido, cat)
    alterada = historia.copy()
    alterada.loc[alterada["fecha"] >= date(2025, 3, 1), "ventas"] = 1e6
    pd.testing.assert_frame_equal(antes, construir(alterada, pedido, cat))


def test_feriado_local_solo_aplica_a_su_ciudad(cat, historia):
    pedido = pd.DataFrame([{"fecha": date(2025, 7, 25), "tienda_id": t, "familia_id": "abarrotes",
                            "en_promocion": 0} for t in (1, 2)])
    x = construir(historia, pedido, cat)
    assert x["feriado"].tolist() == [0.0, 1.0]  # la tienda 2 está en Antigua Guatemala


def test_factor_sigue_a_los_costos(cat):
    rng = np.random.default_rng(0)
    y = rng.poisson(100, 500).astype(float)
    df = pd.DataFrame({"familia_id": "abarrotes", "ventas": y, "venta_esperada": 100.0})
    cat.costos.loc["abarrotes"] = [3.0, 0.1]  # faltar es mucho más caro → pedir de más
    alto = elegir_factores(df, cat, CFG)["abarrotes"]
    cat.costos.loc["abarrotes"] = [0.1, 3.0]  # sobrar es mucho más caro → pedir de menos
    bajo = elegir_factores(df, cat, CFG)["abarrotes"]
    assert alto > 1.05 and bajo < 0.95


@pytest.mark.parametrize("challenger, baseline, vigente, esperado", [
    (90, 100, None, True),     # primer modelo que mejora al baseline
    (110, 100, None, False),   # no mejora al baseline
    (95, 100, 94, True),       # dentro de la tolerancia del 2 %
    (99, 100, 90, False),      # peor que el vigente por más del 2 %
    (80, 100, 90, True),       # mejor que el vigente
])
def test_criterio_de_reemplazo(challenger, baseline, vigente, esperado):
    promovido, motivo = decidir(challenger, baseline, vigente, 0.02)
    assert promovido is esperado and "Costo en validación" in motivo or "Primer" in motivo


def test_ciclo_completo_mejora_al_baseline(cat, historia):
    resultado = ciclo(historia, cat, PARAMS, vigente=Baseline())
    m = resultado.metricas
    assert set(m) == {"baseline_validacion", "lightgbm_validacion", "vigente_validacion"}
    assert m["lightgbm_validacion"]["costo_total"] < m["baseline_validacion"]["costo_total"]
    assert resultado.promovido
    assert resultado.challenger.info["datos_hasta"] == str(historia["fecha"].max())
    assert resultado.ventana[1] - resultado.ventana[0] == timedelta(days=13)


def test_entrenar_es_determinista(cat, historia):
    tabla = Tabla.desde_historia(historia, cat)
    fin = tabla.ultima_fecha
    a = entrenar("lightgbm", tabla, cat, PARAMS, fin).predecir(tabla.rango(fin, fin))
    b = entrenar("lightgbm", tabla, cat, PARAMS, fin).predecir(tabla.rango(fin, fin))
    pd.testing.assert_frame_equal(a, b)


def test_la_ventana_excluye_lo_que_vio_el_vigente(cat, historia):
    from novamercado.entrenamiento import ventana

    fin = date(2025, 3, 9)
    cfg = PARAMS["promocion"]
    assert ventana(fin, None, cfg) == (date(2025, 2, 24), fin)
    reciente = Baseline(info={"datos_hasta": "2025-03-01"})
    assert ventana(fin, reciente, cfg) == (date(2025, 3, 2), fin)


def test_sin_dias_nuevos_suficientes_el_vigente_se_conserva(cat, historia):
    vigente = Baseline(info={"datos_hasta": "2025-03-05"})
    resultado = ciclo(historia, cat, PARAMS, vigente)
    assert not resultado.promovido and "evidencia" in resultado.motivo
