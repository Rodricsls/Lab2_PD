"""Política ante datos defectuosos: catálogo E01–E10 y casos que no están en el catálogo."""

import copy

import pytest

from novamercado.calidad import (
    Rechazo,
    Registro,
    interpretar_fecha,
    interpretar_numero,
    normalizar_observaciones,
    normalizar_prediccion,
)
from tests.conftest import cuerpo_observacion, cuerpo_prediccion


def acciones(reg: Registro) -> list[tuple[str, str]]:
    return [(o.campo, o.accion) for o in reg.observaciones]


def test_solicitud_valida_queda_ok(cat, familias):
    pedido, reg = normalizar_prediccion(cuerpo_prediccion(cat), cat, familias)
    assert reg.estado == "ok" and reg.observaciones == []
    assert pedido.tienda_id == 1 and len(pedido.familias) == 12


# ------------------------------------------------------------------ catálogo E01–E07


@pytest.mark.parametrize("valor", ["T01", "1", 1.0, " tienda 1 ", "T-01"])
def test_E01_tienda_como_texto_se_repara(cat, familias, valor):
    cuerpo = cuerpo_prediccion(cat) | {"tienda_id": valor}
    pedido, reg = normalizar_prediccion(cuerpo, cat, familias)
    assert pedido.tienda_id == 1
    assert ("tienda_id", "reparar") in acciones(reg) and reg.estado == "corregido"


@pytest.mark.parametrize("escrito, esperado", [
    ("LÁCTEOS", "lacteos"), ("lacteos ", "lacteos"), ("Frutas y Verduras", "frutas_verduras"),
    ("Panadería", "panaderia"), ("snacks y dulces", "snacks"), ("cuidado-personal", "cuidado_personal"),
    ("Mascota", "mascotas"),
])
def test_E02_familia_escrita_de_otra_forma_se_repara(cat, familias, escrito, esperado):
    cuerpo = cuerpo_prediccion(cat)
    i = cat.ids_familia.index(esperado)
    cuerpo["familias"][i]["familia_id"] = escrito
    pedido, reg = normalizar_prediccion(cuerpo, cat, familias)
    assert esperado in pedido.familias and reg.estado == "corregido"


def test_E02_familia_parecida_pero_no_segura_se_advierte(cat, familias):
    cuerpo = cuerpo_prediccion(cat)
    cuerpo["familias"][0]["familia_id"] = "abarrotez varios"
    pedido, reg = normalizar_prediccion(cuerpo, cat, familias)
    assert reg.estado in {"corregido", "advertencia"}
    assert "abarrotes" in pedido.familias


def test_familia_desconocida_se_ignora_con_advertencia(cat, familias):
    cuerpo = cuerpo_prediccion(cat)
    cuerpo["familias"].append({"familia_id": "electrodomesticos", "en_promocion": 1})
    pedido, reg = normalizar_prediccion(cuerpo, cat, familias)
    assert "electrodomesticos" not in pedido.familias
    assert reg.estado == "advertencia"


@pytest.mark.parametrize("valor, esperado, accion", [
    ("3", 3, "reparar"), (2.0, 2, "reparar"), ("1,0", 1, "reparar"),
    (2.6, 3, "advertir"), (-1, 0, "advertir"), ("muchos", 0, "advertir"), (None, 0, "advertir"),
])
def test_E03_promocion_mal_escrita(cat, familias, valor, esperado, accion):
    cuerpo = cuerpo_prediccion(cat)
    cuerpo["familias"][1]["en_promocion"] = valor
    pedido, reg = normalizar_prediccion(cuerpo, cat, familias)
    assert pedido.familias["bebidas"] == esperado
    assert ("familias[1].en_promocion", accion) in acciones(reg)


@pytest.mark.parametrize("texto, esperado, accion", [
    ("2025-03-10T00:00:00", "2025-03-10", "reparar"),
    ("10/03/2025", "2025-03-10", "advertir"),  # ambigua: se asume día/mes
    ("17/03/2025", "2025-03-17", "reparar"),  # día > 12: inequívoca
    ("03/17/2025", "2025-03-17", "reparar"),  # mes/día inequívoca
    ("2025/03/10", "2025-03-10", "reparar"),
    ("20250310", "2025-03-10", "reparar"),
    ("10-03-2025", "2025-03-10", "advertir"),
])
def test_E04_fecha_en_otro_formato(texto, esperado, accion):
    reg = Registro()
    assert interpretar_fecha(texto, reg).isoformat() == esperado
    assert reg.observaciones[0].accion == accion


@pytest.mark.parametrize("texto", ["mañana", "31/02/2025", "", None, 20250310, "2025-13-01"])
def test_fecha_invalida_se_rechaza(texto):
    with pytest.raises(Rechazo):
        interpretar_fecha(texto, Registro())


def test_E05_familias_faltantes_se_pronostican_con_advertencia(cat, familias):
    cuerpo = cuerpo_prediccion(cat)
    cuerpo["familias"] = cuerpo["familias"][:10]
    pedido, reg = normalizar_prediccion(cuerpo, cat, familias)
    assert len(pedido.familias) == 12
    assert pedido.familias["bebe"] == 0 and pedido.familias["mascotas"] == 0
    assert ("familias", "advertir") in acciones(reg)


def test_E06_familia_repetida(cat, familias):
    cuerpo = cuerpo_prediccion(cat)
    cuerpo["familias"].append({"familia_id": "abarrotes", "en_promocion": 3})
    pedido, reg = normalizar_prediccion(cuerpo, cat, familias)
    assert pedido.familias["abarrotes"] == 3 and reg.estado == "advertencia"

    cuerpo = cuerpo_prediccion(cat)
    cuerpo["familias"].append({"familia_id": "abarrotes", "en_promocion": 0})
    _, reg = normalizar_prediccion(cuerpo, cat, familias)
    assert reg.estado == "corregido"  # repetida idéntica: sin supuesto


@pytest.mark.parametrize("campo", ["tienda", "Tienda_ID", "store_id", "id_tienda", "tiendaId"])
def test_E07_campo_renombrado_se_repara(cat, familias, campo):
    cuerpo = cuerpo_prediccion(cat)
    cuerpo[campo] = cuerpo.pop("tienda_id")
    pedido, reg = normalizar_prediccion(cuerpo, cat, familias)
    assert pedido.tienda_id == 1 and ("tienda_id", "reparar") in acciones(reg)


def test_campos_renombrados_dentro_de_cada_familia(cat, familias):
    cuerpo = cuerpo_prediccion(cat)
    cuerpo["familias"] = [{"familia": f["familia_id"], "promocion": 1} for f in cuerpo["familias"]]
    pedido, reg = normalizar_prediccion(cuerpo, cat, familias)
    assert set(pedido.familias.values()) == {1} and reg.estado == "corregido"


def test_campo_desconocido_se_ignora_con_advertencia(cat, familias):
    cuerpo = cuerpo_prediccion(cat) | {"canal": "web"}
    _, reg = normalizar_prediccion(cuerpo, cat, familias)
    assert ("canal", "advertir") in acciones(reg)


@pytest.mark.parametrize("cambio, motivo", [
    ({"tienda_id": 99}, "no existe"),
    ({"tienda_id": None}, "tienda"),
    ({"tienda_id": "Tienda X"}, "tienda"),
    ({"fecha": "ayer"}, "fecha"),
    ({"familias": []}, "familias"),
    ({"familias": [{"familia_id": "zzz", "en_promocion": 0}]}, "Ninguna familia"),
])
def test_rechazos(cat, familias, cambio, motivo):
    with pytest.raises(Rechazo, match=motivo):
        normalizar_prediccion(cuerpo_prediccion(cat) | cambio, cat, familias)


def test_tienda_antes_de_abrir_se_rechaza(cat, familias):
    with pytest.raises(Rechazo, match="abrió"):
        normalizar_prediccion(cuerpo_prediccion(cat, tienda=3) | {"fecha": "2025-03-01"}, cat, familias)


def test_campos_equivalentes_con_valores_distintos_se_rechazan(cat, familias):
    with pytest.raises(Rechazo, match="significan lo mismo"):
        normalizar_prediccion(cuerpo_prediccion(cat) | {"tienda": 2}, cat, familias)


@pytest.mark.parametrize("cuerpo", [None, [], "texto", 5])
def test_cuerpo_que_no_es_objeto_se_rechaza(cat, familias, cuerpo):
    with pytest.raises(Rechazo):
        normalizar_prediccion(cuerpo, cat, familias)


# ------------------------------------------------------------------ observaciones E08–E10


def test_E08_venta_negativa_no_se_guarda(cat, familias):
    cuerpo = cuerpo_observacion(cat)
    cuerpo["ventas"][5]["ventas"] = -4
    obs, reg = normalizar_observaciones(cuerpo, cat, familias)
    assert "frutas_verduras" not in obs.ventas and len(obs.ventas) == 11
    assert reg.estado == "advertencia"


@pytest.mark.parametrize("texto, esperado", [
    ("1,234", 1234), ("812,0", 812), ("1.234", 1234), ("1.234,0", 1234), ("1,234.0", 1234),
    (" 50 ", 50), ("50.0", 50),
])
def test_E10_numero_con_formato(cat, familias, texto, esperado):
    cuerpo = cuerpo_observacion(cat)
    cuerpo["ventas"][0]["ventas"] = texto
    obs, reg = normalizar_observaciones(cuerpo, cat, familias)
    assert obs.ventas["abarrotes"] == esperado and reg.estado == "corregido"


@pytest.mark.parametrize("texto, esperado", [
    ("12,5", 12.5), ("1.234.567", 1234567), ("-3", -3), ("abc", None), ("", None), (True, None),
])
def test_interpretar_numero(texto, esperado):
    assert interpretar_numero(texto) == esperado


def test_venta_no_numerica_no_se_guarda(cat, familias):
    cuerpo = cuerpo_observacion(cat)
    cuerpo["ventas"][0]["ventas"] = "sin dato"
    obs, reg = normalizar_observaciones(cuerpo, cat, familias)
    assert "abarrotes" not in obs.ventas and reg.estado == "advertencia"


def test_las_reglas_no_modifican_el_cuerpo_original(cat, familias):
    cuerpo = cuerpo_prediccion(cat) | {"tienda_id": "T01"}
    copia = copy.deepcopy(cuerpo)
    normalizar_prediccion(cuerpo, cat, familias)
    assert cuerpo == copia
