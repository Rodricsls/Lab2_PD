"""Defensa del contrato: normaliza solicitudes y decide reparar, advertir o rechazar.

Política (detalle en DECISIONES.md, D2):

- **Reparar** cuando la corrección es inequívoca: un tipo distinto con el mismo valor
  (`"5"`, `5.0`), un formato conocido (`"T05"`, `"2025-11-17T00:00:00"`, `"1,234"`), una
  familia escrita de otra forma que coincide con el catálogo, un campo con nombre
  equivalente, un duplicado idéntico.
- **Advertir** cuando se responde con un supuesto: una fecha ambigua (`05/11/2025`), una
  familia parecida pero no segura, una promoción faltante o imposible (se asume 0), familias
  faltantes (se pronostican con promoción 0), duplicados con valores distintos (gana el
  último), un registro de ventas imposible (se descarta), un campo desconocido (se ignora).
- **Rechazar** (422) cuando no se puede responder sin inventar: falta o es inválida la
  tienda, la fecha o la lista; tienda inexistente o aún sin abrir; dos campos equivalentes
  con valores distintos; ninguna familia reconocible.

Las reglas son generales (normalización de texto, alias de campos, similitud con el
catálogo, coerción numérica) para cubrir problemas que no están en el catálogo.
"""

from __future__ import annotations

import difflib
import math
import re
import unicodedata
from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Any

from novamercado.catalogos import Catalogos

REPARAR, ADVERTIR = "reparar", "advertir"


class Rechazo(ValueError):
    """La solicitud no se puede atender sin inventar datos: se responde 422."""


@dataclass
class Observacion:
    campo: str
    accion: str
    detalle: str


@dataclass
class Registro:
    observaciones: list[Observacion] = field(default_factory=list)

    def reparar(self, campo: str, detalle: str) -> None:
        self.observaciones.append(Observacion(campo, REPARAR, detalle))

    def advertir(self, campo: str, detalle: str) -> None:
        self.observaciones.append(Observacion(campo, ADVERTIR, detalle))

    @property
    def estado(self) -> str:
        if any(o.accion == ADVERTIR for o in self.observaciones):
            return "advertencia"
        return "corregido" if self.observaciones else "ok"

    def como_dict(self) -> dict[str, Any]:
        return {
            "estado": self.estado,
            "observaciones": [o.__dict__ for o in self.observaciones],
        }


# --------------------------------------------------------------------------- texto


def normalizar_texto(valor: str) -> str:
    """Minúsculas, sin acentos y con `_` como único separador: 'Frutas y Verduras ' →
    'frutas_y_verduras'."""
    sin_acentos = unicodedata.normalize("NFKD", valor).encode("ascii", "ignore").decode()
    return re.sub(r"[^a-z0-9]+", "_", sin_acentos.lower()).strip("_")


def _compacto(valor: str) -> str:
    return normalizar_texto(valor).replace("_", "")


def _repr(valor: Any) -> str:
    return f"'{valor}'" if isinstance(valor, str) else str(valor)


# --------------------------------------------------------------------------- campos

ALIAS_PREDICCION = {
    "tienda_id": ["tienda", "idtienda", "store", "storeid", "sucursal", "codigotienda"],
    "fecha": ["date", "dia", "fechaprediccion", "fechapronostico"],
    "familias": ["familia", "productos", "items", "categorias", "lineas", "detalle"],
}
ALIAS_OBSERVACION = {
    "tienda_id": ALIAS_PREDICCION["tienda_id"],
    "fecha": ["date", "dia", "fechaventa", "fechaobservacion"],
    "ventas": ["venta", "ventasreales", "items", "lineas", "detalle", "observaciones"],
}
ALIAS_ITEM_PREDICCION = {
    "familia_id": ["familia", "idfamilia", "family", "categoria", "familianombre"],
    "en_promocion": ["promocion", "promo", "enpromo", "promociones", "productosenpromocion"],
}
ALIAS_ITEM_OBSERVACION = {
    "familia_id": ALIAS_ITEM_PREDICCION["familia_id"],
    "ventas": ["venta", "unidades", "cantidad", "ventasreales", "total"],
}


def mapear_campos(
    objeto: dict[str, Any], alias: dict[str, list[str]], prefijo: str, reg: Registro
) -> dict[str, Any]:
    """Devuelve el objeto con los nombres del contrato; repara nombres equivalentes."""
    candidatos = {
        _compacto(nombre): oficial
        for oficial, otros in alias.items()
        for nombre in [oficial, *otros]
    }
    salida: dict[str, Any] = {}
    origen: dict[str, str] = {}
    for clave, valor in objeto.items():
        compacta = _compacto(str(clave))
        oficial = candidatos.get(compacta)
        if oficial is None:
            parecido = difflib.get_close_matches(compacta, candidatos, n=1, cutoff=0.85)
            oficial = candidatos[parecido[0]] if parecido else None
        if oficial is None:
            reg.advertir(f"{prefijo}{clave}", f"campo desconocido {_repr(clave)}: se ignoró")
            continue
        if oficial in salida:
            if salida[oficial] != valor:
                raise Rechazo(
                    f"Los campos {_repr(origen[oficial])} y {_repr(clave)} significan lo mismo "
                    f"({oficial}) pero traen valores distintos."
                )
            reg.reparar(f"{prefijo}{clave}", f"campo repetido {_repr(clave)}: se descartó")
            continue
        if clave != oficial:
            reg.reparar(f"{prefijo}{oficial}", f"el campo {_repr(clave)} se interpretó como '{oficial}'")
        salida[oficial] = valor
        origen[oficial] = str(clave)
    return salida


# --------------------------------------------------------------------------- valores


def interpretar_tienda(valor: Any, cat: Catalogos, reg: Registro) -> int:
    if valor is None or isinstance(valor, bool):
        raise Rechazo("Falta el identificador de la tienda (tienda_id).")
    tienda: int | None = None
    if isinstance(valor, int):
        tienda = valor
    elif isinstance(valor, float) and valor.is_integer():
        tienda = int(valor)
        reg.reparar("tienda_id", f"{valor} se interpretó como {tienda}")
    elif isinstance(valor, str):
        texto = valor.strip()
        numeros = re.findall(r"\d+", texto)
        if len(numeros) == 1:
            tienda = int(numeros[0])
            reg.reparar("tienda_id", f"{_repr(valor)} se interpretó como {tienda}")
        else:
            nombres = {
                int(t): _compacto(f"{fila['nombre']}") for t, fila in cat.tiendas.iterrows()
            }
            coincide = [t for t, n in nombres.items() if _compacto(texto) and _compacto(texto) in n]
            if len(coincide) == 1:
                tienda = coincide[0]
                reg.reparar("tienda_id", f"{_repr(valor)} se interpretó como la tienda {tienda}")
    if tienda is None:
        raise Rechazo(f"No se pudo interpretar la tienda {_repr(valor)}.")
    if tienda not in cat.ids_tienda:
        raise Rechazo(f"La tienda {tienda} no existe en el catálogo.")
    return tienda


_FORMATOS_DMY = re.compile(r"^(\d{1,2})[/.\-](\d{1,2})[/.\-](\d{4})$")
_FORMATOS_YMD = re.compile(r"^(\d{4})[/.\-](\d{1,2})[/.\-](\d{1,2})$")


def interpretar_fecha(valor: Any, reg: Registro) -> date:
    if not isinstance(valor, str) or not valor.strip():
        raise Rechazo(f"La fecha {_repr(valor)} no es válida: se espera AAAA-MM-DD.")
    texto = valor.strip()
    try:
        if re.fullmatch(r"\d{4}-\d{2}-\d{2}", texto):
            return date.fromisoformat(texto)
        if re.fullmatch(r"\d{4}-\d{2}-\d{2}[T ].*", texto):
            fecha = datetime.fromisoformat(texto.replace("Z", "+00:00")).date()
            reg.reparar("fecha", f"{_repr(valor)} se interpretó como {fecha.isoformat()}")
            return fecha
        if m := _FORMATOS_YMD.fullmatch(texto):
            fecha = date(int(m[1]), int(m[2]), int(m[3]))
            reg.reparar("fecha", f"{_repr(valor)} se interpretó como {fecha.isoformat()}")
            return fecha
        if re.fullmatch(r"\d{8}", texto):
            fecha = date(int(texto[:4]), int(texto[4:6]), int(texto[6:]))
            reg.reparar("fecha", f"{_repr(valor)} se interpretó como {fecha.isoformat()}")
            return fecha
        if m := _FORMATOS_DMY.fullmatch(texto):
            a, b, anio = int(m[1]), int(m[2]), int(m[3])
            if a > 12 or a == b:  # día/mes sin ambigüedad
                fecha = date(anio, b, a)
                reg.reparar("fecha", f"{_repr(valor)} se interpretó como {fecha.isoformat()} (día/mes)")
            elif b > 12:  # solo puede ser mes/día
                fecha = date(anio, a, b)
                reg.reparar("fecha", f"{_repr(valor)} se interpretó como {fecha.isoformat()} (mes/día)")
            else:  # ambigua: se usa la convención guatemalteca día/mes
                fecha = date(anio, b, a)
                reg.advertir(
                    "fecha",
                    f"{_repr(valor)} es ambigua; se asumió día/mes ({fecha.isoformat()})",
                )
            return fecha
    except ValueError as e:
        raise Rechazo(f"La fecha {_repr(valor)} no existe: {e}.") from e
    raise Rechazo(f"La fecha {_repr(valor)} no tiene un formato reconocible (use AAAA-MM-DD).")


def interpretar_numero(valor: Any) -> float | None:
    """Convierte números y textos numéricos ('1,234', '812,0', '1.234,5'); None si no se puede."""
    if valor is None or isinstance(valor, bool):
        return None
    if isinstance(valor, int | float):
        return float(valor) if math.isfinite(valor) else None
    if not isinstance(valor, str):
        return None
    texto = re.sub(r"\s", "", valor)
    if not re.fullmatch(r"[+-]?[\d.,]+", texto) or not re.search(r"\d", texto):
        return None
    signo = -1 if texto.startswith("-") else 1
    texto = texto.lstrip("+-")
    if "," in texto and "." in texto:
        decimal = "," if texto.rfind(",") > texto.rfind(".") else "."
        miles = "." if decimal == "," else ","
        texto = texto.replace(miles, "").replace(decimal, ".")
    elif "," in texto or "." in texto:
        sep = "," if "," in texto else "."
        partes = texto.split(sep)
        miles = len(partes) > 2 or (len(partes[-1]) == 3 and partes[0] not in ("", "0"))
        texto = texto.replace(sep, "") if miles else texto.replace(sep, ".")
    try:
        return signo * float(texto)
    except ValueError:
        return None


def interpretar_cantidad(
    valor: Any, campo: str, reg: Registro
) -> tuple[int | None, str | None]:
    """Entero no negativo. Devuelve (valor, problema): problema es None si se pudo usar el
    valor (reparado o no), o la razón por la que no se pudo."""
    if isinstance(valor, int) and not isinstance(valor, bool):
        numero = float(valor)
    else:
        numero = interpretar_numero(valor)
        if numero is None:
            return None, f"{_repr(valor)} no es un número"
    if numero < 0:
        return None, f"{_repr(valor)} es negativo"
    entero = round(numero)
    if not isinstance(valor, int) or isinstance(valor, bool):
        if numero == entero:
            reg.reparar(campo, f"{_repr(valor)} se interpretó como {entero}")
        else:
            reg.advertir(campo, f"{_repr(valor)} no es entero; se redondeó a {entero}")
    return entero, None


class ResolutorFamilias:
    """Reconoce una familia aunque llegue escrita de otra forma."""

    SINONIMOS = {
        "frutas": "frutas_verduras",
        "verduras": "frutas_verduras",
        "frutasyverduras": "frutas_verduras",
        "carne": "carnes",
        "pan": "panaderia",
        "licor": "licores",
        "bebidasalcoholicas": "licores",
        "dulces": "snacks",
        "snacksydulces": "snacks",
        "golosinas": "snacks",
        "higiene": "cuidado_personal",
        "higienepersonal": "cuidado_personal",
        "bebes": "bebe",
        "mascota": "mascotas",
        "lacteo": "lacteos",
        "abarrote": "abarrotes",
        "bebida": "bebidas",
    }

    def __init__(self, cat: Catalogos, reparar: float, advertir: float):
        self.umbral_reparar, self.umbral_advertir = reparar, advertir
        self.claves: dict[str, str] = {}
        for familia_id, fila in cat.familias.iterrows():
            self.claves[_compacto(familia_id)] = familia_id
            self.claves[_compacto(str(fila["nombre"]))] = familia_id
        for sinonimo, familia_id in self.SINONIMOS.items():
            self.claves.setdefault(sinonimo, familia_id)

    def resolver(self, valor: Any) -> tuple[str | None, str | None]:
        """Devuelve (familia_id, acción): acción None si llegó exacta, 'reparar' o
        'advertir'; familia_id None si no se reconoce."""
        if not isinstance(valor, str) or not valor.strip():
            return None, None
        if valor in self.claves.values():
            return valor, None
        compacto = _compacto(valor)
        if compacto in self.claves:
            return self.claves[compacto], REPARAR
        sin_plural = re.sub(r"(es|s)$", "", compacto)
        if sin_plural in self.claves:
            return self.claves[sin_plural], REPARAR
        mejor, similitud = None, 0.0
        for clave, familia_id in self.claves.items():
            r = difflib.SequenceMatcher(None, compacto, clave).ratio()
            if r > similitud:
                mejor, similitud = familia_id, r
        if similitud >= self.umbral_reparar:
            return mejor, REPARAR
        if similitud >= self.umbral_advertir:
            return mejor, ADVERTIR
        return None, None


# --------------------------------------------------------------------------- solicitudes


@dataclass
class Prediccion:
    tienda_id: int
    fecha: date
    familias: dict[str, int]  # familia_id → en_promocion, en el orden de la solicitud


@dataclass
class Observaciones:
    tienda_id: int
    fecha: date
    ventas: dict[str, int]


def _encabezado(cuerpo: Any, alias: dict, lista: str, cat: Catalogos, reg: Registro):
    if not isinstance(cuerpo, dict):
        raise Rechazo("El cuerpo debe ser un objeto JSON.")
    campos = mapear_campos(cuerpo, alias, "", reg)
    if "tienda_id" not in campos:
        raise Rechazo("Falta el campo tienda_id.")
    if "fecha" not in campos:
        raise Rechazo("Falta el campo fecha.")
    if not isinstance(campos.get(lista), list) or not campos[lista]:
        raise Rechazo(f"Falta la lista '{lista}' o está vacía.")
    tienda = interpretar_tienda(campos["tienda_id"], cat, reg)
    fecha = interpretar_fecha(campos["fecha"], reg)
    if fecha < cat.apertura(tienda):
        raise Rechazo(
            f"La tienda {tienda} abrió el {cat.apertura(tienda).isoformat()}; "
            f"no hay ventas que pronosticar ni registrar el {fecha.isoformat()}."
        )
    return tienda, fecha, campos[lista]


def _items(lista, alias_item, lista_nombre, valor_nombre, familias: ResolutorFamilias, reg):
    """Recorre la lista: resuelve familia y devuelve [(familia, valor_crudo, campo_valor)]."""
    salida = []
    for i, item in enumerate(lista):
        prefijo = f"{lista_nombre}[{i}]"
        if not isinstance(item, dict):
            reg.advertir(prefijo, f"elemento {_repr(item)} no es un objeto: se ignoró")
            continue
        campos = mapear_campos(item, alias_item, f"{prefijo}.", reg)
        crudo = campos.get("familia_id")
        familia, accion = familias.resolver(crudo)
        if familia is None:
            reg.advertir(f"{prefijo}.familia_id", f"familia {_repr(crudo)} no reconocida: se ignoró")
            continue
        if accion == REPARAR:
            reg.reparar(f"{prefijo}.familia_id", f"{_repr(crudo)} se interpretó como '{familia}'")
        elif accion == ADVERTIR:
            reg.advertir(
                f"{prefijo}.familia_id", f"{_repr(crudo)} se asumió como '{familia}' (parecido)"
            )
        salida.append((familia, campos.get(valor_nombre), f"{prefijo}.{valor_nombre}"))
    return salida


def _sin_duplicados(valores: list[tuple[str, int, str]], lista_nombre: str, reg: Registro):
    final: dict[str, int] = {}
    for familia, valor, campo in valores:
        if familia in final:
            if final[familia] == valor:
                reg.reparar(campo, f"'{familia}' venía repetida con el mismo valor: se usó una vez")
            else:
                reg.advertir(
                    campo,
                    f"'{familia}' venía repetida ({final[familia]} y {valor}): "
                    "se usó el último valor",
                )
        final[familia] = valor
    return final


def normalizar_prediccion(
    cuerpo: Any, cat: Catalogos, familias: ResolutorFamilias
) -> tuple[Prediccion, Registro]:
    reg = Registro()
    tienda, fecha, lista = _encabezado(cuerpo, ALIAS_PREDICCION, "familias", cat, reg)
    if fecha > cat.ultimo_feriado:
        reg.advertir("fecha", f"{fecha.isoformat()} está fuera del calendario de feriados conocido")

    valores = []
    for familia, crudo, campo in _items(
        lista, ALIAS_ITEM_PREDICCION, "familias", "en_promocion", familias, reg
    ):
        if crudo is None:
            reg.advertir(campo, "falta en_promocion: se asumió 0")
            promo = 0
        else:
            promo, problema = interpretar_cantidad(crudo, campo, reg)
            if problema:
                reg.advertir(campo, f"{problema}: se asumió 0")
                promo = 0
        valores.append((familia, promo, campo))
    if not valores:
        raise Rechazo("Ninguna familia de la solicitud se pudo reconocer.")
    recibidas = _sin_duplicados(valores, "familias", reg)

    faltantes = [f for f in cat.ids_familia if f not in recibidas]
    if faltantes:
        reg.advertir(
            "familias",
            f"faltaban {len(faltantes)} familias ({', '.join(faltantes)}): "
            "se pronosticaron asumiendo 0 productos en promoción",
        )
    # Orden de la solicitud; las familias que faltaban van al final.
    return Prediccion(tienda, fecha, {**recibidas, **dict.fromkeys(faltantes, 0)}), reg


def normalizar_observaciones(
    cuerpo: Any, cat: Catalogos, familias: ResolutorFamilias
) -> tuple[Observaciones, Registro]:
    reg = Registro()
    tienda, fecha, lista = _encabezado(cuerpo, ALIAS_OBSERVACION, "ventas", cat, reg)
    valores = []
    for familia, crudo, campo in _items(
        lista, ALIAS_ITEM_OBSERVACION, "ventas", "ventas", familias, reg
    ):
        ventas, problema = interpretar_cantidad(crudo, campo, reg)
        if problema:
            reg.advertir(campo, f"{problema}: el registro de '{familia}' no se guardó")
            continue
        valores.append((familia, ventas, campo))
    return Observaciones(tienda, fecha, _sin_duplicados(valores, "ventas", reg)), reg
