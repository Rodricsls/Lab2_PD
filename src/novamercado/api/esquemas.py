"""Modelos de Pydantic del contrato (Anexo del enunciado)."""

from __future__ import annotations

from datetime import date
from typing import Literal

from pydantic import BaseModel, Field, field_serializer


class FamiliaPromocion(BaseModel):
    familia_id: str
    en_promocion: int = Field(ge=0)


class SolicitudPrediccion(BaseModel):
    """Solicitud ya normalizada por `calidad.normalizar_prediccion`."""

    tienda_id: int
    fecha: date
    familias: list[FamiliaPromocion] = Field(min_length=1)


class ObservacionCalidad(BaseModel):
    campo: str
    accion: Literal["reparar", "advertir"]
    detalle: str


class Calidad(BaseModel):
    estado: Literal["ok", "corregido", "advertencia"]
    observaciones: list[ObservacionCalidad]


class PrediccionFamilia(BaseModel):
    familia_id: str
    venta_esperada: float
    cantidad_sugerida: int = Field(ge=0)


class RespuestaPrediccion(BaseModel):
    tienda_id: int
    fecha: date
    version_modelo: str
    predicciones: list[PrediccionFamilia]
    calidad: Calidad

    @field_serializer("fecha")
    def _fecha(self, fecha: date) -> str:
        return fecha.isoformat()


class RespuestaObservaciones(BaseModel):
    registros_guardados: int = Field(ge=0)
    calidad: Calidad


class RespuestaReentrenar(BaseModel):
    id: str


class EstadoReentrenamiento(BaseModel):
    estado: Literal["en_curso", "exitoso", "fallido"]
    promovido: bool | None = None
    version: str | None = None
    motivo: str | None = None


class RespuestaModelo(BaseModel):
    nombre: str
    alias: str
    version: str


class RespuestaSalud(BaseModel):
    estado: Literal["ok", "degradado", "iniciando"]
    motivo: str | None = None
    version_modelo: str | None = None
    dependencias: dict[str, bool]
