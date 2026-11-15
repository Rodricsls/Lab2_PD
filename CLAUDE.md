# Contexto para agentes · Lab2_PD

Producto: **servicio de pronóstico diario de ventas** para NovaMercado (10 tiendas, 12
familias). Cada mañana una tienda pide el pronóstico del día (venta esperada y cantidad
sugerida por familia) y al día siguiente reporta las ventas reales. Lo difícil es operar:
defender el contrato ante datos defectuosos, registrar todo y reentrenar sin intervención.

## Mapa

| Ruta | Contenido |
|---|---|
| `src/novamercado/calidad.py` | Política reparar / advertir / rechazar (reglas generales) |
| `src/novamercado/features.py` | Features de un día con ventas de días anteriores |
| `src/novamercado/modelo.py` | Baseline, LightGBM, factor por familia, costo |
| `src/novamercado/entrenamiento.py` | Entrenar, evaluar, criterio de reemplazo (`decidir`) |
| `src/novamercado/registro.py` | MLflow: runs, Registry, alias `@champion` |
| `src/novamercado/operacion.py` | Arranque desde cero y pasos del reentrenamiento |
| `src/novamercado/db.py` | PostgreSQL del producto |
| `src/novamercado/api/` | FastAPI: endpoints, esquemas Pydantic, recarga del modelo, salud |
| `dags/reentrenamiento.py` | DAG que dispara `POST /admin/reentrenar` |
| `config/params.yaml` | Parámetros (ventanas, tolerancia, umbrales) |
| `docker-compose.yaml`, `docker/` | Producto completo con un comando |
| `datos/publicados/` | Archivos del GES (no se versionan) |

## Invariantes

- **Contrato exacto** (Anexo del enunciado): no omitir ni renombrar campos;
  `version_modelo` es texto; `cantidad_sugerida` entero ≥ 0; rechazo = 422 con `detail`.
- **La limpieza vive en `calidad.py`**, nunca en los endpoints. Reglas generales
  (normalización, alias, similitud, coerción), no casos sueltos del catálogo.
- **Sin reloj:** prohibido usar la fecha del sistema para validar o predecir.
- **Sin fuga:** el pronóstico de D solo usa ventas con fecha < D.
- **Idempotencia:** observaciones con upsert; reenviar lo mismo guarda 0.
- **El vigente solo cambia al final de un reentrenamiento exitoso** que pase `decidir`.
- **Pruebas sin servicios:** pytest no requiere MLflow, Postgres ni Airflow.
- **Datos y credenciales fuera del repo:** `datos/publicados/` y `.env` están ignorados.

## Forma de trabajar

- Rama `tipo/descripcion` → PR → CI verde → squash merge. Conventional Commits.
- Antes de un PR: `.venv/Scripts/ruff check .` y `.venv/Scripts/python -m pytest -q`.
- Toda decisión no obvia va a `DECISIONES.md`.
