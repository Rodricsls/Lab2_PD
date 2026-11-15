# Decisiones técnicas y de producto

Cada decisión registra el contexto, las alternativas, la elegida y por qué.

## D1 · Arquitectura

- **Servicios** (un `docker compose up`): `api` (FastAPI), `postgres-app` (base del
  producto), `mlflow` (tracking y Model Registry), `inicializador` (una vez por arranque),
  Airflow 3 con LocalExecutor (`airflow-apiserver`, `airflow-scheduler`,
  `airflow-dag-processor`, `airflow-init`) y su propia base `postgres-airflow`.
- **El modelo no va en la imagen:** la API descarga la versión con alias `@champion` del
  Registry y la guarda en una cache en volumen. *Alternativa descartada:* modelo dentro de la
  imagen, que obliga a reconstruir y redesplegar en cada reentrenamiento.
- **Arranque desde cero sin pasar por Airflow:** el `inicializador` crea el esquema, carga
  `historico_ventas.csv` si la base está vacía y, si el Registry no tiene vigente, entrena,
  registra y promueve el primer modelo (≈ 30 s). Así el producto queda listo aunque Airflow
  tarde en arrancar. Al volver a levantar el producto detecta lo que ya existe y no repite
  nada. El DAG y el inicializador usan las mismas funciones (`entrenamiento.ciclo`).
- **MLflow con SQLite y artefactos servidos por HTTP** (`--serve-artifacts`) en un volumen:
  los clientes (API, inicializador, Airflow) solo necesitan `http://mlflow:5000`.
- **Imágenes con versiones fijas:** `python:3.12.12-slim-bookworm`,
  `apache/airflow:3.3.2-python3.12`, `ghcr.io/mlflow/mlflow:v3.18.0`, `postgres:16.10`;
  dependencias de Python con `==`. pandas y numpy son las mismas en la API y en Airflow para
  que un modelo entrenado en el DAG se cargue igual en la API.
- **Datos y credenciales fuera del repositorio y de las imágenes:** los archivos publicados
  se montan en solo lectura desde `datos/publicados/`; las credenciales van en `.env`
  (plantilla `.env.example`).

## D2 · Política ante datos defectuosos

Principio: **reparar** solo si la corrección es inequívoca; **advertir** si se responde con
un supuesto; **rechazar** (422) si responder obligaría a inventar. Las reglas son generales
(`calidad.py`), no casos del catálogo: normalización de texto (minúsculas, sin acentos,
separadores), alias y similitud de nombres de campo, similitud con el catálogo de familias
y coerción numérica y de fechas.

| Problema | Política | Por qué |
|---|---|---|
| E01 tienda como texto o con prefijo (`"5"`, `"T05"`, `5.0`, `"tienda 5"`) | Reparar | Un único número en el texto: inequívoco |
| Tienda inexistente, ausente o sin número reconocible | Rechazar | Pronosticar otra tienda sería inventar |
| Fecha anterior a la apertura de la tienda | Rechazar | No hay ventas que pronosticar ni registrar |
| E02 familia con mayúsculas, acentos, espacios, plural, nombre del catálogo o sinónimo | Reparar | Coincide exactamente tras normalizar |
| Familia parecida (similitud 0.75–0.88) | Advertir | Probable pero no seguro |
| Familia no reconocible | Advertir y se omite | Las demás familias sí se pueden atender |
| Ninguna familia reconocible / lista vacía | Rechazar | No hay nada que pronosticar |
| E03 promoción como texto o decimal entero (`"3"`, `2.0`, `"2,0"`) | Reparar | Mismo valor |
| Promoción con decimales (`2.6`) | Advertir (se redondea) | Supuesto sobre el valor |
| Promoción ausente, negativa o no numérica | Advertir (se asume 0) | Es una feature, no la clave: se puede responder con el supuesto explícito |
| E04 fecha ISO con hora, `AAAA/MM/DD`, `AAAAMMDD`, `DD/MM/AAAA` con día > 12, `MM/DD/AAAA` con día > 12 | Reparar | Una sola lectura posible |
| Fecha `DD/MM` ambigua (ambos ≤ 12) | Advertir (se asume día/mes) | Convención guatemalteca; se informa |
| Fecha inexistente o sin formato reconocible | Rechazar | Sin fecha no hay pronóstico |
| Fecha posterior al calendario de feriados | Advertir | Se pronostica sin conocer feriados |
| E05 faltan familias | Advertir: se pronostican con promoción 0 | La tienda necesita todas sus familias; se informa el supuesto |
| E06 familia repetida igual | Reparar | Duplicado sin ambigüedad |
| E06 familia repetida con valores distintos | Advertir: gana el último | Se asume que el último valor corrige al anterior |
| E07 campo con otro nombre (`tienda`, `store_id`, `Tienda_ID`, `promo`…) | Reparar | Alias conocido o nombre casi idéntico |
| Dos campos equivalentes con valores distintos | Rechazar | No hay forma segura de elegir |
| Campo desconocido | Advertir (se ignora) | No afecta la respuesta, pero se informa |
| Cuerpo que no es JSON u objeto | Rechazar | |
| E08 ventas negativas | Advertir: el registro no se guarda | Una venta no puede ser negativa (son devoluciones); convertirla a 0 sería inventar |
| Ventas no numéricas o ausentes | Advertir: el registro no se guarda | |
| E09 envío repetido idéntico | Sin problema: `registros_guardados = 0` | Upsert idempotente |
| Observación que cambia un valor ya guardado | Advertir: se reemplaza | Se asume que el último envío corrige |
| E10 `"1,234"`, `"812,0"`, `"1.234,5"` | Reparar | Separador de miles = grupo de 3 dígitos; decimal = el último separador |

El estado de `calidad` sigue el contrato: `advertencia` si hay alguna acción `advertir`,
`corregido` si solo hubo reparaciones, `ok` sin observaciones. Cada rechazo y cada corrección
quedan registrados en la base (`solicitudes`, `correcciones`).

## D3 · Información que usa el modelo al predecir

- Para la fecha D de una tienda y familia: **solo ventas con fecha < D** (histórico +
  observaciones recibidas; las observaciones prevalecen si coinciden), la promoción de D que
  trae la solicitud, el calendario (`feriados.csv`, con feriados locales solo para la ciudad
  de la tienda) y los atributos de la tienda. Nunca la fecha del sistema.
- Features: media de los últimos 28 días (`base`) y, relativos a ella, la venta de hace 7 y
  14 días, la media del mismo día de las últimas 4 semanas y las medias de 7 y 14 días;
  dispersión y fracción de ceros; día de la semana y del mes; quincena; tipo de fecha
  especial (feriado, comercial, medio día, pago); tienda, familia y formato.
- **Objetivo relativo:** el modelo predice `log(venta) − log(base)`. Un cambio en el nivel de
  la demanda entra por la base de inmediato, sin esperar un reentrenamiento, y el modelo no
  necesita extrapolar fuera del rango de niveles que vio.
- Para entrenar, las promociones de los días observados se toman de la última solicitud de
  pronóstico de ese día (las observaciones no las traen). Una prueba verifica que alterar
  ventas de D en adelante no cambia ninguna feature de D.

## D4 · Cantidad sugerida

- `cantidad_sugerida = round(venta_esperada × factor_familia)`, con el factor de cada familia
  elegido en una grilla (0.50–2.00) para **minimizar el costo de `costos.csv`** en los últimos
  14 días antes del corte de entrenamiento (con un modelo entrenado sin esos días).
- Es la solución newsvendor sin suponer una distribución: si faltar cuesta más que sobrar el
  factor queda por encima de 1 (licores 1.67, cuidado personal 1.36) y al revés (panadería
  0.85, frutas y verduras 0.89). *Alternativa descartada:* un modelo cuantílico por familia
  (12 cuantiles críticos distintos): más modelos que mantener, y el factor calibrado optimiza
  directamente la métrica que se evalúa.
- Los factores viajan con el modelo en el Registry; un cambio en `costos.csv` se refleja en
  el siguiente reentrenamiento.

## D5 · Modelos y criterio de reemplazo

- **Baseline:** media de las últimas 4 ventas del mismo día de la semana, con el mismo factor
  por familia. **Modelo de ML:** LightGBM global (objetivo L1 sobre el log-relativo,
  determinista). En el arranque, con validación en los últimos 14 días del histórico:
  LightGBM costo **44 746** vs **55 922** del baseline (−20 %), WAPE 0.180 vs 0.223.
- **Ventana de comparación:** los días con ventas posteriores a los datos del vigente (que
  ningún modelo comparado vio), como máximo los últimos 14. El challenger y el baseline se
  entrenan con datos hasta antes de la ventana.
- **Criterio (automático, `entrenamiento.decidir`):** se promueve el challenger si
  1. su costo en la ventana es menor que el del baseline (protege contra un entrenamiento
     roto o datos corruptos), y
  2. su costo en la ventana no supera al del vigente en más de 2 %.
  Si la ventana tiene menos de 7 días, no hay evidencia y el vigente se conserva.
- Lo que se registra y se promueve es el challenger **reentrenado con todos los datos**.
  La tolerancia favorece incorporar los datos más recientes (la demanda puede cambiar)
  mientras la comparación en días no vistos impide promover una regresión.
- *Alternativa descartada:* comparar en los últimos 14 días aunque el vigente los haya
  visto; en las pruebas el vigente siempre ganaba por haber visto parte de la ventana y nunca
  se habría reemplazado.
- Cada ciclo deja en MLflow un run del baseline, un run del challenger con las métricas de
  los tres modelos en la ventana, la huella de los datos y el motivo, y una versión en el
  Registry con `decision = promovido | descartado`. El vigente anterior queda con alias
  `@anterior`.

## D6 · Reentrenamiento y operación

- `POST /admin/reentrenar` dispara el DAG `reentrenamiento` por la API REST de Airflow y
  responde el `dag_run_id` como `id`; `GET /admin/reentrenar/{id}` combina el estado de
  Airflow con el resultado que el DAG escribe en la tabla `reentrenamientos`.
- El alias solo se mueve en la última tarea de decisión; si algo falla antes, el vigente no
  cambia y la ejecución queda `fallido` con el motivo. Duración medida: 35–50 s.
- **Recarga sin reinicio:** la API consulta el alias cada 30 s y cambia de modelo en memoria
  (medido: ≈ 20 s tras terminar la ejecución).
- **MLflow caído:** la API sigue con el modelo en memoria (y en la cache en disco si se
  reinicia sin MLflow). Un monitor en segundo plano revisa base, MLflow y Airflow cada 10 s:
  `/health` pasó a `degradado` en 9 s y volvió a `ok` en 11 s al restablecerse MLflow.
- `/health` responde 503 mientras no haya modelo cargado, `degradado` (200) con el motivo si
  falta alguna dependencia, `ok` si todo funciona.
- Persistencia en volúmenes (`app-db`, `mlflow-datos`, `airflow-db`, `api-cache`): tras
  `docker compose down` y `up` el producto estuvo listo en 63 s con el mismo modelo y datos.
