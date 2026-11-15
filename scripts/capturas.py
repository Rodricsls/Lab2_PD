"""Capturas de MLflow y Airflow para la evidencia (requiere playwright y Edge o Chromium).

    python scripts/capturas.py docs/evidencia
"""

import os
import sys
from pathlib import Path

from playwright.sync_api import sync_playwright

SALIDA = Path(sys.argv[1] if len(sys.argv) > 1 else "docs/evidencia")
SALIDA.mkdir(parents=True, exist_ok=True)
USUARIO = os.environ.get("AIRFLOW_ADMIN_USER", "admin")
CLAVE = os.environ.get("AIRFLOW_ADMIN_PASSWORD", "")

with sync_playwright() as p:
    navegador = p.chromium.launch(channel="msedge", headless=True)
    pagina = navegador.new_page(viewport={"width": 1440, "height": 900}, locale="es-ES")

    for nombre, url in {
        "mlflow_experimento": "http://localhost:5000/#/experiments/1",
        "mlflow_modelo_versiones": "http://localhost:5000/#/models/novamercado-ventas",
    }.items():
        pagina.goto(url)
        pagina.wait_for_timeout(5000)
        pagina.screenshot(path=SALIDA / f"{nombre}.png")
        print("ok", nombre)

    pagina.goto("http://localhost:8080/auth/login/")
    pagina.fill("#username", USUARIO)
    pagina.fill("#password", CLAVE)
    pagina.keyboard.press("Enter")
    pagina.wait_for_load_state("networkidle")
    for nombre, ruta in {
        "airflow_reentrenamiento": "/dags/reentrenamiento",
        "airflow_reentrenamiento_ejecuciones": "/dags/reentrenamiento/runs",
    }.items():
        pagina.goto("http://localhost:8080" + ruta)
        pagina.wait_for_load_state("networkidle")
        pagina.wait_for_timeout(3000)
        pagina.screenshot(path=SALIDA / f"{nombre}.png")
        print("ok", nombre)
    navegador.close()
