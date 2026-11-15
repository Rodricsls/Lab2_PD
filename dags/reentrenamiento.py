"""Reentrenamiento del modelo (se dispara con POST /admin/reentrenar).

entrenar_challenger → aplicar_decision → registrar_resultado

- `entrenar_challenger` entrena con el histórico y las observaciones registradas, compara
  en la ventana reciente y registra la versión nueva en MLflow.
- `aplicar_decision` mueve el alias @champion solo si el criterio lo aprueba; si no, deja
  la versión marcada como descartada. Un fallo antes de este paso no toca el vigente.
- `registrar_resultado` (siempre) escribe el resultado en la base del producto para
  `GET /admin/reentrenar/{id}`; si algo falló, lo registra y deja la ejecución en rojo.
"""

from datetime import datetime, timedelta

from airflow.sdk import dag, task


@dag(
    dag_id="reentrenamiento",
    schedule=None,
    start_date=datetime(2024, 1, 1),
    catchup=False,
    max_active_runs=1,
    is_paused_upon_creation=False,
    dagrun_timeout=timedelta(minutes=10),
    default_args={"retries": 0},
    tags=["mlflow", "reentrenamiento"],
)
def reentrenamiento():
    @task
    def entrenar_challenger(run_id=None) -> dict:
        from novamercado.operacion import reentrenar

        return reentrenar(run_id)

    @task
    def aplicar_decision(propuesta: dict) -> dict:
        from novamercado.operacion import aplicar_decision

        return aplicar_decision(propuesta)

    @task(trigger_rule="all_done")
    def registrar_resultado(run_id=None, ti=None) -> dict:
        from airflow.sdk.exceptions import AirflowFailException

        from novamercado.operacion import registrar_resultado

        resultado = ti.xcom_pull(task_ids="aplicar_decision")
        registrar_resultado(run_id, resultado, None if resultado else "Ver los logs de la ejecución.")
        if resultado is None:
            raise AirflowFailException("El reentrenamiento falló; el modelo vigente no cambió.")
        return resultado

    aplicar_decision(entrenar_challenger()) >> registrar_resultado()


reentrenamiento()
