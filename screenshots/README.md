# Screenshots

Save your screenshots in this folder **using these exact file names**. `scripts/build_report.py` embeds them into the report automatically. Any that are missing show up as red "pending" placeholders.

| # | File name | What to capture | How |
|---|---|---|---|
| 1 | `01_mlflow_experiments.png` | MLflow experiment "heart-disease-classification" with the parent runs and their metrics columns | `mlflow ui --backend-store-uri sqlite:///mlflow.db` |
| 2 | `02_mlflow_nested_runs.png` | A parent run expanded to show its nested grid-search candidate runs | MLflow UI, click "+" next to a run |
| 3 | `03_mlflow_compare.png` | Run comparison (select the 3 model runs → Compare) | MLflow UI |
| 4 | `04_mlflow_artifacts.png` | The champion run's Artifacts tab (model, confusion matrix, ROC curves) | MLflow UI |
| 5 | `05_mlflow_registry.png` | Models tab → `heart-disease-classifier` showing version + `@champion` alias | MLflow UI |
| 6 | `06_pytest.png` | Terminal output of `pytest` (all passing) and `ruff check .` | terminal |
| 7 | `07_github_actions.png` | Successful workflow run graph (lint → test → train → docker) | GitHub → Actions |
| 8 | `08_github_artifacts.png` | Run summary page showing the metrics table and the uploaded artifacts | GitHub → Actions → run |
| 9 | `09_docker_build_run.png` | `docker build` + `docker run` + `docker ps` output | terminal |
| 10 | `10_docker_predict.png` | `curl`/`Invoke-RestMethod` call to `/predict` on the container and its JSON response | terminal |
| 11 | `11_swagger_ui.png` | `http://localhost:8000/docs` with a `/predict` "Try it out" response | browser |
| 12 | `12_k8s_get_all.png` | `kubectl -n heart-disease get all` (2 pods Running, LoadBalancer service) | terminal |
| 13 | `13_k8s_describe.png` | `kubectl -n heart-disease describe deployment heart-disease-api` (probes, replicas) | terminal |
| 14 | `14_k8s_predict.png` | `/predict` called through the LoadBalancer (`http://localhost:8080`) or the Ingress | terminal |
| 15 | `15_prometheus_targets.png` | `http://localhost:9090/targets` with the API target UP | browser |
| 16 | `16_grafana_dashboard.png` | The "Heart Disease API" Grafana dashboard after running `generate_traffic.py` | browser |
| 17 | `17_json_logs.png` | JSON request logs (`docker compose logs api` or `kubectl logs`) | terminal |
| 18 | `18_ci_failed_run.png` | A failed workflow run from a PR with a deliberately failing test: lint passed, test failed, train and docker skipped | GitHub → Actions |
| 19 | `19_ci_failed_log.png` | The failed test step's log showing the `AssertionError`, `1 failed, 42 passed` and exit code 1 | GitHub → Actions → job |

Tip: on Windows, press **Win + Shift + S** to capture a region, then save it as PNG with the names above.
