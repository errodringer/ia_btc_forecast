from pathlib import Path


# Configuración de rutas
BASE_PATH = Path(__file__).parent.parent.parent / "data"
HISTORICAL_PATH = BASE_PATH / "historical"
CURRENT_PATH = BASE_PATH / "current"
REPORTS_PATH = BASE_PATH / "reports"
PROCESSED_PATH = BASE_PATH / "processed"
FEATURES_PATH = BASE_PATH / "features"
MODELS_PATH = BASE_PATH / "models"
PLOTS_PATH = REPORTS_PATH / "plots"
PREDICTIONS_PATH = BASE_PATH / "predictions"
