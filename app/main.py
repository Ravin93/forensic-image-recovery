from fastapi import FastAPI

from app.api.routes.pipeline import router as pipeline_router
from app.api.routes.report import router as report_router
from app.api.routes.files import router as files_router
from app.api.routes.health import router as health_router
from app.api.routes.carving import router as carving_router
from app.api.routes.validation import router as validation_router
from app.api.routes.corruption import router as corruption_router
from app.api.routes.reconstruction import router as reconstruction_router
from app.api.routes.evaluation import router as evaluation_router

app = FastAPI(
    title="Forensic Image Recovery API",
    description="Pipeline complet de récupération et reconstruction d'images",
    version="1.0.0"
)

app.include_router(pipeline_router)
app.include_router(report_router)
app.include_router(files_router)
app.include_router(health_router)
app.include_router(carving_router)
app.include_router(validation_router)
app.include_router(corruption_router)
app.include_router(reconstruction_router)
app.include_router(evaluation_router)