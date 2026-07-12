from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes.pipeline import router as pipeline_router
from app.api.routes.report import router as report_router
from app.api.routes.files import router as files_router
from app.api.routes.health import router as health_router
from app.api.routes.carving import router as carving_router
from app.api.routes.validation import router as validation_router
from app.api.routes.corruption import router as corruption_router
from app.api.routes.reconstruction import router as reconstruction_router
from app.api.routes.evaluation import router as evaluation_router
from app.api.routes.analysis import router as analysis_router
from app.api.routes.benchmark import router as benchmark_router
from app.api.routes.audit import router as audit_router

from app.core.file_cleanup import schedule_cleanup_on_startup as _sched_cleanup

app = FastAPI(
    title="Forensic Image Recovery API",
    description=(
        "Pipeline complet de récupération et reconstruction d'images forensiques.\n\n"
        "Endpoint principal : `POST /pipeline/corrupt-and-repair`\n"
        "Upload une image → corruption réaliste → reconstruction multi-essais → score."
    ),
    version="1.0.0",
)

# Tous les routers — pipeline_router inclut maintenant /pipeline/corrupt-and-repair
_sched_cleanup(app)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
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
app.include_router(analysis_router)
app.include_router(benchmark_router)
app.include_router(audit_router)