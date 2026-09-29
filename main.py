import uuid
from contextlib import asynccontextmanager
import asyncio
import httpx
import structlog
from fastapi import FastAPI, Request, Response, status
from sqlalchemy.ext.asyncio import create_async_engine
from sqlalchemy import text
from dotenv import load_dotenv
import os
from models import Base
from logging_config import configure_logging

load_dotenv()
configure_logging()
log = structlog.get_logger()


@asynccontextmanager
async def lifespan(app: FastAPI):
    # --- startup ---
    log.info("startup_begin")

    app.state.db_engine = create_async_engine(
        os.environ["DATABASE_URL"],
        pool_size=5,
        max_overflow=5,
        pool_pre_ping=True,
        pool_timeout=10,
        connect_args={"timeout": 10, "command_timeout": 10},
    )
    async with app.state.db_engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    app.state.http_client = httpx.AsyncClient(timeout=30.0)

    # fail fast if the DB is unreachable, rather than discovering it on request 1
    async with app.state.db_engine.connect() as conn:
        await asyncio.wait_for(conn.execute(text("SELECT 1")), timeout=10.0)

    log.info("startup_complete")
    yield
    # --- shutdown ---
    log.info("shutdown_begin")
    await app.state.db_engine.dispose()
    await app.state.http_client.aclose()
    log.info("shutdown_complete")


app = FastAPI(lifespan=lifespan)


@app.middleware("http")
async def correlation_id_middleware(request: Request, call_next):
    correlation_id = request.headers.get("X-Request-ID", str(uuid.uuid4()))
    structlog.contextvars.clear_contextvars()
    structlog.contextvars.bind_contextvars(correlation_id=correlation_id)

    log.info("request_started", path=request.url.path)
    response = await call_next(request)
    log.info("request_finished", status_code=response.status_code)

    response.headers["X-Request-ID"] = correlation_id
    return response


@app.get("/")
async def root():
    log.info("root_handler_called")
    return {"status": "alive"}


@app.get("/healthz")
async def liveness():
    """Is the process itself alive? No dependency checks — an
    orchestrator uses this to decide whether to restart the container."""
    return {"status": "alive"}


@app.get("/readyz")
async def readiness(request: Request, response: Response):
    """Can this instance actually serve traffic right now?
    Checks the one dependency that matters: the database."""
    try:
        async with request.app.state.db_engine.connect() as conn:
            await conn.execute(text("SELECT 1"))
        return {"status": "ready", "db": "ok"}
    except Exception as e:
        log.error("readiness_check_failed", error=str(e))
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
        return {"status": "not_ready", "db": "unreachable"}