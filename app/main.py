import time

import structlog
import uvicorn
import logging

from asgi_correlation_id import CorrelationIdMiddleware
from asgi_correlation_id.context import correlation_id
from app.core.config import settings
from app.core.custom_logging import setup_logging
from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from app.api.recommendations import router as recommendation_router
from uvicorn.protocols.utils import get_path_with_query_string

LOG_JSON_FORMAT = settings.LOG_JSON_FORMAT
LOG_LEVEL = settings.LOG_LEVEL
setup_logging(json_logs=LOG_JSON_FORMAT, log_level=LOG_LEVEL)

access_logger = structlog.stdlib.get_logger("api.access")
app = FastAPI(title="BK Sharing - Recommendation System", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Cho phép tất cả các domain gọi API (FE)
    allow_credentials=True,
    allow_methods=["*"],  # Cho phép tất cả các phương thức HTTP
    allow_headers=["*"],  # Cho phép tất cả headers
)

app.include_router(recommendation_router, prefix="/recommendation/api/v1", tags=["recommendation"])

@app.middleware("http")
async def logging_middleware(request: Request, call_next) -> Response:
    structlog.contextvars.clear_contextvars()
    # These context vars will be added to all log entries emitted during the request
    request_id = correlation_id.get()
    structlog.contextvars.bind_contextvars(request_id=request_id)

    start_time = time.perf_counter_ns()
    # If the call_next raises an error, we still want to return our own 500 response,
    # so we can add headers to it (process time, request ID...)
    response = Response(status_code=500)
    try:
        response = await call_next(request)
    except Exception:
        # TODO: Validate that we don't swallow exceptions (unit test?)
        structlog.stdlib.get_logger("api.error").exception("Uncaught exception")
        raise
    finally:
        process_time = time.perf_counter_ns() - start_time
        status_code = response.status_code
        url = get_path_with_query_string(request.scope)
        client_host = request.client.host
        client_port = request.client.port
        http_method = request.method
        http_version = request.scope["http_version"]
        # Recreate the Uvicorn access log format, but add all parameters as structured information
        access_logger.info(
            f"""{client_host}:{client_port} - "{http_method} {url} HTTP/{http_version}" {status_code}""",
            http={
                "url": str(request.url),
                "status_code": status_code,
                "method": http_method,
                "request_id": request_id,
                "version": http_version,
            },
            network={"client": {"ip": client_host, "port": client_port}},
            duration=process_time,
        )
        response.headers["X-Process-Time"] = str(process_time / 10 ** 9)
        return response

# This middleware must be placed after the logging, to populate the context with the request ID
# NOTE: Why last??
# Answer: middlewares are applied in the reverse order of when they are added (you can verify this
# by debugging `app.middleware_stack` and recursively drilling down the `app` property).
app.add_middleware(CorrelationIdMiddleware)

@app.get("/")
def hello():
    custom_structlog_logger = structlog.stdlib.get_logger("my.structlog.logger")
    custom_structlog_logger.info("This is an info message from Structlog")
    custom_structlog_logger.warning("This is a warning message from Structlog, with attributes", an_extra="attribute")
    custom_structlog_logger.error("This is an error message from Structlog")

    custom_logging_logger = logging.getLogger("my.logging.logger")
    custom_logging_logger.info("This is an info message from standard logger")
    custom_logging_logger.warning("This is a warning message from standard logger, with attributes", extra={"another_extra": "attribute"})

    return "Hello, World!"

if __name__ == "__main__":
    uvicorn.run(app, host=settings.HOST, port=8000, log_config=None)
