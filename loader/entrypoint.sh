#!/bin/bash
set -e

if [ "$DEV_MODE" = "true" ]; then
    echo "Starting in DEV mode (with --reload, without OpenTelemetry)"
    exec uvicorn --host 0.0.0.0 --port 8002 --reload loader_api:app
else
    echo "Starting in PRODUCTION mode (with OpenTelemetry instrumentation)"
    exec opentelemetry-instrument uvicorn --host 0.0.0.0 --port 8002 loader_api:app
fi
