# For more information, please refer to https://aka.ms/vscode-docker-python
FROM python:3.12-slim

EXPOSE 8002

# Keeps Python from generating .pyc files in the container
ENV PYTHONDONTWRITEBYTECODE=1

# Turns off buffering for easier container logging
ENV PYTHONUNBUFFERED=1

# Install system dependencies (needed for some Python packages)
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Install pip requirements
COPY requirements.txt .
RUN python -m pip install -r requirements.txt

RUN opentelemetry-bootstrap -a install

WORKDIR /app
COPY . /app

# Make entrypoint script executable (before switching to non-root user)
RUN chmod +x /app/entrypoint.sh

# Creates a non-root user with an explicit UID and adds permission to access the /app folder
# For more info, please refer to https://aka.ms/vscode-docker-python-configure-containers
RUN adduser -u 5678 --disabled-password --gecos "" appuser && chown -R appuser /app

USER appuser

# Entrypoint script checks DEV_MODE env variable:
# - DEV_MODE=true: uvicorn with --reload (no OpenTelemetry)
# - DEV_MODE=false or unset: opentelemetry-instrument uvicorn (production mode)
ENTRYPOINT ["/app/entrypoint.sh"]
