# For more information, please refer to https://aka.ms/vscode-docker-python
FROM python:3.12-slim

EXPOSE 8000

# Keeps Python from generating .pyc files in the container
ENV PYTHONDONTWRITEBYTECODE=1

# Turns off buffering for easier container logging
ENV PYTHONUNBUFFERED=1

# Install pip requirements
COPY requirements.txt .
RUN python -m pip install -r requirements.txt

RUN opentelemetry-bootstrap -a install

WORKDIR /app
COPY . /app

# Creates a non-root user with an explicit UID and adds permission to access the /app folder
# For more info, please refer to https://aka.ms/vscode-docker-python-configure-containers
RUN adduser -u 5678 --disabled-password --gecos "" appuser && chown -R appuser /app
USER appuser

# Start the FastAPI server with OpenTelemetry instrumentation in console
CMD ["opentelemetry-instrument", "uvicorn", "--host", "0.0.0.0", "--port", "8000", "chain_api:app"]

# Alternatively, run the FastAPI server with auto reload on save (but without OpenTelemetry instrumentation) 
# CMD ["uvicorn", "--host", "0.0.0.0", "--port", "8000", "--reload", "chain_api:app"]