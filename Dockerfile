# ---- Builder Stage ----
    FROM python:3.9-slim as builder

    WORKDIR /code
    
    # Install build dependencies (if needed for compiling any packages)
    RUN apt-get update && apt-get install -y --no-install-recommends \
        gcc \
        build-essential && \
        rm -rf /var/lib/apt/lists/*
    
    # Copy requirements file and install dependencies into a custom location
    COPY ./requirements.txt .
    RUN pip install --no-cache-dir --prefix=/install -r requirements.txt
    
    # ---- Final Stage ----
    FROM python:3.9-slim
    
    WORKDIR /code
    
    # Copy the installed Python packages from the builder stage
    COPY --from=builder /install /usr/local
    
    # Copy the application code and environment file
    COPY ./app /code/app
    COPY .env /code/.env
    
    CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8082"]
    