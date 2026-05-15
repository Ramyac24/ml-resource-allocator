# Dockerfile for ML workload containers
# Used by both sample_training and sample_inference jobs

FROM python:3.11-slim

WORKDIR /app

# Install ML dependencies only (keep image small)
RUN pip install --no-cache-dir \
    scikit-learn==1.5.0 \
    numpy==1.26.4 \
    pandas==2.2.2 \
    joblib==1.4.2 \
    flask==3.0.3

# Copy workload scripts
COPY ml_workloads/ ./ml_workloads/

# Default entrypoint — overridden per job in K8s manifests
ENTRYPOINT ["python"]
CMD ["ml_workloads/sample_training.py"]
