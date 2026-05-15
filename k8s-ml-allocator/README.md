# ☸ Kubernetes ML Resource Allocator

Dynamic resource allocation system for ML training and inference workloads on Kubernetes.

---

## Prerequisites (install order matters)

### 1. Docker Desktop
Download from https://www.docker.com/products/docker-desktop/
Open the app and make sure the Docker engine is running (green icon in menu bar).

### 2. Homebrew (if not installed)
```bash
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
```

### 3. kubectl + minikube
```bash
brew install kubectl minikube
```

### 4. Python 3.10+
```bash
# Check first
python3 --version

# Install if needed
brew install python@3.12
```

---

## One-Click Setup

```bash
cd k8s-ml-allocator
bash setup.sh
```

This script automatically:
- Verifies Docker is running
- Installs kubectl and minikube via Homebrew
- Starts a minikube cluster (4 CPUs, 6GB RAM, Docker driver)
- Enables the **metrics-server** addon
- Builds the `ml-workload:latest` Docker image inside minikube
- Applies all Kubernetes manifests (namespace, RBAC, resource quotas)
- Creates a Python `.venv` and installs all dependencies

---

## Running the System

Open **three terminal tabs** in the project directory:

### Terminal 1 — Allocator Controller (live Rich dashboard)
```bash
source .venv/bin/activate
python -m allocator.controller
```
Polls every 15 seconds, shows a live table of pod CPU/memory usage, and applies scale-up/down patches automatically.

### Terminal 2 — REST API
```bash
source .venv/bin/activate
uvicorn api.main:app --reload --port 8000
```
API docs available at: **http://localhost:8000/docs**

### Terminal 3 — Submit Jobs & Check Cluster

**Submit a training job:**
```bash
curl -X POST http://localhost:8000/jobs \
  -H "Content-Type: application/json" \
  -d '{"name":"iris-classifier","job_type":"training","priority":"HIGH"}'
```

**Submit an inference deployment:**
```bash
curl -X POST http://localhost:8000/jobs \
  -H "Content-Type: application/json" \
  -d '{"name":"iris-server","job_type":"inference","priority":"MEDIUM","cpu_limit_m":300,"mem_limit_mi":256}'
```

**List all jobs:**
```bash
curl http://localhost:8000/jobs | python3 -m json.tool
```

**Live cluster metrics:**
```bash
curl http://localhost:8000/cluster | python3 -m json.tool
```

**Manually scale an inference deployment:**
```bash
curl -X POST http://localhost:8000/jobs/<job_id>/scale \
  -H "Content-Type: application/json" \
  -d '{"cpu_limit_m":800,"mem_limit_mi":768}'
```

**Queue depth:**
```bash
curl http://localhost:8000/queue
```

---

## Kubernetes Commands

```bash
# Watch pods in real time
kubectl get pods -n ml-workloads -w

# View pod resource usage (requires metrics-server)
kubectl top pods -n ml-workloads

# View jobs
kubectl get jobs -n ml-workloads

# View deployments
kubectl get deployments -n ml-workloads

# Check HPA status
kubectl get hpa -n ml-workloads

# View resource quota
kubectl describe resourcequota ml-quota -n ml-workloads

# Open minikube dashboard (browser)
minikube dashboard

# Apply a sample training job directly
kubectl apply -f k8s/training-job.yaml

# Apply inference deployment + HPA
kubectl apply -f k8s/inference-deployment.yaml
kubectl apply -f k8s/hpa.yaml

# Pod logs
kubectl logs -n ml-workloads <pod-name>
```

---

## Architecture

```
k8s-ml-allocator/
├── allocator/
│   ├── controller.py       ← Main control loop (poll → dispatch → scale → display)
│   ├── resource_monitor.py ← Queries metrics-server for live CPU/mem usage
│   ├── scaler.py           ← Scale-up/down logic with cooldown + K8s patch
│   └── priority_queue.py   ← Thread-safe min-heap job queue (HIGH/MEDIUM/LOW)
├── api/
│   └── main.py             ← FastAPI REST interface (submit, list, scale, monitor)
├── k8s/
│   ├── namespace.yaml      ← ml-workloads namespace
│   ├── rbac.yaml           ← ServiceAccount + Role + RoleBinding
│   ├── resource-quota.yaml ← Namespace CPU/mem caps + LimitRange defaults
│   ├── training-job.yaml   ← Sample K8s Job (one-shot training)
│   ├── inference-deployment.yaml ← Sample Deployment + Service
│   └── hpa.yaml            ← HPA: scale inference 1–5 replicas on CPU > 60%
├── ml_workloads/
│   ├── sample_training.py  ← RandomForest on Iris dataset, saves model.pkl
│   └── sample_inference.py ← Flask server: POST /predict → class + confidence
├── Dockerfile              ← Container for ML workloads
├── requirements.txt
└── setup.sh                ← Full Mac setup script
```

---

## How Dynamic Scaling Works

The `DynamicScaler` monitors each pod every 15 seconds:

| Condition | Consecutive cycles | Action |
|-----------|-------------------|--------|
| CPU > 75% OR Mem > 80% | 2 | Scale up limits +30% |
| CPU < 25% AND Mem < 30% | 2 | Scale down limits -20% |
| Otherwise | — | No action |

Hard limits: CPU 100m–2000m, Memory 128Mi–2048Mi per pod.

The HPA (`k8s/hpa.yaml`) independently scales inference **replicas** (1–5) based on average CPU utilisation across all pods.

---

## Troubleshooting

**`minikube start` fails** — Make sure Docker Desktop is open and the engine is running.

**`metrics-server` not available** — Run: `minikube addons enable metrics-server` then wait ~60s.

**Image pull error in pods** — Ensure you built the image inside minikube's Docker daemon: `eval $(minikube docker-env) && docker build -t ml-workload:latest .`

**`kubectl` can't find cluster** — Run: `minikube update-context`

**Stop everything:**
```bash
minikube stop       # pause cluster (preserves state)
minikube delete     # destroy cluster completely
```
