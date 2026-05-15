#!/bin/bash
# ============================================================
#  Kubernetes ML Resource Allocator — Full Mac Setup Script
#  Run once from project root:  bash setup.sh
# ============================================================
set -e

GREEN='\033[0;32m'; CYAN='\033[0;36m'
YELLOW='\033[1;33m'; RED='\033[0;31m'; NC='\033[0m'

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="$PROJECT_DIR/.venv"
NAMESPACE="ml-workloads"

echo ""
echo -e "${CYAN}╔══════════════════════════════════════════════════╗${NC}"
echo -e "${CYAN}║   ☸  Kubernetes ML Resource Allocator — Setup   ║${NC}"
echo -e "${CYAN}╚══════════════════════════════════════════════════╝${NC}"
echo ""

# ── 1. Homebrew ───────────────────────────────────────────────────────────────
echo -e "${YELLOW}[1/8] Checking Homebrew…${NC}"
if ! command -v brew &>/dev/null; then
    echo "Installing Homebrew…"
    /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
fi
echo -e "${GREEN}  ✓ Homebrew OK${NC}"

# ── 2. Docker Desktop ─────────────────────────────────────────────────────────
echo ""
echo -e "${YELLOW}[2/8] Checking Docker…${NC}"
if ! command -v docker &>/dev/null; then
    echo -e "${RED}  ✗ Docker not found.${NC}"
    echo "  Install Docker Desktop from: https://www.docker.com/products/docker-desktop/"
    echo "  Then re-run this script."
    exit 1
fi
if ! docker info &>/dev/null; then
    echo -e "${YELLOW}  ⚠ Docker daemon not running — start Docker Desktop first.${NC}"
    echo "  Waiting 10s for Docker to start…"
    sleep 10
    if ! docker info &>/dev/null; then
        echo -e "${RED}  ✗ Docker still not running. Please open Docker Desktop and retry.${NC}"
        exit 1
    fi
fi
echo -e "${GREEN}  ✓ Docker OK${NC}"

# ── 3. kubectl ────────────────────────────────────────────────────────────────
echo ""
echo -e "${YELLOW}[3/8] Installing kubectl…${NC}"
if ! command -v kubectl &>/dev/null; then
    brew install kubectl
fi
echo -e "${GREEN}  ✓ kubectl $(kubectl version --client --short 2>/dev/null | head -1)${NC}"

# ── 4. minikube ───────────────────────────────────────────────────────────────
echo ""
echo -e "${YELLOW}[4/8] Installing minikube…${NC}"
if ! command -v minikube &>/dev/null; then
    brew install minikube
fi
echo -e "${GREEN}  ✓ minikube $(minikube version --short 2>/dev/null)${NC}"

# ── 5. Start minikube cluster ─────────────────────────────────────────────────
echo ""
echo -e "${YELLOW}[5/8] Starting minikube cluster…${NC}"
MINIKUBE_STATUS=$(minikube status --format='{{.Host}}' 2>/dev/null || echo "Stopped")
if [[ "$MINIKUBE_STATUS" != "Running" ]]; then
    echo "  Allocating: 4 CPUs, 6GB RAM, 20GB disk…"
    minikube start \
        --driver=docker \
        --cpus=4 \
        --memory=6144 \
        --disk-size=20g \
        --kubernetes-version=stable \
        --addons=metrics-server
    echo -e "${GREEN}  ✓ minikube cluster started${NC}"
else
    echo -e "${GREEN}  ✓ minikube already running${NC}"
    # Ensure metrics-server is enabled
    minikube addons enable metrics-server 2>/dev/null || true
fi

# ── 6. Build Docker image inside minikube ────────────────────────────────────
echo ""
echo -e "${YELLOW}[6/8] Building ML workload Docker image…${NC}"
eval "$(minikube docker-env)"    # point Docker CLI to minikube's daemon
cd "$PROJECT_DIR"
docker build -t ml-workload:latest . --quiet
echo -e "${GREEN}  ✓ Image ml-workload:latest built inside minikube${NC}"

# ── 7. Apply Kubernetes manifests ─────────────────────────────────────────────
echo ""
echo -e "${YELLOW}[7/8] Applying Kubernetes manifests…${NC}"
kubectl apply -f k8s/namespace.yaml
kubectl apply -f k8s/rbac.yaml
kubectl apply -f k8s/resource-quota.yaml
echo -e "${GREEN}  ✓ Namespace, RBAC, and ResourceQuota applied${NC}"

# ── 8. Python venv ────────────────────────────────────────────────────────────
echo ""
echo -e "${YELLOW}[8/8] Setting up Python virtual environment…${NC}"
PYTHON_CMD=""
for cmd in python3.12 python3.11 python3.10 python3; do
    if command -v $cmd &>/dev/null; then
        PYTHON_CMD=$cmd; break
    fi
done
if [ -z "$PYTHON_CMD" ]; then
    echo -e "${RED}ERROR: Python 3.10+ not found. Install: brew install python@3.12${NC}"
    exit 1
fi

if [ ! -d "$VENV_DIR" ]; then
    $PYTHON_CMD -m venv "$VENV_DIR"
fi
source "$VENV_DIR/bin/activate"
pip install --upgrade pip --quiet
pip install -r "$PROJECT_DIR/requirements.txt" --quiet
echo -e "${GREEN}  ✓ Python venv ready${NC}"

# ── Done ──────────────────────────────────────────────────────────────────────
echo ""
echo -e "${CYAN}╔══════════════════════════════════════════════════════════════╗${NC}"
echo -e "${CYAN}║  ✅  Setup complete! Here's how to run everything:           ║${NC}"
echo -e "${CYAN}║                                                              ║${NC}"
echo -e "${CYAN}║  source .venv/bin/activate                                   ║${NC}"
echo -e "${CYAN}║                                                              ║${NC}"
echo -e "${CYAN}║  # Terminal 1 — Live controller (Rich dashboard)             ║${NC}"
echo -e "${CYAN}║  python -m allocator.controller                              ║${NC}"
echo -e "${CYAN}║                                                              ║${NC}"
echo -e "${CYAN}║  # Terminal 2 — REST API                                     ║${NC}"
echo -e "${CYAN}║  uvicorn api.main:app --reload --port 8000                   ║${NC}"
echo -e "${CYAN}║                                                              ║${NC}"
echo -e "${CYAN}║  # Terminal 3 — Submit a training job                        ║${NC}"
echo -e "${CYAN}║  curl -X POST http://localhost:8000/jobs \                   ║${NC}"
echo -e "${CYAN}║    -H 'Content-Type: application/json' \                     ║${NC}"
echo -e "${CYAN}║    -d '{\"name\":\"iris\",\"job_type\":\"training\",\"priority\":\"HIGH\"}'  ║${NC}"
echo -e "${CYAN}║                                                              ║${NC}"
echo -e "${CYAN}║  # View cluster metrics                                      ║${NC}"
echo -e "${CYAN}║  curl http://localhost:8000/cluster | python3 -m json.tool   ║${NC}"
echo -e "${CYAN}║                                                              ║${NC}"
echo -e "${CYAN}║  # K8s dashboards                                            ║${NC}"
echo -e "${CYAN}║  kubectl get pods -n ml-workloads                            ║${NC}"
echo -e "${CYAN}║  minikube dashboard                                          ║${NC}"
echo -e "${CYAN}╚══════════════════════════════════════════════════════════════╝${NC}"
echo ""
