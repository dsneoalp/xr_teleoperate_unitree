# Multi-stage: mock (LiveKit only), robot (DDS + SDK), operator (README conda env tv).
# livekit-portal FFI is injected as extra context `portalpkg` (prebuilt .so).
# Python 3.10: cyclonedds==0.10.2 (Unitree pin) has binary wheels; 3.11 is sdist-only.
#   docker compose -f docker/compose.yml build mock operator robot

FROM python:3.10-slim-bookworm AS base

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app:/opt/livekit-portal \
    PIP_DISABLE_PIP_VERSION_CHECK=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates \
        libglib2.0-0 \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY --from=portalpkg / /opt/livekit-portal
RUN pip install --no-cache-dir \
        "numpy<2.0.0" \
        pyyaml \
        python-dotenv \
        "livekit-api>=0.7" \
        logging-mp

COPY teleop/ /app/teleop/
WORKDIR /app/teleop

# ----- mock: echo actions / test-pattern video, no Unitree SDK -----
FROM base AS mock
CMD ["python", "portal_robot_mock.py"]

# ----- robot: README 1.2 unitree_sdk2_python -----
FROM base AS robot
COPY --from=unitreesdk / /opt/unitree_sdk2_python
RUN pip install --no-cache-dir --no-deps /opt/unitree_sdk2_python \
    && pip install --no-cache-dir \
        "cyclonedds==0.10.2" \
        "opencv-python-headless>=4.8,<4.11" \
        "numpy<2.0.0"
CMD ["python", "teleop_robot.py", "--ee", "dex3", "--arm", "G1_29"]

# ----- operator: README 1.1 conda env tv (not FROM slim; one Python only) -----
#   conda create -n tv python=3.10 pinocchio=3.1.0 numpy=1.26.4 -c conda-forge
#   pip install -e teleop/teleimager --no-deps
#   pip install -e teleop/televuer
#   pip install -e teleop/robot_control/dex-retargeting
#   pip install -r requirements.txt
FROM debian:bookworm-slim AS operator

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app:/opt/livekit-portal \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    MAMBA_ROOT_PREFIX=/opt/conda \
    PATH=/opt/conda/bin:$PATH

RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates \
        curl \
        bzip2 \
        libgl1 \
        libglib2.0-0 \
        libgomp1 \
        libsm6 \
        libxext6 \
        libxrender1 \
        openssl \
    && curl -Ls https://micro.mamba.pm/api/micromamba/linux-64/latest \
        | tar -xvj -C /usr/local bin/micromamba \
    && micromamba create -y -p /opt/conda -c conda-forge \
        python=3.10 \
        pinocchio=3.1.0 \
        numpy=1.26.4 \
    && micromamba clean -a -y \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY --from=portalpkg / /opt/livekit-portal
COPY teleop/ /app/teleop/
COPY requirements.txt /app/requirements.txt
COPY assets/g1 /app/assets/g1
COPY assets/unitree_hand /app/assets/unitree_hand

# README pip, then Portal extras (not in upstream README). dex-retargeting
# may pip-install pin; restore conda pinocchio so pinocchio.casadi stays.
RUN pip install --no-cache-dir -e /app/teleop/teleimager --no-deps \
    && pip install --no-cache-dir -e /app/teleop/televuer \
    && pip install --no-cache-dir -e /app/teleop/robot_control/dex-retargeting \
    && pip install --no-cache-dir -r /app/requirements.txt \
    && pip install --no-cache-dir \
        pyyaml \
        python-dotenv \
        "livekit-api>=0.7" \
        logging-mp \
    && python -c "import pinocchio.casadi" \
        || micromamba install -y -p /opt/conda -c conda-forge pinocchio=3.1.0 numpy=1.26.4 \
    && python -c "import pinocchio.casadi; import torch; from dex_retargeting import RetargetingConfig" \
    && micromamba clean -a -y \
    && rm -f /usr/local/bin/micromamba \
    && apt-get purge -y curl bzip2 \
    && apt-get autoremove -y \
    && rm -rf /var/lib/apt/lists/* /root/.cache/pip

# vuer==0.0.60 (from televuer) needs params-proto 2.x; current PyPI 3.3 drops Flag.
# Host conda env tv has params-proto==2.13.2.
RUN pip install --no-cache-dir "params-proto==2.13.2" \
    && rm -rf /root/.cache/pip

COPY docker/entrypoint-operator.sh /entrypoint-operator.sh
RUN chmod +x /entrypoint-operator.sh \
    && mkdir -p /certs

WORKDIR /app/teleop

ENV XR_TELEOP_CERT=/certs/cert.pem \
    XR_TELEOP_KEY=/certs/key.pem

ENTRYPOINT ["/entrypoint-operator.sh"]
CMD ["--arm", "G1_29", "--ee", "dex3", "--input-mode", "hand"]
