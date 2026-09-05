# Multi-stage: mock (LiveKit only), robot (DDS + SDK), operator (TeleVuer + IK).
# livekit-portal FFI is injected as extra context `portalpkg` (prebuilt .so).
# Python 3.10: cyclonedds==0.10.2 (Unitree pin) has binary wheels; 3.11 is sdist-only.
#   docker compose build mock operator robot

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

# ----- robot: Unitree DDS controllers -----
FROM base AS robot
COPY --from=unitreesdk / /opt/unitree_sdk2_python
RUN pip install --no-cache-dir --no-deps /opt/unitree_sdk2_python \
    && pip install --no-cache-dir \
        "cyclonedds==0.10.2" \
        "opencv-python-headless>=4.8,<4.11" \
        "numpy<2.0.0"
CMD ["python", "teleop_robot.py", "--ee", "dex3", "--arm", "G1_29"]

# ----- operator: TeleVuer + Pinocchio IK (conda-forge pinocchio has casadi; pip `pin` does not)
FROM base AS operator
RUN apt-get update && apt-get install -y --no-install-recommends \
        curl \
        bzip2 \
        libgl1 \
        openssl \
    && curl -Ls https://micro.mamba.pm/api/micromamba/linux-64/latest \
        | tar -xvj -C /usr/local bin/micromamba \
    && /usr/local/bin/micromamba --version \
    && MAMBA_ROOT_PREFIX=/opt/conda /usr/local/bin/micromamba create -y -p /opt/conda -c conda-forge \
        python=3.10 \
        pinocchio=3.1.0 \
        casadi=3.6.7 \
        "numpy<2" \
    && /opt/conda/bin/pip install --no-cache-dir \
        pyyaml \
        python-dotenv \
        "livekit-api>=0.7" \
        logging-mp \
        meshcat \
        opencv-python-headless \
        sshkeyboard \
        "params-proto==2.13.2" \
        "vuer[all]==0.0.60" \
    && /opt/conda/bin/pip install --no-cache-dir /app/teleop/televuer \
    && MAMBA_ROOT_PREFIX=/opt/conda /usr/local/bin/micromamba clean -a -y \
    && rm -f /usr/local/bin/micromamba \
    && apt-get purge -y curl bzip2 \
    && apt-get autoremove -y \
    && rm -rf /var/lib/apt/lists/*

ENV PATH=/opt/conda/bin:$PATH
RUN pip install --no-cache-dir matplotlib rerun-sdk "numpy<2.0.0"

COPY assets/g1 /app/assets/g1
COPY docker/entrypoint-operator.sh /entrypoint-operator.sh
RUN chmod +x /entrypoint-operator.sh \
    && mkdir -p /certs

ENV XR_TELEOP_CERT=/certs/cert.pem \
    XR_TELEOP_KEY=/certs/key.pem

ENTRYPOINT ["/entrypoint-operator.sh"]
CMD ["--arm", "G1_29"]
