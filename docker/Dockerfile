# Multi-stage: mock (LiveKit only), robot (DDS + SDK), operator (README conda env tv).
# livekit-portal PyPI wheels are cp312-only; we build the FFI from git for Python 3.10.
# unitree_sdk2_python is pip-installed at a pinned commit — no host checkouts.
#   docker compose -f docker/compose.yml build mock operator robot

ARG PORTAL_REF=4fb4385f3e4bfc8e5019abce64be9042d338166e
ARG UNITREE_SDK2_REF=65691c8a8bc53b98d3976dba4dbf9d5d20b2e7f5
ARG TELEIMAGER_REF=57cf2a40572227273fa001cd17833b755331ec97
ARG TELEVUER_REF=766de45e74373ae0ea66321d942ce538385655a5
ARG DEX_RETARGETING_REF=d7753d38c9ff11f80bafea6cd168351fd3db9b0e

# ----- livekit-portal wheel (Python 3.10 + Rust FFI) -----
FROM python:3.10-bookworm AS portal-build
ARG PORTAL_REF
ENV DEBIAN_FRONTEND=noninteractive \
    PIP_DISABLE_PIP_VERSION_CHECK=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        git curl ca-certificates build-essential pkg-config \
        clang libclang-dev \
        libssl-dev libglib2.0-dev libx11-dev libxext-dev libxfixes-dev \
        libxdamage-dev libxrandr-dev libxcomposite-dev libgl1-mesa-dev \
        libgbm-dev libdrm-dev libva-dev \
    && curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs \
        | sh -s -- -y --default-toolchain stable \
    && rm -rf /var/lib/apt/lists/*

ENV PATH=/root/.cargo/bin:$PATH

RUN git clone https://github.com/livekit/portal.git /src/portal \
    && git -C /src/portal checkout "${PORTAL_REF}"

WORKDIR /src/portal
RUN bash scripts/build_ffi_python.sh release

WORKDIR /src/portal/python/packages/livekit-portal
RUN pip install --no-cache-dir hatchling hatch-vcs \
    && pip wheel --no-deps --no-cache-dir . -w /wheels

FROM python:3.10-slim-bookworm AS base

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app \
    PIP_DISABLE_PIP_VERSION_CHECK=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates \
        libglib2.0-0 \
        libgomp1 \
        libssl3 \
        libx11-6 \
        libxext6 \
        libxfixes3 \
        libgl1 \
        libgbm1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY --from=portal-build /wheels /tmp/wheels
RUN pip install --no-cache-dir \
        "numpy<2.0.0" \
        pyyaml \
        python-dotenv \
        "livekit-api>=0.7" \
        logging-mp \
        /tmp/wheels/*.whl \
    && rm -rf /tmp/wheels

COPY teleop/ /app/teleop/
WORKDIR /app/teleop
RUN python -c "from livekit.portal import OperatorConfig, RobotConfig; \
    OperatorConfig.from_yaml_file('/app/teleop/portal.yaml', 'x'); \
    RobotConfig.from_yaml_file('/app/teleop/portal.yaml', 'x')"

# ----- mock: echo actions / test-pattern video, no Unitree SDK -----
FROM base AS mock
CMD ["python", "portal_robot_mock.py"]

# ----- robot: README 1.2 unitree_sdk2_python at a pinned commit -----
# setup.py has no package_data for utils/lib/*.so; pip install git+… drops crc_*.so.
FROM base AS robot
ARG UNITREE_SDK2_REF
ARG TELEIMAGER_REF
RUN apt-get update && apt-get install -y --no-install-recommends git cmake build-essential \
    && git clone --branch releases/0.10.x https://github.com/eclipse-cyclonedds/cyclonedds.git /tmp/cyclonedds \
    && mkdir /tmp/cyclonedds/build && cd /tmp/cyclonedds/build \
    && cmake -DCMAKE_INSTALL_PREFIX=/usr/local .. \
    && make -j$(nproc) install \
    && git clone https://github.com/unitreerobotics/unitree_sdk2_python.git /tmp/unitree_sdk2_python \
    && git -C /tmp/unitree_sdk2_python checkout "${UNITREE_SDK2_REF}" \
    && pip install --no-cache-dir --no-deps /tmp/unitree_sdk2_python \
    && env CYCLONEDDS_HOME=/usr/local pip install --no-cache-dir \
        "cyclonedds==0.10.2" \
        "opencv-python-headless>=4.8,<4.11" \
        "numpy<2.0.0" \
        pyzmq \
    && rm -rf /app/teleop/teleimager \
    && git clone https://github.com/unitreerobotics/teleimager.git /tmp/teleimager \
    && git -C /tmp/teleimager checkout "${TELEIMAGER_REF}" \
    && pip install --no-cache-dir --no-deps /tmp/teleimager \
    && python -c "\
import os, shutil, unitree_sdk2py.utils.crc as crc; \
src = '/tmp/unitree_sdk2_python/unitree_sdk2py/utils/lib'; \
dst = os.path.join(os.path.dirname(crc.__file__), 'lib'); \
os.makedirs(dst, exist_ok=True); \
shutil.copytree(src, dst, dirs_exist_ok=True)" \
    && python -c "from unitree_sdk2py.utils.crc import CRC; CRC()" \
    && python -c "from teleimager.image_client import ImageClient" \
    && rm -rf /tmp/unitree_sdk2_python /tmp/cyclonedds /tmp/teleimager /root/.cache/pip \
    && apt-get purge -y git cmake build-essential \
    && apt-get autoremove -y \
    && rm -rf /var/lib/apt/lists/*
CMD ["python", "teleop_robot.py", "--ee", "dex3", "--arm", "G1_29"]

# ----- operator: README 1.1 conda env tv (not FROM slim; one Python only) -----
#   conda create -n tv python=3.10 pinocchio=3.1.0 numpy=1.26.4 -c conda-forge
#   pip install -e teleop/teleimager --no-deps
#   pip install -e teleop/televuer
#   pip install -e teleop/robot_control/dex-retargeting
#   pip install -r requirements.txt
FROM debian:bookworm-slim AS operator
ARG TELEIMAGER_REF
ARG TELEVUER_REF
ARG DEX_RETARGETING_REF

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
        git \
        libgl1 \
        libglib2.0-0 \
        libgomp1 \
        libsm6 \
        libxext6 \
        libxrender1 \
        libssl3 \
        libx11-6 \
        libxfixes3 \
        libgbm1 \
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
COPY teleop/ /app/teleop/
COPY requirements.txt /app/requirements.txt
COPY assets/g1 /app/assets/g1
COPY assets/unitree_hand /app/assets/unitree_hand

# Empty submodule dirs in the build context; clone the Superproject gitlink SHAs.
RUN rm -rf /app/teleop/teleimager /app/teleop/televuer /app/teleop/robot_control/dex-retargeting \
    && git clone https://github.com/unitreerobotics/teleimager.git /app/teleop/teleimager \
    && git -C /app/teleop/teleimager checkout "${TELEIMAGER_REF}" \
    && git clone https://github.com/unitreerobotics/televuer.git /app/teleop/televuer \
    && git -C /app/teleop/televuer checkout "${TELEVUER_REF}" \
    && git clone https://github.com/silencht/dex-retargeting.git /app/teleop/robot_control/dex-retargeting \
    && git -C /app/teleop/robot_control/dex-retargeting checkout "${DEX_RETARGETING_REF}" \
    && pip install --no-cache-dir -e /app/teleop/teleimager --no-deps \
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

COPY --from=portal-build /wheels /tmp/wheels
RUN pip install --no-cache-dir /tmp/wheels/*.whl \
    && python -c "from livekit.portal import OperatorConfig, RobotConfig; \
        OperatorConfig.from_yaml_file('/app/teleop/portal.yaml', 'x'); \
        RobotConfig.from_yaml_file('/app/teleop/portal.yaml', 'x')" \
    && rm -rf /tmp/wheels /root/.cache/pip

COPY docker/entrypoint-operator.sh /entrypoint-operator.sh
RUN chmod +x /entrypoint-operator.sh \
    && mkdir -p /certs

WORKDIR /app/teleop

ENV PYTHONPATH=/app \
    XR_TELEOP_CERT=/certs/cert.pem \
    XR_TELEOP_KEY=/certs/key.pem

ENTRYPOINT ["/entrypoint-operator.sh"]
CMD ["--arm", "G1_29", "--ee", "dex3", "--input-mode", "hand"]