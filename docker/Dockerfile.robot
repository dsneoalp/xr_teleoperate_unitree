# PC / sim robot: Python 3.10 software Portal FFI + CycloneDDS + unitree_sdk2.
# Same stack as control_fix robot. Not for G1 Jetson MMAPI (see Dockerfile.robot-g1).
# Requires xr-teleop:portal-wheel.
ARG PORTAL_WHEEL_IMAGE=xr-teleop:portal-wheel
ARG UNITREE_SDK2_REF=65691c8a8bc53b98d3976dba4dbf9d5d20b2e7f5
ARG TELEIMAGER_REF=57cf2a40572227273fa001cd17833b755331ec97

FROM ${PORTAL_WHEEL_IMAGE} AS portal-wheel

FROM python:3.10-slim-bookworm
ARG UNITREE_SDK2_REF
ARG TELEIMAGER_REF

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

COPY --from=portal-wheel /wheels /tmp/wheels
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

# setup.py has no package_data for utils/lib/*.so; pip install git+… drops crc_*.so.
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
