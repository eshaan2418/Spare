# Reproducible environment (CPU). MLX training is Apple-silicon only and not containerised.
FROM python:3.12-slim
RUN apt-get update && apt-get install -y --no-install-recommends \
      libgl1 libglu1-mesa libxrender1 libxcursor1 libxft2 libxinerama1 libgomp1 wget ca-certificates \
      && rm -rf /var/lib/apt/lists/*
# PrusaSlicer CLI (AppImage, extracted)
RUN wget -q https://github.com/prusa3d/PrusaSlicer/releases/download/version_2.9.2/PrusaSlicer-2.9.2+linux-x64-older-distros-GTK3-202503191424.AppImage -O /tmp/ps.AppImage \
    && chmod +x /tmp/ps.AppImage && cd /opt && /tmp/ps.AppImage --appimage-extract >/dev/null && rm /tmp/ps.AppImage || true
ENV PRUSASLICER=/opt/squashfs-root/AppRun
WORKDIR /spare
COPY pyproject.toml ./
COPY spare ./spare
RUN pip install --no-cache-dir -e ".[dev]"
COPY . .
CMD ["make", "test-fast"]
