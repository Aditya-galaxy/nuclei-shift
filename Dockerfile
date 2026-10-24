# CPU image for batch nucleus segmentation with Cellpose 3.
FROM python:3.12-slim
RUN apt-get update && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 && rm -rf /var/lib/apt/lists/*
# CPU-only PyTorch keeps the image small; use a CUDA base image and wheel for GPU machines.
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir "cellpose>=3.1,<4" scikit-image scipy
WORKDIR /app
COPY predict.py .
# Download the model weights at build time so containers start without network access.
RUN python -c "from cellpose import models; models.Cellpose(model_type='nuclei'); models.Cellpose(model_type='cyto3')"
ENTRYPOINT ["python", "predict.py"]
CMD ["--input", "/data/in", "--output", "/data/out"]
