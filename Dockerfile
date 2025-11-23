FROM pytorch/pytorch:2.7.1-cuda11.8-cudnn9-runtime

COPY . /app/
WORKDIR /app

# Install Python dependencies
RUN pip install --no-cache-dir \
    pillow \
    torchvision \
    numpy

# Unbuffered to avoid problems with output in Docker logs
ENV PYTHONUNBUFFERED=1

# Uncomment to specify GPU device instead of any available
# ENV CUDA_VISIBLE_DEVICES=0

# Default command
ENTRYPOINT ["python", "demo.py"]
