# Use an official image as a parent image
FROM ubuntu:20.04

# Set environment variables to avoid interactive prompts during installation
ENV DEBIAN_FRONTEND=noninteractive

# Install necessary dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    libopencv-dev \
    wget \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install ONNX Runtime
RUN wget https://github.com/microsoft/onnxruntime/releases/download/v1.10.0/onnxruntime-linux-x64-1.10.0.tgz && \
    tar -xzf onnxruntime-linux-x64-1.10.0.tgz && \
    rm onnxruntime-linux-x64-1.10.0.tgz

# Set ONNX Runtime library path
ENV LD_LIBRARY_PATH="/onnxruntime-linux-x64-1.10.0/lib:$LD_LIBRARY_PATH"

# Create a directory for your application
WORKDIR /app

# Copy your source code into the container
COPY . .

# Build your C++ application
RUN mkdir build && cd build && \
    cmake .. && \
    make

# # Run the application
# CMD ["./build/object_detection", "model.onnx", "test_image.jpg"]
