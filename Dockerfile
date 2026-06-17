# Use python-slim as base image to keep size relatively small
FROM python:3.10-slim

# Install system dependencies needed for some Python packages (e.g., pandas, matplotlib)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements file first to leverage Docker cache
COPY requirements.txt .

# Install dependencies (CPU-only version of PyTorch to save space and resources)
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
COPY . .

# Expose the port Dash runs on
EXPOSE 8050

# Default command to run the dashboard
CMD ["python", "main.py", "--phase", "5", "--dashboard"]
