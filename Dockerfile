FROM ubuntu:24.04

# Install system dependencies and Python
RUN apt update && \
    apt install -y python3.12 python3.12-venv python3-pip curl && \
    rm -rf /var/lib/apt/lists/*  # Clean up apt cache

# Set working directory inside the container
WORKDIR /home/appuser/app

# Copy project files
COPY . .

# Create a non-root user
RUN useradd --create-home appuser

# Reset file permissions
RUN chown -R appuser:appuser /home/appuser/app

# Switch to non-root user
USER appuser

# Create virtual environment
RUN python3 -m venv /home/appuser/app/venv

# Upgrade pip and install Python dependencies in virtual environment
RUN /home/appuser/app/venv/bin/python -m pip install --upgrade pip setuptools wheel && \
    /home/appuser/app/venv/bin/python -m pip install --no-cache-dir -r requirements.txt

# Ensure the venv is recognized
ENV PATH="/home/appuser/app/venv/bin:$PATH"

# Expose port 8000 for FastAPI
EXPOSE 8000

# Run the app exposing port and binding correctly to allow external connections
CMD ["python3", "-m", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
