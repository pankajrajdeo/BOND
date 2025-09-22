FROM python:3.11-slim

WORKDIR /app

# Install system dependencies (FAISS needs libgomp on Debian slim)
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 && rm -rf /var/lib/apt/lists/*

# Copy entire project, then install (no lockfile available)
COPY . .
RUN pip install --no-cache-dir --upgrade pip && pip install --no-cache-dir .

EXPOSE 8000

# The command to run the API server
CMD ["bond-serve"]
