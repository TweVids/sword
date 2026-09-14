FROM python:3.10-slim

# Install system dependencies for git, compilation, and testing
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    patch \
    curl \
    build-essential \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Install common testing tools
RUN pip install --no-cache-dir pytest pytest-timeout coverage

# Setup testbed workspace
WORKDIR /testbed
RUN git config --global --add safe.directory /testbed \
    && git config --global user.email "sword-agent@swe.local" \
    && git config --global user.name "Sword SWE Agent"

CMD ["bash"]
