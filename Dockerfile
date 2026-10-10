FROM python:3.12-slim
WORKDIR /app
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu
COPY pyproject.toml README.md ./
COPY src ./src
COPY tests ./tests
COPY docs ./docs
RUN pip install --no-cache-dir -e ".[dev]"
# Default: the calibration tests. Benchmark: docker run --rm -v "$PWD/results:/app/results" IMAGE python -m src.calibration.run --seeds 3
CMD ["python", "-m", "pytest", "tests/test_calibration.py", "-q"]
