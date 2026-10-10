FROM python:3.13-slim
ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=/app
WORKDIR /app
COPY pyproject.toml README.md ./
COPY src ./src
COPY evals ./evals
RUN pip install --no-cache-dir . && useradd -r -u 10001 app && mkdir /app/data && chown app /app/data
USER app
ENTRYPOINT ["booksreview"]
CMD ["--help"]
