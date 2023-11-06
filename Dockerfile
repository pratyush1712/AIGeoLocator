# Base image
FROM python:3.8-slim-buster

# Set working directory
WORKDIR /app
ARG FLASK_ENV

# Copy requirements and install Python dependencies
RUN pip install --upgrade pip
COPY requirements.txt .
RUN if [[ "$FLASK_ENV" = "development" ]]; then \
        pip install -r requirements.txt; \
    else \
        pip install --no-cache-dir -r requirements.txt; \
    fi

# Copy app files
COPY . .

# Environment variables for Flask
ENV FLASK_APP=app.py
ENV FLASK_RUN_HOST=0.0.0.0

EXPOSE 8080

CMD ["gunicorn", "--bind", "0.0.0.0:8080", "app:app"]