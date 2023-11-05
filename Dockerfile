FROM python:3.8-slim-buster

# Set environment variables to make the container run in a non-interactive mode
ENV PYTHONUNBUFFERED 1
ENV FLASK_ENV=production
ENV FLASK_APP=app.py
ENV FLASK_RUN_HOST=0.0.0.0

# Set the working directory in the container
WORKDIR /app

# Install system dependencies
RUN apt-get update \
    && apt-get install -y --no-install-recommends gcc \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

# Install any needed packages specified in requirements.txt
RUN pip install --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

COPY . .

# Make sure the app runs on port 8080
EXPOSE 8080

# Run the application
CMD ["gunicorn", "--bind", "0.0.0.0:8080", "app:app"]
