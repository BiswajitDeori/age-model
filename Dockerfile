FROM python:3.10-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libglib2.0-0 \
    libgthread-2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install -r requirements.txt

# Copy the application code
COPY . /app

# Set the working directory
WORKDIR /app

# Expose port 5000 for Flask
EXPOSE 5000

# Run the Flask application
CMD ["flask", "run"]
