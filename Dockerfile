# Standard Python image
FROM python:3.11-slim

WORKDIR /app

COPY . /app

# Install dependencies
# COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy only the app logic
# COPY app.py .

# Expose port 8000
EXPOSE 8000

# Run using Uvicorn
# CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
CMD ["python","main.py"]