"""
Obligation Tracking and Alerts API
Main application entry point.

An intelligent document processing agent that extracts all future dates
and obligations from contracts including renewals, payment schedules,
service delivery deadlines, and compliance milestones.
"""

import yaml
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.api.routes import router
from src.utils.logger import setup_logging

# Load environment variables
load_dotenv(override=True)

# Setup logging
logger = setup_logging()

# Load API configuration
with open("config/api_config.yaml", 'r') as f:
    api_config = yaml.safe_load(f)

# Create FastAPI application
app = FastAPI(
    title=api_config['api']['title'],
    description=api_config['api']['description'],
    version=api_config['api']['version'],
    docs_url="/docs",
    redoc_url="/redoc"
)

# Configure CORS
cors_config = api_config.get('cors', {})
app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_config.get('allow_origins', ["*"]),
    allow_credentials=True,
    allow_methods=cors_config.get('allow_methods', ["*"]),
    allow_headers=cors_config.get('allow_headers', ["*"]),
)

# Include API routes
app.include_router(router, prefix=api_config['api']['prefix'])


@app.on_event("startup")
async def startup_event():
    """Initialize services on startup."""
    logger.info("Starting Obligation Tracking API...")
    logger.info(f"API Version: {api_config['api']['version']}")
    port = api_config.get('server', {}).get('port', 8000)
    logger.info(f"API available at: http://localhost:{port}")
    logger.info(f"Docs available at: http://localhost:{port}/docs")


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown."""
    logger.info("Shutting down Obligation Tracking API...")


@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "name": api_config['api']['title'],
        "version": api_config['api']['version'],
        "description": api_config['api']['description'],
        "docs": "/docs",
        "health": f"{api_config['api']['prefix']}/health"
    }

