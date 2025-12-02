"""
Pydantic Models for Obligation Tracking API.
Defines request and response schemas.
"""

from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field


class UserResponse(BaseModel):
    """Response model for user."""
    user_id: str
    email: str
    full_name: Optional[str] = None


class HealthResponse(BaseModel):
    """Health check response model."""
    status: str = Field("healthy", description="Service status")
    version: str = Field(..., description="API version")
    timestamp: str = Field(..., description="Current timestamp")


class ChatRequest(BaseModel):
    """Request model for chat."""
    message: str = Field(..., description="User message")
    user_metadata: str = Field(..., description="JSON string with team_id")
    document_id: Optional[str] = Field(None, description="Optional document ID to search within specific document")


class ChatResponse(BaseModel):
    """Response model for chat."""
    message: str = Field(..., description="AI response")


class UploadRequest(BaseModel):
    """Request model for file upload (for documentation)."""
    s3_url: str = Field(..., description="S3 presigned URL of the document")
    query: Optional[str] = Field(None, description="Optional question about the document")
    user_metadata: str = Field(..., description="JSON string with team_id")


class UploadResponse(BaseModel):
    """Response model for file upload."""
    message: str = Field(..., description="Upload status message")
    file_id: str = Field(..., description="Unique file ID")
    filename: str = Field(..., description="Original filename")
    pdf_url: str = Field(..., description="S3 URL for the generated PDF report")
    obligations: list = Field(default=[], description="Extracted obligations with dates")
