"""
API Routes for Obligation Tracking System.
Implements RESTful endpoints for file upload and chat.
"""

import asyncio
import json
import logging
import uuid
import yaml
from typing import List, Dict, Any
from datetime import datetime, timezone
from fastapi import APIRouter, HTTPException, Depends, Form, Request

from src.api.models import (
    HealthResponse,
    ChatResponse,
    UserResponse,
    UploadResponse
)
from src.processing.document_processor import DocumentProcessor
from src.utils.s3_utility import get_s3_client, S3_BUCKET, get_s3_file
from src.llm.litellm_client import LiteLLMClient
from src.storage.milvus_client import MilvusClient
from src.utils.report_generator import ReportSynthesizer
from src.utils.config import get_model_config
from src.utils.obs import LLMUsageTracker
from src.processing.obligation_extractor import ObligationExtractor
from src.utils.kafka import create_event_logger

logger = logging.getLogger(__name__)
router = APIRouter()

# Initialize components (singleton pattern)
_obligation_extractor = None


def get_obligation_extractor() -> ObligationExtractor:
    global _obligation_extractor
    if _obligation_extractor is None:
        _obligation_extractor = ObligationExtractor(get_llm_client())
    return _obligation_extractor
_llm_client = None
_milvus_client = None
_doc_processor = None


def get_llm_client() -> LiteLLMClient:
    global _llm_client
    if _llm_client is None:
        _llm_client = LiteLLMClient()
    return _llm_client


def get_milvus_client() -> MilvusClient:
    global _milvus_client
    if _milvus_client is None:
        _milvus_client = MilvusClient()
    return _milvus_client


def get_doc_processor() -> DocumentProcessor:
    global _doc_processor
    if _doc_processor is None:
        _doc_processor = DocumentProcessor()
    return _doc_processor


def get_app_version() -> str:
    with open("config/model_config.yaml", 'r') as f:
        config = yaml.safe_load(f)
    return config.get('app', {}).get('version', '1.0.0')


async def get_llm_config(user_metadata):
    """Get dynamic LLM configuration based on user's team."""
    try:
        async with get_model_config() as config:
            user_metadata = json.loads(user_metadata) if user_metadata else {}
            team_id = user_metadata.get("team_id")
            team_config = await config.get_team_model_config(team_id)
            model = team_config["selected_model"]
            provider = team_config["provider"]
            provider_model = f"{provider}/{model}"
            model_config = team_config["config"]
            llm_params = {
                "model": provider_model,
                **model_config
            }
            return llm_params
    except Exception as e:
        logging.error(f"Failed to get LLM config: {str(e)}")
        raise ValueError(f"Failed to get model configuration: {str(e)}")


@router.get("/health", response_model=HealthResponse, tags=["System"])
async def health_check():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy",
        version=get_app_version(),
        timestamp=datetime.now(timezone.utc).isoformat()
    )

@router.post("/upload", response_model=UploadResponse, tags=["Upload"])
async def upload_file(
    request: Request,
    s3_url: str = Form(...),
    query: str = Form(None),
    # current_user: User = Depends(get_current_user)  # Auth disabled for testing
    user_metadata: str = Form(...)
):
    """
    Process a file from S3 URL, extract text, store in Milvus, and return PDF report URL.
    """
    # Initialize Kafka event logger
    event_logger = create_event_logger()
    auth_token = request.headers.get("Authorization", "N/A")
    
    try:
        logger.info(f"Received S3 URL: {s3_url}")

        # Get dynamic LLM config
        llm_params = await get_llm_config(user_metadata)
        token_tracker = LLMUsageTracker(request, llm_params["model"])

        doc_processor = get_doc_processor()
        milvus_client = get_milvus_client()
        llm_client = get_llm_client()

        # Get file bytes from S3
        content = get_s3_file(s3_url)
        logger.info(f"File size: {len(content)} bytes")

        # Extract filename from S3 URL
        filename = s3_url.split('/')[-1].split('?')[0]
        
        # Log: Document processing started
        event_logger.log_event(f"📄 Processing contract document: {filename}", auth_token)

        # Validate file
        is_valid, error = doc_processor.validate_file(filename, len(content))
        if not is_valid:
            raise HTTPException(status_code=400, detail=error)

        # Extract text from file bytes
        text = doc_processor.extract_text(content, filename)
        logger.info(f"Extracted text length: {len(text)} characters")

        # Chunk the document
        chunk_data_list = doc_processor.chunk_text(text)
        logger.info(f"Document chunked into {len(chunk_data_list)} chunks")

        # Generate human-readable contract ID (doc-YYYY-XXX)
        year = datetime.now().year
        random_num = str(uuid.uuid4().int)[:3].zfill(3)
        contract_id = f"doc-{year}-{random_num}"

        # Store extracted data chunks in extract_data collection (BATCH + ASYNC)
        # Step 1: Generate embeddings in parallel
        chunk_texts = [chunk_info.get("text", "")[:1000] for chunk_info in chunk_data_list]
        chunk_embeddings = await asyncio.to_thread(
            llm_client.get_embeddings_batch, chunk_texts
        )

        # Step 2: Prepare batch records
        chunk_records = []
        for idx, chunk_info in enumerate(chunk_data_list):
            chunk_records.append({
                "id": f"{contract_id}_chunk_{chunk_info.get('index', idx)}",
                "user_id": "test_user",  # Auth disabled for testing
                "contract_id": contract_id,
                "chunk_index": chunk_info.get("index", idx),
                "content": chunk_info.get("text", "")[:7900],
                "embedding": chunk_embeddings[idx] if idx < len(chunk_embeddings) else [0.0] * 768
            })

        # Step 3: Batch insert (non-blocking)
        await asyncio.to_thread(milvus_client.insert_extract_data_batch, chunk_records)
        logger.info(f"Batch stored {len(chunk_records)} chunks in extract_data collection")

        # Store contract metadata (async)
        milvus_client.create_contracts_collection()
        contract_embedding = await asyncio.to_thread(
            llm_client.get_single_embedding, text[:1000]
        )
        contract_data = {
            "id": contract_id,
            "user_id": "test_user",  # Auth disabled for testing
            "filename": filename,
            "title": filename,
            "parties": "",
            "effective_date": "",
            "expiration_date": "",
            "contract_value": "",
            "content_summary": text[:2000],
            "embedding": contract_embedding
        }
        await asyncio.to_thread(milvus_client.insert_contract, contract_data)
        logger.info(f"Contract stored with ID: {contract_id}")

        # Log: Extracting obligations
        event_logger.log_event("🔍 Analyzing contract for obligations and deadlines...", auth_token)

        # Extract obligations using LLM (async)
        obligation_extractor = get_obligation_extractor()
        obligations = await asyncio.to_thread(
            obligation_extractor.extract_obligations,
            text, contract_id, token_tracker, llm_params
        )

        # Store extracted obligations in Milvus (BATCH + ASYNC)
        if obligations:
            for obl in obligations:
                obl["user_id"] = "test_user"  # Auth disabled for testing

            # Generate embeddings for all obligations in batch
            obl_texts = [
                f"{obl.get('type', '')} {obl.get('description', '')} {obl.get('due_date', '')}"
                for obl in obligations
            ]
            obl_embeddings = await asyncio.to_thread(
                llm_client.get_embeddings_batch, obl_texts
            )

            # Batch insert obligations
            await asyncio.to_thread(
                milvus_client.insert_obligations_batch, obligations, obl_embeddings
            )
        logger.info(f"Extracted and batch stored {len(obligations)} obligations for contract {contract_id}")

        # Log: Obligations found
        if obligations:
            event_logger.log_event(f"✅ Found {len(obligations)} obligations and deadlines in the contract", auth_token)
        else:
            event_logger.log_event("ℹ️ No specific obligations found in the contract", auth_token)

        # Build context from extracted chunks
        context = "\n\n".join([chunk_info.get("text", "") for chunk_info in chunk_data_list[:5]])

        if query:
            # If query provided, search extract_data and generate answer
            query_embedding = llm_client.get_single_embedding(query)
            search_results = milvus_client.search_extract_data(
                query_embedding, top_k=5, user_id="test_user"  # Auth disabled for testing
            )
            search_context = "\n\n".join([r.get("content", "") for r in search_results])

            prompt = f"""You are a contract analysis assistant. Your task is to answer the user's question based ONLY on the provided document content.

## DOCUMENT CONTENT:
{search_context}

## USER QUESTION:
{query}

## INSTRUCTIONS:
1. **Answer based on evidence**: Only use information explicitly stated in the document
2. **Quote relevant sections**: Include direct quotes to support your answer
3. **Be specific**: Provide exact dates, amounts, parties, and terms when available
4. **Acknowledge limitations**: If the document doesn't contain enough information, clearly state what's missing
5. **Structure your response**: Use bullet points or numbered lists for clarity
6. **Highlight key terms**: Emphasize important obligations, deadlines, and parties

## RESPONSE FORMAT:
- Start with a direct answer to the question
- Provide supporting evidence from the document
- Include relevant quotes in "quotation marks"
- If information is not in the document, say: "This information is not explicitly stated in the provided document"

## YOUR ANSWER:"""
            answer = llm_client.generate(prompt, token_tracker=token_tracker, llm_params=llm_params)

            document_context = f"""## File: {filename}

## Document Content
{context[:3000]}

## Query
{query}

## Answer
{answer}
"""
            response_message = answer
        else:
            document_context = f"""## File: {filename}

## Document Content
{context[:4000]}

## Summary
File uploaded and processed successfully. {len(chunk_data_list)} chunks stored for semantic search.
"""
            response_message = "File processed successfully. Check the PDF report for detailed summary."

        synthesizer = ReportSynthesizer(llm_params=llm_params, token_tracker=token_tracker)
        report_data = await synthesizer.synthesize_report(document_context, format="pdf")

        # Upload PDF to S3
        pdf_filename = f"report_{contract_id[:8]}.pdf"
        s3_key = f"reports/test_user/{pdf_filename}"  # Auth disabled for testing

        s3_client = get_s3_client()
        s3_client.put_object(
            Bucket=S3_BUCKET,
            Key=s3_key,
            Body=report_data['content'],
            ContentType='application/pdf'
        )

        pdf_url = s3_client.generate_presigned_url(
            ClientMethod="get_object",
            Params={"Bucket": S3_BUCKET, "Key": s3_key},
            ExpiresIn=3600
        )

        logger.info(f"PDF uploaded to S3: {s3_key}")

        # Log: Report ready
        event_logger.log_event("📋 Obligation tracking report generated successfully", auth_token)

        return UploadResponse(
            message=response_message,
            file_id=contract_id,
            filename=filename,
            pdf_url=pdf_url,
            obligations=obligations
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error uploading file: {e}", exc_info=True)
        # Log: Processing failed
        event_logger.log_event("❌ Failed to process contract document", auth_token)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/chat", response_model=ChatResponse, tags=["Chat"])
async def chat(
    request: Request,
    message: str = Form(...),
    user_metadata: str = Form(...),
    document_id: str = Form(None)
):
    """Chat with stored documents using semantic search on extract_data collection."""
    # Initialize Kafka event logger
    event_logger = create_event_logger()
    auth_token = request.headers.get("Authorization", "N/A")
    
    try:
        # Log: Query received
        event_logger.log_event("🔍 Searching contract obligations...", auth_token)
        
        # Get dynamic LLM config
        llm_params = await get_llm_config(user_metadata)
        token_tracker = LLMUsageTracker(request, llm_params["model"])

        llm_client = get_llm_client()
        milvus_client = get_milvus_client()

        logger.info(f"Chat request: {message[:100]}... document_id: {document_id}")

        # Get embedding for user query
        query_embedding = llm_client.get_single_embedding(message)

        # Search for relevant chunks from extract_data collection
        results = milvus_client.search_extract_data(query_embedding, top_k=5, user_id="test_user", document_id=document_id)  # Auth disabled for testing

        # Build context from chunks
        context = "\n\n".join([r.get("content", "") for r in results])

        if not context.strip():
            context = "No relevant documents found."

        # Generate response using LLM
        prompt = f"""You are an intelligent contract assistant helping users understand their documents. Answer the user's question using the provided document content.

## DOCUMENT CONTEXT:
{context}

## USER QUESTION:
{message}

## RESPONSE GUIDELINES:

### If the answer IS in the document:
1. Provide a clear, direct answer
2. Quote relevant sections to support your response
3. Explain any complex legal or technical terms
4. Highlight important dates, amounts, or parties
5. Use bullet points for multiple items
6. Be concise but comprehensive

### If the answer is NOT in the document:
1. Clearly state: "I cannot find this information in the provided documents"
2. Suggest what type of document might contain this information
3. Offer to help with related questions that CAN be answered

### If the question is ambiguous:
1. Ask for clarification
2. Provide possible interpretations
3. Answer each interpretation if possible

### Always:
- Use professional but friendly language
- Avoid legal advice (you're providing information, not counsel)
- Be accurate - don't make assumptions
- Format responses for readability

## YOUR RESPONSE:"""

        response_text = llm_client.generate(prompt, token_tracker=token_tracker, llm_params=llm_params)

        # Log: Response ready
        event_logger.log_event("✅ Answer ready", auth_token)

        return ChatResponse(message=response_text)

    except Exception as e:
        logger.error(f"Error in chat: {e}", exc_info=True)
        # Log: Chat failed
        event_logger.log_event("❌ Unable to answer your question", auth_token)
        raise HTTPException(status_code=500, detail=str(e))
