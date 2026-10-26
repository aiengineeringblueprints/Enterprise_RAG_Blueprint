import os
import uvicorn
import logging
import asyncio
from fastapi import FastAPI, HTTPException, Header
from pydantic import BaseModel, Field
from typing import Union, List
from prompts.promt_manager import PromptKey
from check_answers_llm import check_answer_with_second_llm, check_answer_with_keywords
from handle_llms import call_llm
from guardrails_manager import ensure_input_allowed


LLM_SOURCE_ANSWER = os.getenv("LLM_SOURCE_ANSWER")
LLM_SOURCE_CHECK = os.getenv("LLM_SOURCE_CHECK")

app = FastAPI()

class HealthResponse(BaseModel):
    status: str
    service: str

class ChatMessage(BaseModel):
    role: str 
    content: str

class QuestionRequest(BaseModel):
    question: str = Field(min_length=1)
    use_mcp: bool = False
    prompt_key: Union[PromptKey, str] = PromptKey.SOURCE
    used_model: str = LLM_SOURCE_ANSWER
    show_sources: bool = True
    user_roles: List[str] = Field(default_factory=lambda: ["General"])
    chat_history: List[ChatMessage] = Field(
        default_factory=list
    ) 

class QuestionCheckRequest(BaseModel):
    question: str = Field(min_length=1)
    answer: str = Field(min_length=1)
    relevant_documents: list[str]
    prompt_key: Union[PromptKey, str] = PromptKey.CHECK
    used_model: str = LLM_SOURCE_CHECK    
    question_prompt_key: Union[PromptKey, str] = PromptKey.SOURCE

class KeywordCheckRequest(BaseModel):
    question: str = Field(min_length=1)
    answer: str = Field(min_length=1)
    expected_keywords: list[str] = []
    threshold: float = Field(default=0.6, ge=0.0, le=1.0)

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    return HealthResponse(status="healthy", service="chain")

@app.post("/call_llm")
async def call_llm_endpoint(request: QuestionRequest, authorization: str | None = Header(default=None)):
    try:
        if request.use_mcp:
            return await call_mcp_endpoint(request, authorization)
        allowed, reason = ensure_input_allowed(request.question)
        if not allowed:
            raise HTTPException(status_code=400, detail=reason or "Input blocked by guardrails.")

        # Convert string to PromptKey if needed
        prompt_key = request.prompt_key
        if isinstance(prompt_key, str):
            try:
                prompt_key = PromptKey(prompt_key)
            except ValueError:
                raise HTTPException(status_code=422, detail=f"Invalid prompt_key: {prompt_key}")

        # Convert ChatMessage Pydantic objects to dicts
        chat_history_dicts = []
        if request.chat_history:
            for msg in request.chat_history:
                if isinstance(msg, ChatMessage):
                    chat_history_dicts.append(
                        {"role": msg.role, "content": msg.content}
                    )
                else:
                    chat_history_dicts.append(msg)

        result, relevant_documents, prompt_key_result = call_llm(
            question=request.question,
            prompt_key=prompt_key,
            used_model=request.used_model if request.used_model else LLM_SOURCE_ANSWER,
            show_sources=request.show_sources,
            user_roles=request.user_roles if request.user_roles else ["General"],
            chat_history=chat_history_dicts,
        )
        return {
            "result": result,
            "relevant_documents": relevant_documents,
            "prompt_key": prompt_key_result.value if hasattr(prompt_key_result, "value") else str(prompt_key_result)
        }
    except HTTPException:
        raise
    except Exception as e:
        logging.exception("/call_llm failed: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


async def call_mcp_endpoint(request: QuestionRequest, authorization: str | None):
    from mcp_integration.config import enabled, load_config
    from mcp_integration.identity import verify_identity
    from mcp_integration.execution import answer_with_mcp, MCPModelError

    if not enabled():
        raise HTTPException(503, "MCP is disabled")
    identity = verify_identity(authorization)
    if len(request.question) > 16000 or len(request.chat_history) > 100:
        raise HTTPException(422, "MCP question or history exceeds the input limit")
    try:
        prompt_key = PromptKey(request.prompt_key)
    except ValueError:
        raise HTTPException(422, "Invalid prompt_key") from None
    if prompt_key not in {PromptKey.SOURCE, PromptKey.SUMMARY, PromptKey.WEEKLY_REPORT}:
        raise HTTPException(422, "This prompt is not supported for MCP chat")
    try:
        config = load_config()
    except Exception:
        logging.getLogger("rag.mcp").error("mcp_configuration_invalid")
        raise HTTPException(503, "MCP configuration is invalid") from None
    try:
        async with asyncio.timeout(config.request_timeout_seconds):
            allowed, reason = await asyncio.to_thread(ensure_input_allowed, request.question, fail_closed=True)
            if not allowed:
                raise HTTPException(400, reason or "Input blocked by guardrails.")
            normalized = request.model_copy(update={
                "prompt_key": prompt_key,
                "used_model": request.used_model or LLM_SOURCE_ANSWER,
                "user_roles": list(identity.roles),
            })
            return await answer_with_mcp(normalized, identity, config)
    except HTTPException:
        raise
    except TimeoutError:
        raise HTTPException(504, "MCP request timed out") from None
    except MCPModelError as exc:
        raise HTTPException(502, str(exc)) from None
    except Exception:
        logging.getLogger("rag.mcp").error("mcp_request_failed")
        raise HTTPException(502, "MCP-enabled RAG request failed") from None
    
@app.post("/check_answer")
async def check_answer_endpoint(request: QuestionCheckRequest):
    try:
        # Convert strings to PromptKey if needed
        prompt_key = request.prompt_key
        if isinstance(prompt_key, str):
            prompt_key = PromptKey(prompt_key)
            
        question_prompt_key = request.question_prompt_key
        if isinstance(question_prompt_key, str):
            question_prompt_key = PromptKey(question_prompt_key)
            
        evaluation = check_answer_with_second_llm(
            question=request.question,
            answer=request.answer,
            relevant_documents=request.relevant_documents,
            question_prompt_key=question_prompt_key,
            prompt=prompt_key,
            used_model=request.used_model if request.used_model else LLM_SOURCE_CHECK,
        )
        return {"evaluation": evaluation}
    except HTTPException:
        raise
    except Exception as e:
        logging.exception("/check_answer failed: %s", e)
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/check_keywords")
async def check_keywords_endpoint(request: KeywordCheckRequest):
    try:
        return check_answer_with_keywords(
            question=request.question,
            answer=request.answer,
            expected_keywords=request.expected_keywords,
            threshold=request.threshold
        )
    except Exception as e:
        logging.exception("/check_keywords failed: %s", e)
        raise HTTPException(status_code=500, detail=str(e))
    

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
