import os
import uvicorn
import logging
import traceback
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Union
from prompts.promt_manager import PromptKey
from check_answers_llm import check_answer_with_second_llm, check_answer_with_keywords
from handle_llms import call_llm

LLM_SOURCE_ANSWER = os.getenv("LLM_SOURCE_ANSWER")
LLM_SOURCE_CHECK = os.getenv("LLM_SOURCE_CHECK")

app = FastAPI()
class QuestionRequest(BaseModel):
    question: str
    prompt_key: Union[PromptKey, str] = PromptKey.SOURCE
    used_model: str = LLM_SOURCE_ANSWER
    show_sources: bool = True
    user_roles: list = ["General"]

class QuestionCheckRequest(BaseModel):
    question: str
    answer: str
    relevant_documents: list[str]
    prompt_key: Union[PromptKey, str] = PromptKey.CHECK
    used_model: str = LLM_SOURCE_CHECK    
    question_prompt_key: Union[PromptKey, str] = PromptKey.SOURCE

class KeywordCheckRequest(BaseModel):
    question: str
    answer: str
    expected_keywords: list[str] = []
    threshold: float = 0.6

@app.post("/call_llm")
async def call_llm_endpoint(request: QuestionRequest):
    try:
        # Convert string to PromptKey if needed
        prompt_key = request.prompt_key
        if isinstance(prompt_key, str):
            prompt_key = PromptKey(prompt_key)
            
        result, relevant_documents, prompt_key_result = call_llm(
            question=request.question,
            prompt_key=prompt_key,
            used_model=request.used_model if request.used_model else LLM_SOURCE_ANSWER,
            show_sources=request.show_sources,
            user_roles=request.user_roles if request.user_roles else ["General"]
        )
        return {
            "result": result,
            "relevant_documents": relevant_documents,
            "prompt_key": prompt_key_result.value if hasattr(prompt_key_result, "value") else str(prompt_key_result)
        }
    except Exception as e:
        logging.exception("/call_llm failed: %s", e)
        raise HTTPException(status_code=500, detail=str(e))
    
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
