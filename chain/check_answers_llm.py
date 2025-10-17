import os
from load_chain import load_chain, rag_chain
from handle_llms import load_llm_model
from prompts.promt_manager import PromptManager, PromptKey
from langchain_core.prompts import ChatPromptTemplate
import re
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))


LLM_SOURCE_CHECK = os.getenv("LLM_SOURCE_CHECK")


def check_answer_with_second_llm(question: str, question_prompt_key, answer: str, relevant_documents, prompt: PromptKey = PromptKey.CHECK, used_model: str = LLM_SOURCE_CHECK) -> str:
    """
    Evaluates an answer to a given question using a second LLM (Language Model) 
    and a specified prompt template. The function utilizes a retriever, a chain, 
    and a model to perform the evaluation.
    
    Args:
        question (str): The question to be evaluated.
        question_promt: The original prompt template used for the question.
        answer (str): The answer to be evaluated.
        relevant_documents: The documents used for evaluation.
        promt (PromptKey, optional): The prompt template to be used. Defaults to PromptKey.CHECK.
        used_model (str, optional): The LLM model to be used for evaluation. Defaults to LLM_SOURCE_CHECK.
        user_roles (list, optional): List of roles/categories the user has for role-based filtering.
        
    Returns:
        str: The evaluation result of the question and answer using the specified LLM.
        
    Raises:
        ValueError: If any of the required components (retriever, prompt, model, chain, or evaluation result) 
                    are not found or are invalid.
    """
    retriever = answer # answer from the first llm. This is "context" in the prompt


    raw_template = PromptManager().load_template(question_prompt_key)
    rag_prompt = ChatPromptTemplate.from_template(raw_template)  
    if "question" in rag_prompt.input_variables:
        input_question_with_context = rag_prompt.format_messages(
        context=relevant_documents,
        question=question
        )
    elif "question" not in rag_prompt.input_variables:
        input_question_with_context = rag_prompt.format_messages(
        context=relevant_documents
        )

    model = load_llm_model(used_model)

    chain = load_chain(
        retriever, 
        prompt, 
        model, 
        include_doc_names=True
    )

    evaluation, _ = rag_chain(
        question=input_question_with_context, 
        chain=chain, 
        retriever=retriever, 
        show_sources=False
    )

    return evaluation  
 


# TODO: use an LLM for keyword extraction instead?
def extract_keywords_from_text(text: str, min_length: int = 4) -> list:
    """
    Extracts potential keywords from the provided text.
    
    All words without punctuation are extracted, and common
    stopwords (e.g., "and", "or", "the", "a", ...) as well as
    words shorter than min_length are filtered out.
    
    Returns:
        A list of unique keywords.
    """
    # Einfache Liste deutscher und allgemeiner Stopwords
    stopwords = {
    "aber", "alle", "allem", "allen", "aller", "alles", "als", "also", "am", "an",
    "ander", "andere", "anderem", "anderen", "anderer", "anderes", "auch", "auf", "aus",
    "bei", "bin", "bis", "bist", "da", "dabei", "deshalb", "dass", "dein", "deine", "dem",
    "den", "der", "des", "dessen", "die", "dies", "diese", "diesem", "diesen", "dieser",
    "dieses", "doch", "dort", "du", "durch", "ein", "eine", "einem", "einen", "einer",
    "eines", "er", "es", "etwas", "euer", "eure", "für", "gegen", "gewesen", "hab", "habe",
    "haben", "hat", "hatte", "hatten", "hier", "hin", "hinter", "ich", "ihm", "ihn", "ihr",
    "ihre", "im", "in", "ist", "ja", "jede", "jedem", "jeden", "jeder", "jedes", "jemals",
    "jetzt", "kann", "kein", "keine", "keinem", "keinen", "keiner", "keines", "mit",
    "muss", "musste", "nach", "nicht", "nichts", "noch", "nun", "nur", "ob", "oder",
    "ohne", "sehr", "sein", "seine", "sich", "sie", "sind", "so", "solche", "solchem",
    "solchen", "solcher", "solches", "soll", "sollten", "sondern", "sonst", "über", "um",
    "und", "uns", "unse", "unser", "unsere", "unter", "vom", "von", "vor", "was", "weil",
    "weiter", "welche", "welchem", "welchen", "welcher", "welches", "wenn", "wer", "werden",
    "wie", "wieder", "will", "wir", "wird", "wo", "wollen", "wollte", "würde", "würden",
    "zu", "zum", "zur", "zwar", "zwischen",

    "a", "about", "above", "after", "again", "against", "all", "am", "an", "and", "any",
    "are", "aren't", "as", "at", "be", "because", "been", "before", "being", "below",
    "between", "both", "but", "by", "can't", "cannot", "could", "couldn't", "did",
    "didn't", "do", "does", "doesn't", "doing", "don't", "down", "during", "each", "few",
    "for", "from", "further", "had", "hadn't", "has", "hasn't", "have", "haven't",
    "having", "he", "he'd", "he'll", "he's", "her", "here", "here's", "hers", "herself",
    "him", "himself", "his", "how", "how's", "i", "i'd", "i'll", "i'm", "i've", "if",
    "in", "into", "is", "isn't", "it", "it's", "its", "itself", "let's", "me", "more",
    "most", "mustn't", "my", "myself", "no", "nor", "not", "of", "off", "on", "once",
    "only", "or", "other", "ought", "our", "ours", "ourselves", "out", "over", "own",
    "same", "shan't", "she", "she'd", "she'll", "she's", "should", "shouldn't", "so",
    "some", "such", "than", "that", "that's", "the", "their", "theirs", "them",
    "themselves", "then", "there", "there's", "these", "they", "they'd", "they'll",
    "they're", "they've", "this", "those", "through", "to", "too", "under", "until",
    "up", "very", "was", "wasn't", "we", "we'd", "we'll", "we're", "we've", "were",
    "weren't", "what", "what's", "when", "when's", "where", "where's", "which", "while",
    "who", "who's", "whom", "why", "why's", "with", "won't", "would", "wouldn't", "you",
    "you'd", "you'll", "you're", "you've", "your", "yours", "yourself", "yourselves"
    }

    # Removes punctuation and splits the text into words
    words = re.findall(r'\b\w+\b', text.lower())
    # Filters out stopwords and short words
    keywords = [word for word in words if word not in stopwords and len(word) >= min_length]
    return list(set(keywords))


def check_answer_with_keywords(question: str, answer: str, expected_keywords: list[str] = None, threshold: float = 1.0) -> dict:
    """
    Checks whether the LLM's answer contains the expected keywords.
    
    Args:
       question (str): The original question.
       answer (str): The answer generated by the LLM.
       expected_keywords (list[str], optional): A list of keywords that must appear in the answer.
                                                If not specified, potential keywords are extracted from the question.
       threshold (float, optional): The ratio (between 0 and 1) of keywords that must be present in the answer
                                    to evaluate the answer as correct. The default value 1.0 requires all keywords to be present.
                                    
    Returns:
       dict: A dictionary with the following information:
           - evaluation: "yes" if the proportion of found keywords >= threshold,
                         "partial" if only some keywords were found,
                         "no" if no keywords were found.
           - present_keywords: List of keywords found in the answer.
           - missing_keywords: List of expected keywords missing from the answer.
           - total_keywords: Total number of expected keywords.
           - ratio: Ratio of found keywords to the total number.
    """
    answer_lower = answer.lower()
    if expected_keywords is None:
        expected_keywords = extract_keywords_from_text(question)
    
    present_keywords = []
    missing_keywords = []
    for keyword in expected_keywords:
        if keyword.lower() in answer_lower:
            present_keywords.append(keyword)
        else:
            missing_keywords.append(keyword)
            
    total = len(expected_keywords)
    ratio = len(present_keywords) / total if total > 0 else 1.0
    if total == 0:
        evaluation = "undetermined"
    elif ratio >= threshold:
        evaluation = "yes"
    elif ratio > 0:
        evaluation = "partial"
    else:
        evaluation = "no"
    
    result = {
        "evaluation": evaluation,
        "present_keywords": present_keywords,
        "missing_keywords": missing_keywords,
        "total_keywords": total,
        "ratio": ratio
    }
    return result

