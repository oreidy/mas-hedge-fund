"""Helper functions for LLM"""

import json
from typing import TypeVar, Type, Optional, Any
from pydantic import BaseModel
from utils.progress import progress
from llm.models import ModelProvider

T = TypeVar('T', bound=BaseModel)

def call_llm(
    prompt: Any,
    model_name: str,
    model_provider: str,
    pydantic_model: Type[T],
    agent_name: Optional[str] = None,
    max_retries: int = 5,
    default_factory = None
) -> T:
    """
    Makes an LLM call with retry logic, handling both Deepseek and non-Deepseek models.
    
    Args:
        prompt: The prompt to send to the LLM
        model_name: Name of the model to use
        model_provider: Provider of the model
        pydantic_model: The Pydantic model class to structure the output
        agent_name: Optional name of the agent for progress updates
        max_retries: Maximum number of retries (default: 5)
        default_factory: Optional factory function to create default response on failure
        
    Returns:
        An instance of the specified Pydantic model
    """
    from llm.models import get_model, get_model_info
    
    model_info = get_model_info(model_name)
    llm = get_model(model_name, model_provider)
    
    # For non-Deepseek models, we can use structured output
    if model_provider not in [ModelProvider.OLLAMA] and not (model_info and model_info.is_deepseek()):
        llm = llm.with_structured_output(
            pydantic_model,
            method="json_mode",
        )
    
    # Call the LLM with retries
    for attempt in range(max_retries):
        try:
            # Call the LLM
            result = llm.invoke(prompt)
            
            # For Deepseek, we need to extract and parse the JSON manually
            if model_info and model_info.is_deepseek():
                parsed_result = extract_json_from_deepseek_response(result.content)
                if parsed_result:
                    return pydantic_model(**parsed_result)
                
            # Ollama: manual JSON extraction (assuming same formatting)
            elif model_provider == ModelProvider.OLLAMA:
                parsed_result = extract_json_from_ollama_response(result.content)
                if parsed_result:
                    return pydantic_model(**parsed_result)
                
            else:
                return result
                
        except Exception as e:
            if agent_name:
                progress.update_status(agent_name, None, f"Error - retry {attempt + 1}/{max_retries}")
            
            # Check if this is a rate limit error for Groq
            error_str = str(e).lower()
            if model_provider == ModelProvider.GROQ and ("rate limit" in error_str or "429" in error_str or "rate_limit_exceeded" in error_str):
                print(f"Rate limit hit during LLM call: {e}")
                print(f"Retrying (attempt {attempt + 1}/{max_retries})")
                # Wait a bit before retrying
                import time
                time.sleep(1 * (2 ** attempt))  # Exponential backoff: 1s, 2s, 4s, 8s, 16s
                continue
            
            if attempt == max_retries - 1:
                print(f"Error in LLM call after {max_retries} attempts: {e}")
                # Use default_factory if provided, otherwise create a basic default
                if default_factory:
                    return default_factory()
                return create_default_response(pydantic_model)

    # This should never be reached due to the retry logic above
    return create_default_response(pydantic_model)

def create_default_response(model_class: Type[T]) -> T:
    """Creates a safe default response based on the model's fields."""
    default_values = {}
    for field_name, field in model_class.model_fields.items():
        if field.annotation == str:
            default_values[field_name] = "Error in analysis, using default"
        elif field.annotation == float:
            default_values[field_name] = 0.0
        elif field.annotation == int:
            default_values[field_name] = 0
        elif hasattr(field.annotation, "__origin__") and field.annotation.__origin__ == dict:
            default_values[field_name] = {}
        else:
            # For other types (like Literal), try to use the first allowed value
            if hasattr(field.annotation, "__args__"):
                default_values[field_name] = field.annotation.__args__[0]
            else:
                default_values[field_name] = None
    
    return model_class(**default_values)

def extract_json_from_deepseek_response(content: str) -> Optional[dict]:
    """Extracts JSON from Deepseek's markdown-formatted response."""
    try:
        json_start = content.find("```json")
        if json_start != -1:
            json_text = content[json_start + 7:]  # Skip past ```json
            json_end = json_text.find("```")
            if json_end != -1:
                json_text = json_text[:json_end].strip()
                return json.loads(json_text)
    except Exception as e:
        print(f"Error extracting JSON from Deepseek response: {e}")
    return None

def extract_json_from_ollama_response(content: str) -> Optional[dict]:
    """Extract JSON from Ollama's response, handling cases where JSON is not formatted inside code blocks."""
    try:
        # First, try loading raw JSON if it's returned as plain text
        return json.loads(content.strip())
    except json.JSONDecodeError:
        pass  # If this fails, proceed to extract from markdown

    try:
        # If JSON is wrapped inside markdown ```json ... ```
        json_start = content.find("```json")
        if json_start != -1:
            json_text = content[json_start + 7:]
            json_end = json_text.find("```")
            if json_end != -1:
                json_text = json_text[:json_end].strip()
            return json.loads(json_text)
    except Exception as e:
        print(f"Error extracting JSON from Ollama response: {e}")

    return None
