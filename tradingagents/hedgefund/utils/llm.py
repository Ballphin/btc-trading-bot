"""Helper functions for LLM"""

import json
import os
import threading
import time
from pydantic import BaseModel
from tradingagents.hedgefund.llm.models import get_model, get_model_info
from tradingagents.hedgefund.utils.progress import progress
from tradingagents.hedgefund.graph.state import AgentState


# Process-wide throttle for the NVIDIA DeepSeek route. NVIDIA's free tier
# rate-limits aggressively (HTTP 429) when 13 analysts fire concurrently;
# serializing requests with a minimum interval trades latency for stability.
_nvidia_throttle_lock = threading.Lock()
_nvidia_last_call_ts: float = 0.0


def _throttle_nvidia() -> None:
    """Block until at least NVIDIA_MIN_INTERVAL_S has passed since last call."""
    global _nvidia_last_call_ts
    try:
        min_interval = float(os.getenv("NVIDIA_MIN_INTERVAL_S", "6"))
    except ValueError:
        min_interval = 6.0
    if min_interval <= 0:
        return
    with _nvidia_throttle_lock:
        now = time.monotonic()
        wait = min_interval - (now - _nvidia_last_call_ts)
        if wait > 0:
            time.sleep(wait)
        _nvidia_last_call_ts = time.monotonic()


def _build_error_message(err: Exception) -> str:
    msg = str(err or "unknown error").strip().replace("\n", " ")
    msg = " ".join(msg.split())
    if len(msg) > 300:
        msg = f"{msg[:297]}..."
    return f"{type(err).__name__}: {msg}" if msg else type(err).__name__


def _default_with_error(
    pydantic_model: type[BaseModel],
    default_factory,
    err: Exception,
) -> BaseModel:
    base = default_factory() if default_factory else create_default_response(pydantic_model)
    err_text = _build_error_message(err)
    if isinstance(base, BaseModel):
        payload = base.model_dump()
        reasoning = payload.get("reasoning")
        if isinstance(reasoning, str):
            payload["reasoning"] = f"{reasoning} | error: {err_text}"
            return pydantic_model(**payload)
    return base


def call_llm(
    prompt: any,
    pydantic_model: type[BaseModel],
    agent_name: str | None = None,
    state: AgentState | None = None,
    max_retries: int = 3,
    default_factory=None,
) -> BaseModel:
    """
    Makes an LLM call with retry logic, handling both JSON supported and non-JSON supported models.

    Args:
        prompt: The prompt to send to the LLM
        pydantic_model: The Pydantic model class to structure the output
        agent_name: Optional name of the agent for progress updates and model config extraction
        state: Optional state object to extract agent-specific model configuration
        max_retries: Maximum number of retries (default: 3)
        default_factory: Optional factory function to create default response on failure

    Returns:
        An instance of the specified Pydantic model
    """
    
    # Extract model configuration if state is provided and agent_name is available
    if state and agent_name:
        model_name, model_provider = get_agent_model_config(state, agent_name)
    else:
        # Use system defaults when no state or agent_name is provided
        model_name = "gpt-4.1"
        model_provider = "OPENAI"

    # Extract API keys from state if available
    api_keys = None
    use_nvidia_deepseek = False
    if state:
        meta = state.get("metadata", {}) or {}
        request = meta.get("request")
        if request and hasattr(request, 'api_keys'):
            api_keys = request.api_keys
        use_nvidia_deepseek = bool(meta.get("use_nvidia_deepseek", False))

    try:
        model_info = get_model_info(model_name, model_provider)
        llm = get_model(model_name, model_provider, api_keys, use_nvidia=use_nvidia_deepseek)

        # For non-JSON support models, we can use structured output
        if not (model_info and not model_info.has_json_mode()):
            llm = llm.with_structured_output(
                pydantic_model,
                method="json_mode",
            )
    except Exception as e:
        if agent_name:
            progress.update_status(agent_name, None, "Error - model initialization failed")
        print(f"Error initializing model for LLM call: {e}")
        return _default_with_error(pydantic_model, default_factory, e)

    # Call the LLM with retries
    last_error = None
    for attempt in range(max_retries):
        try:
            # NVIDIA free tier rate-limits aggressively; serialize calls.
            if use_nvidia_deepseek and str(model_provider).lower().endswith("deepseek"):
                _throttle_nvidia()
            # Call the LLM
            result = llm.invoke(prompt)

            # For non-JSON support models, we need to extract and parse the JSON manually
            if model_info and not model_info.has_json_mode():
                parsed_result = extract_json_from_response(result.content)
                if parsed_result:
                    return pydantic_model(**parsed_result)
                if attempt == max_retries - 1:
                    raise ValueError("Model response did not contain parseable JSON")
            else:
                return result

        except Exception as e:
            last_error = e
            if agent_name:
                progress.update_status(agent_name, None, f"Error - retry {attempt + 1}/{max_retries}")

            if attempt == max_retries - 1:
                print(f"Error in LLM call after {max_retries} attempts: {e}")
                return _default_with_error(pydantic_model, default_factory, e)

    # Fallback guard in case of an unexpected loop exit
    if isinstance(last_error, Exception):
        return _default_with_error(pydantic_model, default_factory, last_error)
    return _default_with_error(
        pydantic_model,
        default_factory,
        RuntimeError("LLM call failed for unknown reason"),
    )


def create_default_response(model_class: type[BaseModel]) -> BaseModel:
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


def extract_json_from_response(content: str) -> dict | None:
    """Extracts JSON from a response, handling markdown-wrapped and raw JSON formats."""
    try:
        # 1. Try markdown code block with ```json
        json_start = content.find("```json")
        if json_start != -1:
            json_text = content[json_start + 7:]  # Skip past ```json
            json_end = json_text.find("```")
            if json_end != -1:
                json_text = json_text[:json_end].strip()
                try:
                    return json.loads(json_text)
                except json.JSONDecodeError:
                    pass

        # 2. Try markdown code block without json specifier
        json_start = content.find("```")
        if json_start != -1:
            json_text = content[json_start + 3:]
            json_end = json_text.find("```")
            if json_end != -1:
                json_text = json_text[:json_end].strip()
                try:
                    return json.loads(json_text)
                except json.JSONDecodeError:
                    pass

        # 3. Try to parse the entire content as JSON
        try:
            return json.loads(content.strip())
        except json.JSONDecodeError:
            pass

        # 4. Find the first top-level JSON object by matching braces
        brace_start = content.find("{")
        if brace_start != -1:
            depth = 0
            for i, char in enumerate(content[brace_start:], brace_start):
                if char == "{":
                    depth += 1
                elif char == "}":
                    depth -= 1
                    if depth == 0:
                        try:
                            return json.loads(content[brace_start:i + 1])
                        except json.JSONDecodeError:
                            break

    except Exception as e:
        print(f"Error extracting JSON from response: {e}")
    return None


def get_agent_model_config(state, agent_name):
    """
    Get model configuration for a specific agent from the state.
    Falls back to global model configuration if agent-specific config is not available.
    Always returns valid model_name and model_provider values.
    """
    request = state.get("metadata", {}).get("request")
    
    if request and hasattr(request, 'get_agent_model_config'):
        # Get agent-specific model configuration
        model_name, model_provider = request.get_agent_model_config(agent_name)
        # Ensure we have valid values
        if model_name and model_provider:
            return model_name, model_provider.value if hasattr(model_provider, 'value') else str(model_provider)
    
    # Fall back to global configuration (system defaults)
    model_name = state.get("metadata", {}).get("model_name") or "gpt-4.1"
    model_provider = state.get("metadata", {}).get("model_provider") or "OPENAI"
    
    # Convert enum to string if necessary
    if hasattr(model_provider, 'value'):
        model_provider = model_provider.value
    
    return model_name, model_provider
