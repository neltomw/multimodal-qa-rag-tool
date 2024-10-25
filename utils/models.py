from langchain.chat_models import ChatOpenAI
from langchain.llms import OpenAI
from typing import Dict, Any
import os
from dotenv import load_dotenv

load_dotenv()  # Load environment variables from .env file

# Default model configurations
DEFAULT_CHAT_MODEL = "gpt-4-turbo"
DEFAULT_COMPLETION_MODEL = "text-davinci-003"
DEFAULT_TEMPERATURE = 0

def get_chat_model(model_name: str = DEFAULT_CHAT_MODEL, 
                   temperature: float = DEFAULT_TEMPERATURE, 
                   **kwargs: Any) -> ChatOpenAI:
    """
    Get a ChatOpenAI model instance.

    Args:
        model_name (str): The name of the model to use.
        temperature (float): The temperature for the model's output.
        **kwargs: Additional keyword arguments for the ChatOpenAI constructor.

    Returns:
        ChatOpenAI: An instance of the ChatOpenAI model.
    """
    return ChatOpenAI(
        model_name=model_name,
        temperature=temperature,
        openai_api_key=os.getenv("OPENAI_API_KEY"),
        **kwargs
    )

def get_completion_model(model_name: str = DEFAULT_COMPLETION_MODEL, 
                         temperature: float = DEFAULT_TEMPERATURE, 
                         **kwargs: Any) -> OpenAI:
    """
    Get an OpenAI completion model instance.

    Args:
        model_name (str): The name of the model to use.
        temperature (float): The temperature for the model's output.
        **kwargs: Additional keyword arguments for the OpenAI constructor.

    Returns:
        OpenAI: An instance of the OpenAI model.
    """
    return OpenAI(
        model_name=model_name,
        temperature=temperature,
        openai_api_key=os.getenv("OPENAI_API_KEY"),
        **kwargs
    )

def get_model_by_type(model_type: str, **kwargs: Any) -> Any:
    """
    Get a model instance based on the specified type.

    Args:
        model_type (str): The type of model to get ('chat' or 'completion').
        **kwargs: Additional keyword arguments for the model constructor.

    Returns:
        Any: An instance of the specified model type.
    """
    if model_type == 'chat':
        return get_chat_model(**kwargs)
    elif model_type == 'completion':
        return get_completion_model(**kwargs)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
