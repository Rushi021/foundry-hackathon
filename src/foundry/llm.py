"""Provider-agnostic chat model factory."""

from langchain.chat_models import init_chat_model
from langchain_core.language_models.chat_models import BaseChatModel


def get_chat_model(model_spec: str, *, temperature: float = 0.2) -> BaseChatModel:
    """
    Create a chat model from a ``provider:model`` string.

    Examples:
        ``openai:gpt-4o``
        ``anthropic:claude-sonnet-4-5``
    """
    if ":" not in model_spec:
        raise ValueError(
            f"Invalid model spec '{model_spec}'. Use 'provider:model' "
            "(e.g. 'openai:gpt-4o-mini')."
        )
    return init_chat_model(model_spec, temperature=temperature)
