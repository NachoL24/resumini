from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from resumini.config import Settings, get_settings


def get_llm(settings: Settings | None = None) -> ChatOpenAI:
    if settings is None:
        settings = get_settings()
    return ChatOpenAI(
        model=settings.llm_model,
        temperature=settings.llm_temperature,
        api_key=settings.openai_api_key or None,
    )


def get_embeddings(settings: Settings | None = None) -> OpenAIEmbeddings:
    if settings is None:
        settings = get_settings()
    return OpenAIEmbeddings(
        model=settings.embedding_model,
        api_key=settings.openai_api_key or None,
    )
