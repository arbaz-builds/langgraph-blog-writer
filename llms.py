from langchain_openai import ChatOpenAI
import config

router_llm = ChatOpenAI(
    base_url=config.BASE_URL,
    api_key=config.API_KEY,
    model="deepseek-ai/deepseek-v4-flash",
    temperature=0,
    timeout=15
)

general_LLM = ChatOpenAI(
    base_url=config.BASE_URL,
    api_key=config.API_KEY,
    model="openai/gpt-oss-120b",
    timeout=60
)

fallback_LLM = ChatOpenAI(
    base_url=config.BASE_URL,
    api_key=config.API_KEY,
    model="mistralai/mistral-nemotron",
    timeout=60
)
