from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda

import os

def get_llm():
    return ChatMistralAI(model = "mistral-small-latest", mistral_api_key = os.getenv("MISTRAL_API_KEY"),temperature = 0.3)


def build_chain(system_prompt: str):
    llm = get_llm()

    return (
        RunnablePassthrough() | RunnableLambda(lambda x: {"text": x}) | ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", "{text}"),
        ]) | llm | StrOutputParser()
    )

def extract_action_items(transcript: str) -> str:
    chain = build_chain(
        "You are an expert video analyst. from the video transcript,"
        "extract all the action items. for each provide:\n"
        "_ Task description\n"
        "_ Owner (who is responsible for the task)\n"
        "_ deadline (if mentioned, else write 'Not specified')\n\n"
        "Format as the numbered list. if none found say 'No action items found.'"
    )
    return chain.invoke(transcript)

def extract_key_decisions(transcript: str) -> str:
    chain = build_chain(
        "You are an expert video analyst. from the video transcript,"
        "extract all the key decisions made. format as a numbered list. " 
        "If none found, say 'No key decisions found.'"

    )
    return chain.invoke(transcript)

def extract_questions(transcript: str) -> str:
    chain = build_chain(
        "From the video transcript, extract all unresolved questions."
        "or topics needing follow-up. format as a number list"
        "If none found, say 'No open questions found.'"
    )
    return chain.invoke(transcript)

