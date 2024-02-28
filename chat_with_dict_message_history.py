import uuid

import chainlit as cl
from langchain_openai import AzureChatOpenAI
from langchain_openai import AzureChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate, MessagesPlaceholder
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain.schema import StrOutputParser
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.runnables.base import RunnableSequence
from langchain.schema.runnable.config import RunnableConfig
from dotenv import load_dotenv

# Input .env path here
load_dotenv(dotenv_path='.venv/.env')


store: dict = {}


def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]


@cl.on_chat_start
async def on_start():
    # Set 'AZURE_OPENAI_API_KEY' and 'AZURE_OPENAI_ENDPOINT' in .env
    llm: AzureChatOpenAI = AzureChatOpenAI(
        azure_deployment="gpt-4-1106",
        openai_api_version="2023-09-01-preview",
    )
    # Change System Prompt here
    prompt: ChatPromptTemplate = ChatPromptTemplate.from_messages(
        [
            ("system", 'You are a helpful assistant.'),
            MessagesPlaceholder(variable_name="history"),
            ("human", "{question}"),
        ]
    )
    # Create chain of Prompt -> LLM -> Parse Output
    chain: RunnableSequence = prompt | llm | StrOutputParser()
    # Add message history to chain
    # Currently it uses a dictionary, but can be swapped to Redis or any other database
    runnable: RunnableWithMessageHistory = RunnableWithMessageHistory(
        chain,
        get_session_history,
        input_messages_key="question",
        history_messages_key="history",
    )
    # Run the below to get a graph representation of the runnable
    # runnable.get_graph().print_ascii()
    # Set the runnable for the session
    cl.user_session.set("runnable", runnable)


@cl.on_message
async def on_message(message: cl.Message):
    runnable: RunnableWithMessageHistory = cl.user_session.get("runnable")
    # Chainlit generates uuid per user session and this can be retrieved using the below command
    user_id: uuid.uuid4 = cl.user_session.get("id")
    # Below block of code is to call and stream responses
    msg: cl.Message = cl.Message(content="")
    async for chunk in runnable.astream(
            {"question": message.content},
            config=RunnableConfig(callbacks=[cl.AsyncLangchainCallbackHandler()],
                                  configurable={"session_id": user_id})
    ):
        await msg.stream_token(chunk)
    await msg.send()
