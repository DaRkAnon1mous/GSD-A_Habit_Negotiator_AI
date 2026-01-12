# chat.py (v0.3 - modern memory, better prompt)

import os
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.output_parsers import StrOutputParser

from prompts.habit_nudge import SYSTEM_PROMPT

# Config
MODEL = "llama3.2:latest"          # or "llama3.2:3b", etc.
TEMPERATURE = 0.65

llm = ChatOllama(model=MODEL, temperature=TEMPERATURE)

prompt = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PROMPT),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{input}"),
])

chain = prompt | llm | StrOutputParser()

# In-memory store for session histories (later → DB)
store = {}  # key: session_id, value: ChatMessageHistory

def get_session_history(session_id: str):
    if session_id not in store:
        store[session_id] = InMemoryChatMessageHistory()
    return store[session_id]

# Runnable with history
with_history = RunnableWithMessageHistory(
    chain,
    get_session_history,
    input_messages_key="input",
    history_messages_key="history",
)

print("HabitNudge v0.3 ready! (modern memory - type 'exit' to quit)\n")
print("Session ID: default_session (change in code if multi-user later)\n")

session_id = "default_session"

while True:
    user_input = input("You: ").strip()
    if user_input.lower() in ['exit', 'quit', 'q']:
        print("Bye! Keep stacking those tiny wins 💪")
        break

    if not user_input:
        continue

    print("\nHabitNudge: ", end="", flush=True)

    # Stream response
    full_response = ""
    for chunk in with_history.stream(
        {"input": user_input},
        config={"configurable": {"session_id": session_id}},
    ):
        print(chunk, end="", flush=True)
        full_response += chunk

    print("\n")