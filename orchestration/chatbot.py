from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, AIMessage, trim_messages
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START, MessagesState, StateGraph

from dotenv import load_dotenv

load_dotenv()

model = init_chat_model("gpt-4o-mini", model_provider="openai")


prompt_template = ChatPromptTemplate.from_messages(
    [
        ("system",
         "You're a helpful Assistant. You answer to the name Misha. If the input doesn't have the"
         "name Misha, Respond with you do not understand"
        ),
        MessagesPlaceholder(variable_name="messages"),
    ]
)

trimmer = trim_messages(
    max_tokens=65,
    strategy="last",
    token_counter=model,
    include_system=True,
    allow_partial=False,
    start_on="human",
)

# Define a new graph
workflow = StateGraph(state_schema=MessagesState)

# Define a function that calls the model given a MessagesState
def call_model(state: MessagesState):
    trimmed_messages = trimmer.invoke(state["messages"])
    prompt = prompt_template.invoke(state)

    print("\nMisha is typing...")

    stream=model.stream(prompt)

    full_text=""
    
    for chunk in stream:
        content = chunk.content if hasattr(chunk, "content") else chunk.delta
        if content:
            print(content, end="", flush=True)
            full_text += content
    
    print()
    # response = model.invoke(prompt)
    return {"messages":AIMessage(content=full_text)}


# Define the (single) node in the graph
workflow.add_edge(START, "model")
workflow.add_node("model", call_model)


# Add memory
memory = MemorySaver()
app = workflow.compile(checkpointer=memory)


def chatbot():
    config = {"configurable": {"thread_id": "abc123"}}

    while True:
        query = input("Me: ")

        if query.lower() == 'exit':
            break

        input_message=[HumanMessage(query)]

        output = app.invoke({"messages":input_message}, config)


if __name__ == "__main__":
    chatbot()