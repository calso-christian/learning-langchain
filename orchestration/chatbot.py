from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, AIMessage

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START, MessagesState, StateGraph

from dotenv import load_dotenv

load_dotenv()

model = init_chat_model("gpt-4o-mini", model_provider="openai")

# Define a new graph
workflow = StateGraph(state_schema=MessagesState)

# Define a function that calls the model given a MessagesState
def call_model(state: MessagesState):
    response = model.invoke(state['messages'])
    return {"messages":response}

# Define the (single) node in the graph
workflow.add_edge(START, "model")
workflow.add_node("model", call_model)

# Add memory
memory = MemorySaver()
app = workflow.compile(checkpointer=memory)



# input_message=[HumanMessage("Hi, I'm Christian, I was born in October 19, 2000")]
# output= app.invoke({"messages":input_message}, config)
# output["messages"][-1].pretty_print()


# input_message=[HumanMessage("How old am I today?")]
# output= app.invoke({"messages":input_message}, config)
# output["messages"][-1].pretty_print()

# input_message=[HumanMessage("What do you think is my gender?")]
# output= app.invoke({"messages":input_message}, config)
# output["messages"][-1].pretty_print()

# input_message=[HumanMessage("What is the day of my birthday?")]
# output= app.invoke({"messages":input_message}, config)
# output["messages"][-1].pretty_print()


def chatbot():
    config = {"configurable": {"thread_id": "abc123"}}

    while True:
        query = input("Me: ")

        if query.lower() == 'exit':
            break

        input_message=[HumanMessage(query)]

        output = app.invoke({"messages":input_message}, config)
        output['messages'][-1].pretty_print()


if __name__ == "__main__":
    chatbot()