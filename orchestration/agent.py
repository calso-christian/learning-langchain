from dotenv import load_dotenv

from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_openai import ChatOpenAI
from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage
from langgraph.prebuilt import create_react_agent

load_dotenv()


search = TavilySearchResults(max_results=2)


model = init_chat_model('gpt-4o-mini', model_provider='openai')


tools = [search,]

model_with_tools = model.bind_tools(tools)

agent_executor = create_react_agent(model, tools)

response = agent_executor.invoke(
    {
        "messages": [HumanMessage(content="What is the weather like in General Trias, Cavite and Ibaraki, Japan")]
    }
)

print(response['messages'])