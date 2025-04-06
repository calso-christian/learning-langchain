from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate

from dotenv import load_dotenv

try:
    load_dotenv()
except ImportError:
    pass

model = init_chat_model('gpt-4o-mini',model_provider='openai')

system_prompt = "Translate the given text from Tagalog into {language}"

prompt_template = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("user", "{text}")
    ]
)
user_input="Gusto kong matuto ng AI Programming gamit ang LangChain"

prompt = prompt_template.invoke({"language":"Romanized Japanese", "text":user_input})

prompt.to_messages()

response=model.invoke(prompt)

print(response)