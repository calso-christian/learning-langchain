from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, SystemMessage


from dotenv import load_dotenv

try:
    load_dotenv()
except ImportError:
    pass

model = init_chat_model('gpt-4o-mini',model_provider='openai')

messages=[
    SystemMessage(
        "You're a smart translator, Translate given text from English to Tagalog"
    ),
    HumanMessage(
        "Chair"
    )
]

for token in model.stream(messages):
    print(token.content, end=" | ")
