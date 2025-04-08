from langchain.chat_models import init_chat_model
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

from dotenv import load_dotenv

load_dotenv()

tagging_prompt = ChatPromptTemplate.from_template(
    """
    Extract the desired information from following passage.
    Only Extract the properties mentioned in the "Classification" Function.

    Passage:
    {input}

    """
)

class Classification(BaseModel):
    sentiment: str = Field(description="The sentiment of the Text")
    aggressiveness: int = Field(description="How aggressive is the text from the scale of 1-10")
    language: str = Field(description="The language the text is written")


llm = init_chat_model('gpt-4o-mini', model_provider='openai').with_structured_output(Classification)



response=llm.invoke(tagging_prompt.invoke({'input':'¡Estoy harto de tus mentiras!, كفاية بقى! مش قادر أتحمّل أكتر من كده!'}))

print(f"Object: {response}\n")

print(f"Dictionary: {response.model_dump()}")