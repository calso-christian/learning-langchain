from pydantic import BaseModel, Field
from typing import Optional
from dotenv import load_dotenv

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chat_models import init_chat_model

load_dotenv()


class Person(BaseModel):
    name: Optional[str] = Field(default=None, description="The name of the person from the text")
    hair_color: Optional[str] = Field(default=None, description="The color of the hair of the person if stated")
    height_in_meters: Optional[str] = Field(default=None, description="Height of the person in Meters")


prompt_template = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You're an expert extraction algorithm."
            "Extract information only from the text provided."
            "If you cannot extract a specific information."
            "return a null value"
        ),
        (
            "human",
            "{input_text}"
        )
    ]
)


llm = init_chat_model("gpt-4o-mini", model_provider="openai")

structured_llm = llm.with_structured_output(schema=Person)

text="Christian Paul Calso is a 5.6FT person with a black hair"

prompt = prompt_template.invoke({'input_text':text})
response=structured_llm.invoke(prompt)

print(response)