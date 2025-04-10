from pydantic import BaseModel, Field
from typing import Optional

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

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
