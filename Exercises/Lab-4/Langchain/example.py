from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv


load_dotenv()

model1 = ChatGroq(
    model="openai/gpt-oss-20b",
    temperature=0
)

model2 = ChatGroq(
    model="openai/gpt-oss-20b",
    temperature=0
)

prompt1 = ChatPromptTemplate.from_template(
    "Summarize the following paragraph:\n\n{paragraph}"
)

prompt2 = ChatPromptTemplate.from_template(
    "Make the following summary shorter and clearer:\n\n{summary}"
)

parser = StrOutputParser()


chain = prompt1 | model1 | parser | prompt2 | model2 | parser

paragraph = input("Enter your paragraph: ")

result = chain.invoke({
    "paragraph": paragraph
})

print("\nFinal Summary:")
print(result)