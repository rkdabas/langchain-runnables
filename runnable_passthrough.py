from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
from langchain.schema.runnable import RunnableSequence, RunnablePassthrough, RunnableParallel
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv() 

prompt1 = PromptTemplate(
    template='Write a joke about {topic}',
    input_variables=['topic']
)

model = ChatGoogleGenerativeAI(model='gemini-1.5-flash', temperature=0)

parser = StrOutputParser()

prompt2 = PromptTemplate(
    template='explain the following joke - {text}',
    input_variables=['text']
)

joke_generator_chain = RunnableSequence(prompt1, model, parser)
parallel_chain = RunnableParallel({
    'joke':RunnablePassthrough(),
    'explanation':RunnableSequence(prompt2, model, parser)
}
)

final_chain = RunnableSequence(joke_generator_chain, parallel_chain)

result = final_chain.invoke({'topic': 'pen'})

print(result)
