from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
from langchain.schema.runnable import RunnableSequence, RunnablePassthrough, RunnableParallel, RunnableLambda
from langchain_google_genai import ChatGoogleGenerativeAI

# custom python function
def word_counter(text):
    return len(text.split())
# runnable for the above function
runnable_word_counter = RunnableLambda(word_counter)

load_dotenv() 

prompt1 = PromptTemplate(
    template='Write a joke about {topic}',
    input_variables=['topic']
)

model = ChatGoogleGenerativeAI(model='gemini-1.5-flash', temperature=0)

parser = StrOutputParser()

joke_gen_chain = RunnableSequence(prompt1, model, parser)

parallel_chain = RunnableParallel({
    'joke':RunnablePassthrough(),
    'word_count':runnable_word_counter
})

final_chain = RunnableSequence(joke_gen_chain, parallel_chain)

result = final_chain.invoke({'topic': 'animal'})

final_result = """{} \n word count - {}""".format(result['joke'], result['word_count'])

print(final_result)

