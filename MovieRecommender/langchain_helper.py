from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.agents import load_tools, initialize_agent, AgentType
from dotenv import load_dotenv

load_dotenv()

def recommend_movie(topic, genre):
    llm = OpenAI(temperature=0.5)

    prompt_template = PromptTemplate(
        input_variables=['topic', 'genre'],
        template = "Suggest to me 3 {genre} movies about this topic: {topic}. Include a short description of each movie."
    )

    chain = LLMChain(llm=llm, prompt=prompt_template, output_key="movie")    

    response = chain({'topic': topic, 'genre': genre})

    return response

def langchain_agent():
    llm= OpenAI(temperature=0.5)

    tools= load_tools(["wikipedia"], llm=llm)

    agent = initialize_agent(
        tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True
    )

    result = agent.run(
        "What is the top 10 highest ranked movie on imdb? How old are these movies from now?"
    )

    print (result)


if __name__ == '__main__':
    langchain_agent()
