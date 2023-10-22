from langchain.document_loaders import YoutubeLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.llms import OpenAI
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.vectorstores import FAISS
from dotenv import load_dotenv

load_dotenv()

embeddings = OpenAIEmbeddings()

video = 'https://www.youtube.com/watch?v=6T7pUEZfgdI'

def create_vectordb(video_url: str) -> FAISS:
    loader = YoutubeLoader.from_youtube_url(video_url)
    transcript = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    docs = text_splitter.split_documents(transcript)

    db = FAISS.from_documents(docs, embedding=embeddings)

    return db



def get_response(db, query, k=4):
    # text-davinci can handle 4097 token

    docs = db.similarity_search(query, k=k)
    docs_page_content = " ".join([d.page_content for d in docs])

    llm = OpenAI(model='text-davinci-003')

    prompt = PromptTemplate(
        input_variables=["query", "docs"],

        template="""
        You are a helpful YouTube assistant that can answer questions about
        videos based in the video's transcript.

        Answer the following question: {query}
        By searching the following videp transcript: {docs}

        Only use the factual information from the transcript to answer the question.
        
        If you feel like you don't have enough information to answer the question,
        say "I don't know".

        Your answer should be detailed.
        """,
    )

    chain = LLMChain(llm=llm, prompt=prompt)

    response = chain.run(query=query, docs = docs_page_content)
    response = response.replace("\n", "")

    return response

query = "What did Jordan said about tribalism"
db = create_vectordb('https://www.youtube.com/watch?v=6T7pUEZfgdI')
response = get_response(db, query)
print(response)
    