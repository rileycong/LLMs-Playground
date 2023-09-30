''' 
    build an app that takes image input 
    and create an audio story from that
'''

from dotenv import find_dotenv, load_dotenv
from transformers import pipeline #to download the modelS from huggingface
from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
import requests
import os
import streamlit as st

load_dotenv(find_dotenv())
HUGGINGFACEHUB_API_TOKEN = os.getenv('HUGGINGFACEHUB_API_TOKEN')

# image to text
def img2text(url):
    img_to_text = pipeline("image-to-text",model="Salesforce/blip-image-captioning-base", max_new_tokens=15)
    text = img_to_text(url)[0]['generated_text']

    #print(text)
    return text

# llm
def gen_story(scenario):
    llm = ChatOpenAI(model_name='gpt-3.5-turbo', temperature=1)

    template = '''
    Act as a story teller.
    Generate a short story based on a simple narrative. The story should be no more than 200 words;

    CONTEXT: {scenario}
    STORY:
    '''

    prompt = PromptTemplate(template=template, input_variables=["scenario"])

    story_llm = LLMChain(llm=llm, prompt=prompt, verbose=True)

    story = story_llm.predict(scenario=scenario)
    #print(story)
    
    return story

# text to speech
def text2speech(message):
    API_URL = "https://api-inference.huggingface.co/models/espnet/kan-bayashi_ljspeech_vits"
    headers = {"Authorization": f"Bearer {HUGGINGFACEHUB_API_TOKEN}"}
    
    payloads = {
        "inputs": message
    } 

    response = requests.post(API_URL, headers=headers, json=payloads)
    with open('audio.flac', 'wb') as file:
        file.write(response.content)


def main():
    st.set_page_config(page_title="Image to Audio Story", page_icon="📖")
    st.header("Turn any picture into an audio story 📖")
    uploaded_file = st.file_uploader("Choose an image...", type=['jpg','jpeg'])

    if uploaded_file is not None:
        if os.path.exists('audio.flac'):
            os.remove('audio.flac')
            
        if os.path.exists(uploaded_file.name):
            os.remove(uploaded_file.name)

        bytes_data = uploaded_file.getvalue()
        with open(uploaded_file.name, "wb") as file:
            file.write(bytes_data)
        st.image(uploaded_file, caption='Uploaded Image', use_column_width=True)
        
        scenario = img2text(uploaded_file.name)
        story = gen_story(scenario)
        text2speech(story)

        with st.expander("Scenario"):
            st.write(scenario)
        with st.expander("Story"):
            st.write(story)

        st.audio("audio.flac")

if __name__ == '__main__':
    main()

# scenario = img2text('image.jpg')
# story = gen_story(scenario)
# text2speech(story)