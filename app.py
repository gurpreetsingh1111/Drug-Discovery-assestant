import streamlit as st
import cohere
import weaviate
import bs4
from bs4 import BeautifulSoup
import json
import requests 
import time
import numpy as np
from PIL import Image
from langchain.chains import LLMChain
from langchain.llms import Cohere
from langchain.prompts import PromptTemplate

# from langchain.chat_models import ChatCohere
# from langchain.schema import HumanMessage, AIMessage

MED_PROMPT = """
You are an expert biomedical researcher who has keen interest in exploring the latest research trends in drug discovery.
While working, you found the following sources
Sources : {0}\n\n
You have to refer to the above resource and answer the following question on drug discovery specific to the above sources.
Question : {1}
"""
cohere_api_key = st.secrets["cohere_api_key"]
co = cohere.Client(cohere_api_key)


def get_client():
# Weaviate config
    weaviate_cluster_url = st.secrets["weaviate_cluster_url"]
    weaviate_api_key = st.secrets["weaviate_api_key"]
    weaviate_index_name = st.secrets["weaviate_index_name"]

# Cohere config
    cohere_api_key = st.secrets["cohere_api_key"]
    
# Create Weaviate Cloud Services client
    client = weaviate.Client(
        url = weaviate_cluster_url,                                           # Replace with your cluster URL
        auth_client_secret=weaviate.AuthApiKey(api_key=weaviate_api_key),     # Replace w/ your Weaviate instance API key
        additional_headers = {
            "X-Cohere-Api-Key": cohere_api_key                                # Replace with your inference API key
        }
    )
    return client   

def get_chunks(query, client, limit=3):
    chunks, sources = [],[]
    response = client.query.get("Document", ["source","abstract"]).with_bm25(query=query).with_limit(limit).do()
    response_list = response['data']['Get']['Document']
    for r in response_list:
        chunks.append(r['abstract'])
        sources.append(r["source"])
    content = ["Source: "+c+"\n\n" for c in chunks]
    return content, sources

st.title("Drug Discovery Assist 🥼🔬🧬")
# st.image("hero.jpeg")

img = Image.open('./hero.jpeg')
numpydata = np.asarray(img)
st.image(numpydata)
if "messages" not in st.session_state:
    st.session_state = {"messages":[], "cohere_model":"command"}

# Display chat messages from history on app rerun
# for message in st.session_state["messages"]:
#     with st.chat_message(message["user_name"]):
#         st.markdown(message["text"])

# Accept user input
home , search = st.tabs(["Research Chat (Home)","Drug Search (beta)"])

with search:
    with open('./data.json', 'r') as f:
        data = json.load(f)
    if name := st.selectbox(label="Search drug", options=list(data.keys())):
        st.button("Submit", type="primary")
        time.sleep(10)
        if st.button:
            info = ""
            with st.spinner("Generating response ....."):
                while info == "":
                    if len(name)>0:
                        with open('./data.json', 'r') as f:
                            data = json.load(f) 
                        # print(data)
                        id_ = data[name.lower()]
                        url = st.secrets["web_url"]+id_
                        html_content = requests.get(url).text
                        soup = BeautifulSoup(html_content, "html.parser")
                        texts = soup.find_all('p')
                        texts = [i for i in texts[:-1] if i != "Not Available"][:2]
                        prompt_template = "Write a short summary on {name} \n {source}, don't ask questions"
                        llm = Cohere(max_tokens=500, temperature=0, p=1, cohere_api_key=cohere_api_key)
                        llm_chain = LLMChain(llm=llm, prompt=PromptTemplate.from_template(prompt_template))
                        info = llm_chain({"name":name, "source":texts})["text"]
                        st.markdown(info)
    

if query:=st.chat_input(" "):
    st.session_state["messages"].append({"user_name": "User", "text": query})
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(query)
    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        with st.spinner("Generating response ....."):
            while full_response == "":    
                client = get_client()
                content, sources = get_chunks(query, client)
                prompt = MED_PROMPT.format(content, query)
                print("Prompt:",prompt)
                response = co.chat(
                    model=st.session_state["cohere_model"],
                    message=prompt,
                    chat_history=st.session_state["messages"],
                    prompt_truncation='AUTO'
                )
                    
                full_response += (response.text or "")
                if sources != []:
                    full_response += "\n\n\nThese are some sources you can refer to based on you query\n"
                    full_response += "\n".join(["https://arxiv.org/pdf/"+i for i in sources]) 
                message_placeholder.markdown(full_response + "▌")
                message_placeholder.markdown(full_response) 
            st.session_state["messages"].append({"user_name": "Chatbot", "text": response.text})
