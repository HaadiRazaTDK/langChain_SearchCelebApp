## Integrating code with openAI API

import os
from constants import openai_key
from langchain_community.llms import OpenAI
os.environ["OPENAI_API_KEY"] = openai_key
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.chains.sequential import SequentialChain
from langchain.memory import ConversationBufferMemory

import streamlit as st


## Streamlit framework

st.title("Celebrity Search Application")
input_text = st.text_input("Enter the topic you want to search. ")

## Prompt Templates

first_input_prompt = PromptTemplate(
    input_variables = ['name'],
    template = "Tell me about {name}"
)

## Memory
person_memory = ConversationBufferMemory(
    input_key = 'name', 
    memory_key = 'chat_history'
    )

dob_memory = ConversationBufferMemory(
    input_key = 'person',
    memory_key = 'chat_history',
)

event_memory = ConversationBufferMemory(
    input_key = 'dob',
    memory_key = 'event_history'
)



## OPENAI LLMs
llm = OpenAI(temperature = 0.8) 
chain = LLMChain(llm = llm, 
                 prompt = first_input_prompt, 
                 verbose = True, 
                 output_key = 'person',
                 memory = person_memory,
                 )

second_input_prompt = PromptTemplate(
    input_variables = ['person'],
    template = "When was {person} was born"
)
chain2 = LLMChain(llm = llm, 
                  prompt = second_input_prompt, 
                  verbose = True, 
                  output_key = 'dob',
                  memory = dob_memory,
                  )

third_input_prompt = PromptTemplate(
    input_variables = ['dob'],
    template = "mention 5 major events happened around {dob} in the world"
)

chain3 = LLMChain(llm = llm, 
                  prompt = third_input_prompt, 
                  verbose = True, 
                  output_key = 'events',
                  memory = event_memory,
                  )

parent_chain = SequentialChain(
    chains=[chain, chain2, chain3], 
    input_variables=['name'], 
    output_variables=['person', 'dob', 'events'], 
    verbose=True
)

if input_text:
    st.write(chain.run({'name': input_text}))

    with st.expander('Person Name'):
        st.info(person_memory.buffer)

    with st.expander('Major Events'):
        st.info(event_memory.buffer)