
from langchain.chat_models import init_chat_model
import os
import streamlit as st
from dotenv import load_dotenv
import pandas as pd
import pandasql as ps

load_dotenv()


llm = init_chat_model(
    model = "llama-3.3-70b-versatile",
    model_provider = "openai",
    base_url = "https://api.groq.com/openai/v1",
    api_key = os.getenv("GROQ_API_KEY")
)

if 'conversation' not in st.session_state:
    st.session_state.conversation = [
    
        {"role": "system", "content": "You are SQLite expert developer with 10 years of experience."}
    ]

for msg in st.session_state.conversation:
        if msg["role"] != "system":
            with st.chat_message(msg["role"]):
                st.write(msg["content"])    

data_file=st.file_uploader("Upload your SQLite database file", type=["csv"]) 
if data_file:
    df = pd.read_csv(data_file)
    st.dataframe(df)  
    st.write("CSV schema: ")
    st.write(df.dtypes)

    user_input =st.chat_input("ask anything about the data: ")
    if user_input:
        st.session_state.conversation.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
           st.write("You:", user_input)
        
        llm_input = f"""
            Table Name: data 
            Table Schema: {df.dtypes}
            Question: {user_input}
            Instruction:
                Write a SQL query for the above question. 
                Generate SQL query only in plain text format and nothing else.
                If you cannot generate the query, then output 'Error'.
                
        """
        query = llm.invoke(llm_input)
        st.session_state.conversation.append({"role": "assistant", "content": query.content})
        st.write("generated SQL query: ", query.content)
        if "Error" not in query.content:

            result = ps.sqldf(query.content, {"data": df})
            st.write("Query Result: ")
            st.dataframe(result)

            llm_input_explain = f"""
                Table Name: data 
                Table Schema: {df.dtypes}
                SQL Query: {query.content}
                Query Result: {result}
                Instruction:
                    Explain the query result in English.
            """
            explanation = llm.invoke(llm_input_explain)
            st.session_state.conversation.append({"role": "assistant", "content": explanation.content})
            st.write("Explanation: ", explanation.content)

