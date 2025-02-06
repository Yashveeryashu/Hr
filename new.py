import os
import pandas as pd
import streamlit as st
# import matplotlib.pyplot as plt
import seaborn as sns
from langchain_openai import AzureChatOpenAI
from langchain.agents.agent_types import AgentType
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent

# Set Azure OpenAI Credentials
AZURE_ENDPOINT = "https://firstinsightaoai.openai.azure.com/"
API_KEY = '5f4cc617ad2343b794045f85185fa6ee'  # Your API key as a string
API_VERSION = "2024-02-01"
MODEL_NAME = "firstinsightdeployment"

# Apply custom CSS to align everything to the left and adjust spacing
st.markdown("""
    <style>
        # .block-container {
        #     margin: 0;
        #     padding: 0;
        # }
        .stTextInput, .stFileUploader {
            width: 45% !important;
        }
        .section {
            margin-bottom: 0.1%;
        }
    </style>
""", unsafe_allow_html=True)

# Streamlit UI Setup
st.title("📊 HR Dashboard")
st.markdown("<h2>Enter Your URL or Upload Your File</h2>", unsafe_allow_html=True)

# URL Input Section
st.subheader("🔍 Enter URL")
st.markdown('<div class="section">', unsafe_allow_html=True)
url_input = st.text_input("🌐 Enter a URL (e.g., Yahoo Finance):")
st.markdown('</div>', unsafe_allow_html=True)

if url_input:
    user_question_url = st.text_input("💬 Ask a question related to the URL:")
    if user_question_url:
        st.write(f"🔎 You asked about: {user_question_url}")

# File Upload Section
st.subheader("📂 Upload Your Data File")
uploaded_file = st.file_uploader("Choose an Excel file", type=["xlsx"])

if uploaded_file:
    df = pd.read_excel(uploaded_file)
    st.write("🔍 Data Preview:")
    st.write(df.head())

    # User input for questions related to the uploaded file
    user_question = st.text_input("💬 Ask a question about the data:")

    if user_question:
        if "salary distribution" in user_question.lower():
            # Generate the Salary Distribution Plot
            st.write("📊 Salary Distribution Plot:")
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.histplot(df['MonthlyIncome'], bins=30, kde=True, ax=ax)
            ax.set_title('Salary Distribution')
            ax.set_xlabel('Monthly Income')
            ax.set_ylabel('Frequency')
            st.pyplot(fig)
        else:
            # Create the LangChain agent with allow_dangerous_code=True
            agent = create_pandas_dataframe_agent(
                AzureChatOpenAI(
                    temperature=0,
                    azure_endpoint=AZURE_ENDPOINT,
                    openai_api_key=API_KEY,
                    api_version=API_VERSION,
                    deployment_name=MODEL_NAME
                ),
                df,
                verbose=True,
                allow_dangerous_code=True,
                agent_type=AgentType.OPENAI_FUNCTIONS
            )

            # Process user query
            with st.spinner("⚡ Processing..."):
                result = agent.invoke(user_question)

            # Clean up the result to only show relevant data
            cleaned_result = result['output']
            lines = cleaned_result.split("\n")
            formatted_result = "\n".join([line.strip() for line in lines if line.strip()])

            # Display the cleaned and formatted result
            st.write("✅ Answer:")
            st.write(formatted_result)