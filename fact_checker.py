#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import streamlit as st
from langchain.chains import LLMChain, SimpleSequentialChain
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate


def main():
    OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY', None)
    # Set page title
    st.title("LangChain Fact Checker")
    if not OPENAI_API_KEY:
        # If OPENAI_API_KEY has not been provided, display a warning
        st.warning("An OPENAI_API_KEY is required to use LangChain Fact Checker")
        # Prompt for the API key
        OPENAI_API_KEY = st.text_input(
            "Please enter your OpenAI API key",
            type="password"
        )

    llm = OpenAI(temperature=0.7, openai_api_key=OPENAI_API_KEY)

    # Create a text input to get the user's question
    question = st.text_input(
        "What is your question?",
        placeholder="e.g. What is the capital of France?"
    )

    # Pass the question through SimpleSequentialChain pipeline
    if st.button("Submit", type="primary"):
        # Chain 1: Generate a rephrased version of the question
        template = """{question}\n\n"""
        prompt_template = PromptTemplate(template=template, input_variables=["question"])
        question_chain = LLMChain(llm=llm, prompt=prompt_template)

        # Chain 2: Generate assumptions
        template = """
            Given the following statement:
            {statement}
            Provide a list of assumptions made when producing the above statement.\n\n
        """
        prompt_template = PromptTemplate(template=template, input_variables=["statement"])
        assumptions_chain = LLMChain(llm=llm, prompt=prompt_template)
        #assumptions_chain_sequence = SimpleSequentialChain(
        #    chains=[question_chain, assumptions_chain],
        #    verbose=True
        #)

        # Chain 3: Fact check assertions
        template = """
            Given the following list of assertions:
            {assertions}
            Determine whether each assertion is true or false.
            If it is false, explain why it is false.\n\n
        """
        prompt_template = PromptTemplate(template=template, input_variables=["assertions"])
        fact_check_chain = LLMChain(llm=llm, prompt=prompt_template)
        #fact_check_chain_sequence = SimpleSequentialChain(
        #    chains=[question_chain, assumptions_chain, fact_check_chain],
        #    verbose=True
        #)

        # Chain 4: Answer the question
        template = """
            {facts}
            In light of the above facts, how would you answer the following question:
        """ + question
        prompt_template = PromptTemplate(template=template, input_variables=["facts"])
        answer_chain = LLMChain(llm=llm, prompt=prompt_template)

        # Run the chains
        sequential_chain = SimpleSequentialChain(
            chains=[
                question_chain,
                assumptions_chain,
                fact_check_chain,
                answer_chain
            ],
            verbose=True
        )
        st.success(sequential_chain.run(question))


if __name__ == '__main__':
    main()
