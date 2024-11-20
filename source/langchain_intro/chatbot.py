#To Install use pip install python-dotenv
import dotenv
from langchain_openai import ChatOpenAI
from langchain.prompts import (PromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate, ChatPromptTemplate,)
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain.schema.runnable import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain.agents import(create_openai_functions_agent, Tool, AgentExecutor)
from langchain import hub
from langchain_intro.tools import current_wait_time

#Used to read and load environment variables from ".env" file
dotenv.load_dotenv()

review_template_str = """Your job is to use patient
reviews to answer questions about their experience at
a hospital. Use the following context to answer questions.
Be as detailed as possible, but don't make up any information
that's not from the context. If you don't know an answer, say
you don't know.

{context}
"""
#Path where we store the vector database
REVIEWS_CHROMA_PATH = "chroma_data/"
#Giving instruction to system to how to answer
review_system_prompt = SystemMessagePromptTemplate(prompt=PromptTemplate(input_variables=["context"],template=review_template_str,))
#This is giving questions by human
review_human_prompt = HumanMessagePromptTemplate(prompt=PromptTemplate(input_variables=["question"], template="{question}",))
messages = [review_system_prompt, review_human_prompt]
review_prompt_template = ChatPromptTemplate(input_variables=["context", "question"], messages=messages,)

#For instanticiate model
chat_model = ChatOpenAI(model="gpt-3.5-turbo-0125", temperature=0)
#Removes unnecessary information and gives good string formation output
output_parser = StrOutputParser()
# Creating chroma vector DB
reviews_vector_db = Chroma(persist_directory=REVIEWS_CHROMA_PATH,embedding_function=OpenAIEmbeddings())
#Retreving the 10 revies from vector db
reviews_retriever  = reviews_vector_db.as_retriever(k=10)
#Used to link the between the reviews(stored as embedings), question, prompts and model and output parser
review_chain = ({"context": reviews_retriever , "question": RunnablePassthrough()}|review_prompt_template | chat_model | output_parser)
tools =[Tool(name = "Reviews", 
             func=review_chain.invoke, 
             description= """Useful when you need to answer questions
        about patient reviews or experiences at the hospital.
        Not useful for answering questions about specific visit
        details such as payer, billing, treatment, diagnosis,
        chief complaint, hospital, or physician information.
        Pass the entire question as input to the tool. For instance,
        if the question is "What do patients think about the triage system?",
        the input should be "What do patients think about the triage system?"
        """),
        Tool(name= "Waits",
             func= current_wait_time,
             description= """Use when asked about current wait times
        at a specific hospital. This tool can only get the current
        wait time at a hospital and does not have any information about
        aggregate or historical wait times. This tool returns wait times in
        minutes. Do not pass the word "hospital" as input,
        only the hospital name itself. For instance, if the question is
        "What is the wait time at hospital A?", the input should be "A".
        """

        )]

#Used to give instructions to agent to how to answer and which tools need to use
hospital_agent_prompt = hub.pull("hwchase17/openai-functions-agent")
#Instanciate Model
agent_chat_model = ChatOpenAI(model= "gpt-3.5-turbo-1106", temperature=0)
#Creating agent for which model to use and tools, prompt
hospital_agent = create_openai_functions_agent(llm= agent_chat_model, prompt= hospital_agent_prompt, tools= tools)
#Agentexecutor will execute the agent created and return results, intermidate and verbose used to know debugging and logging information of agent
hospital_agent_executor = AgentExecutor(agent=hospital_agent, tools=tools, return_intermediate_steps= True, verbose = True)

