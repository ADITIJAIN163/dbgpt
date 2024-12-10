import streamlit as st
import os
import openai
import dtale
#from pandas_profiling import ProfileReport
import sweetviz as sv
from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient
from azure.search.documents.models import QueryType
from azure.search.documents.models import VectorizableTextQuery
from azure.search.documents.models import RawVectorQuery
from azure.search.documents.models import RawVectorQuery
from azure.core.credentials import AzureKeyCredential
import ast  
#from execute_query import execute_query_function
import json
import pandas as pd
import pyodbc 
#from speech_to_text import recognize_from_microphone
OPENAI_API_KEY = ""
OPENAI_API_ENDPOINT = ""
OPENAI_API_VERSION = "2024-02-15-preview"
 
AZURE_COGNITIVE_SEARCH_SERVICE_NAME = ""
AZURE_COGNITIVE_SEARCH_API_KEY = ""
AZURE_COGNITIVE_SEARCH_ENDPOINT = ""
AZURE_COGNITIVE_SEARCH_INDEX_NAME  = "schema-bikestore"
azure_credential = AzureKeyCredential(AZURE_COGNITIVE_SEARCH_API_KEY)
 
 
 
############################################################## Log-IN Module ##########################################################    

import base64
st.markdown("""
    <style>
    .title-font {
        color: #00008B;
        font-size: 42px;
    }
    </style>
    """, unsafe_allow_html=True)

st.markdown("<h1 class='title-font'>Chat With Db-Gpt🧑🏻‍💻</h1>", unsafe_allow_html=True)
def get_base64_of_bin_file(bin_file):
    """
    Read image file and return its base64 encoded string
    """
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()

def set_background(jpg_file):
    """
    Set background image for Streamlit app
    """
    bin_str = get_base64_of_bin_file(jpg_file)
    page_bg_img = '''
    <style>
    .stApp {
        background-image: url("data:image/jpg;base64,%s");
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
        background-attachment: fixed;
    }
    .stSidebar {
        background-image: url("data:image/png;base64,%s");
        
        background-position: center;
        background-repeat: no-repeat;
        background-attachment: fixed;
        opacity: 0.9;
    }
    .stChatInput textarea {
        background-color: transparent !important;
        color: black !important;
    } .stMarkdown, .stSelectbox, .stRadio, .stCheckbox {
        background-color: rgba(255,255,255,0.7);
        border-radius: 5px;
        padding: 10px;
    }
    </style>
    ''' % (bin_str, bin_str)
    st.markdown(page_bg_img, unsafe_allow_html=True)

set_background('bg2.png')

st.sidebar.markdown("<h2 style='font-weight: bold; text-align: center; font-size: 24px;color: white; background-color: #00008B;'>Db-Gpt</h2>", unsafe_allow_html=True)
st.sidebar.image("img_logo.jpg", use_column_width=True)
dropdown_3_prompt = " "
modell = st.sidebar.selectbox('**Select Model**',("Azure OpenAI GPT-3.5",'Azure OpenAI GPT-4','Azure OpenAI GPT-4o'))
 
if modell == "Azure OpenAI GPT-3.5" :
    model_to_use = "gpt3516k"
   
if modell == "Azure OpenAI GPT-4" :
    model_to_use = "gpt4-demo"
   
if modell == "Azure OpenAI GPT-4o" :
    model_to_use = "gpt-4o"

selected_database = st.sidebar.selectbox('**Select Database**',("Azure SQL",'SQL Server'))
 
if selected_database == "Azure SQL" :
    index_to_use = "schema-bikestore"
   
if selected_database == "SQL Server" :
    index_to_use = "schema-sqlserver"
 
 
 
pmt="""
You are an Expert SQL server developer and you have the ability to generate query on the basis of user prompts aksed in natural language.Apart from that you also have a knowledge about the domains like sales, e-commerce and many more and with the help of that knowledge you can generate the perfect sql queries on user asks.To generate the queries you took the information from knowledge base contains the schema information for the database and consider all the relation and contraints with your experience in sql to generate the required query.
Your primary goal is to understand the user's information needs, formulate the most accurate SQL query to retrieve the relevant data, and present the results in a clear and concise manner. You should leverage the semantic and vector-based search capabilities of Azure Cognitive Search to generate the most relevant queries.
When a user provides a natural language query, you should:
Analyze the query to understand the user's intent and the specific data they are looking for.
Use the database schema information in your knowledge base to map the user's query to the relevant tables, columns, and relationships in the database.
Construct an efficient SQL query that retrieves the requested data, leveraging techniques like joins, filters, and aggregations as needed.
Execute the query against the SQL Server database and format the results in a clear, concise, and user-friendly way.
Always rembember your responses should contain only the sql query that I can directly pass to my sql server to get the desired results.Never include anything other than sql query not even extra single quotes, escape sequences, brackets.No other explainations are needed. Just give the exact without any other word. Don't even generate the sql query in triple quotes just generate sql query.
"""
 
 
######################################################### Neom ##########################################################    
use_memory = st.sidebar.checkbox('**Keep History**')

 
######################################################### Neom ##########################################################    
   
 
######################################################### Neom ##########################################################
 
######################################################### Neom ##########################################################
import os
from openai import AzureOpenAI
 
client = AzureOpenAI(
  api_key = OPENAI_API_KEY,  
  api_version = OPENAI_API_VERSION,
  azure_endpoint = OPENAI_API_ENDPOINT
)
 
def generate_embeddings_azure_openai(text = " "):
    response = client.embeddings.create(
        input = text,
        model= "embedding-model"
    )
    return response.data[0].embedding
 
 
 
 
def call_gpt_model(model= model_to_use,
                                  messages= [],
                                  temperature=0.1,
                                  max_tokens = 700,
                                  stream = True):
 
    print("Using model :",model_to_use)
 
    response = client.chat.completions.create(model=model_to_use,
                                              messages=messages,
                                              temperature = temperature,
                                              max_tokens = max_tokens,
                                              stream= stream)
 
    return response
   
system_message_query_generation_for_retriver = """
You are a very good text analyzer.
You will be provided a chat history and a user question.
You task is generate a search query that will return the best answer from the knowledge base.
Try and generate a grammatical sentence for the search query.
Do NOT use quotes and avoid other search operators.
Do not include cited source filenames and document names such as info.txt or doc.pdf in the search query terms.
Do not include any text inside [] or <<>> in the search query terms.
"""
 
 
def generate_query_for_retriver(user_query = " ",messages = []):
 
    start = time.time()
    user_message = summary_prompt_template = """Chat History:
    {chat_history}
 
    Question:
    {question}
 
    Search query:"""
 
    user_message = user_message.format(chat_history=str(messages), question=user_query)
 
    chat_conversations_for_query_generation_for_retriver = [{"role" : "system", "content" : system_message_query_generation_for_retriver}]
    chat_conversations_for_query_generation_for_retriver.append({"role": "user", "content": user_message })
 
    response = call_gpt_model(messages = chat_conversations_for_query_generation_for_retriver,stream = False ).choices[0].message.content
    print("Generated Query for Retriver in :", time.time()-start,'seconds.')
    print("Generated Query for Retriver is :",response)
 
    return response

def setup_database():
    if index_to_use=="schema-sqlserver":
        conn_str = "Driver={SQL Server Native Client 11.0};Server=(LocalDB)\MSSQLLocalDB;Database=Northwind;Trusted_Connection=yes;"  
    if index_to_use=="schema-bikestore":
        conn_str = "Driver={SQL Server Native Client 11.0};Server=sqlchatdemodbserver.database.windows.net;Database=BikeStore;UID=SqlChatDb;PWD=Junaid@123;"  

    cnxn = pyodbc.connect(conn_str)
#cnxn = pyodbc.connect('DRIVER={Devart ODBC Driver for SQL Server};Server=myserver;Database=mydatabase;Port=myport;User ID=myuserid;Password=mypassword')
    cursor = cnxn.cursor()	
    return cursor
def execute_query_function(query):
    cursor=setup_database()
    
    cursor.execute(query)
    results=[]
    # Get column names from cursor description
    columns = [column[0] for column in cursor.description]
    print("Column names:", columns)   
    row = cursor.fetchone()
    while row:
        row_as_list = list(row)  # or tuple(row)
        results.append(row_as_list)      
        row = cursor.fetchone()    
    cursor.close()

    return columns,results   

   
class retrive_similiar_docs :
 
    def __init__(self,query = " ", retrive_fields = ["content", "source"],
                      ):
        if query:
            self.query = query
 
        self.search_client = SearchClient(AZURE_COGNITIVE_SEARCH_ENDPOINT, index_to_use, azure_credential)
        self.retrive_fields = retrive_fields
   
    def text_search(self,top = 2):
        results = self.search_client.search(search_text= self.query,
                                select=self.retrive_fields,top=top)
       
        return results
       
 
    def pure_vector_search(self, k = 2, vector_field = 'contentVector',query_embedding = []):
 
        vector_query = RawVectorQuery(vector=query_embedding, k=k, fields=vector_field)
 
        results = self.search_client.search( search_text=None,  vector_queries= [vector_query],
                                            select=self.retrive_fields)
 
        return results
       
    # def hybrid_search(self, top=2, k=2, vector_field="contentVector", query_embedding=[]):
    #     vector_query = RawVectorQuery(vector=query_embedding, k=k, fields=vector_field)
    #     role_filter = f"read_access_entity/any(r: r eq '{ROLE}')"
    #     print("Using Role filter:", role_filter)
       
    #     results = self.search_client.search(
    #         search_text=self.query,
    #         vector_queries=[vector_query],
    #         select=self.retrive_fields,
    #         top=top,
    #         filter=role_filter
    #     )
       
    #     return results
 
    def hybrid_search(self, top=2, k=2, vector_field="contentVector", query_embedding=[]):
        vector_query = RawVectorQuery(vector=query_embedding, k=k, fields=vector_field)
               
        results = self.search_client.search(
            search_text=self.query,
            vector_queries=[vector_query],
            select=self.retrive_fields,
            top=top,
        )
       
        return results
 
 
 
import time
start = time.time()
 
 
def get_similiar_content(user_query = " ",
                      search_type = "hybrid",top = 2, k =2):
 
    #print("Generating query for embedding...")
    #embedding_query = get_query_for_embedding(user_query=user_query)
    retrive_docs = retrive_similiar_docs(query = user_query)
 
    if search_type == "text":
        start = time.time()
        r = retrive_docs.text_search(top =top)
 
        sources = []
        similiar_doc = []
        for doc in r:
            similiar_doc.append(doc["source"] + ": " + doc["content"].replace("\n", "").replace("\r", ""))
            sources.append(doc["source"])
        similiar_docs = "\n".join(similiar_doc)
        print("Retrived similiar documents with text search in :", time.time()-start,'seconds.')
        #print("Retrived Docs are :",sources,"\n")
 
    if search_type == "contentVector":
        start = time.time()
        vector_of_search_query = generate_embeddings_azure_openai(user_query)
        print("Generated embedding for search query in :", time.time()-start,'seconds.')
 
        start = time.time()
        r = retrive_docs.pure_vector_search(k=k, query_embedding = vector_of_search_query)
 
        sources = []
        similiar_doc = []
        for doc in r:
            similiar_doc.append(doc["source"] + ": " + doc["content"].replace("\n", "").replace("\r", ""))
            sources.append(doc["source"])
        similiar_docs = "\n".join(similiar_doc)
        print("Retrived similiar documents with text search in :", time.time()-start,'seconds.')
       # print("Retrived Docs are :",sources,"\n")
 
 
    if search_type == "hybrid":
        start = time.time()
        vector_of_search_query = generate_embeddings_azure_openai(user_query)
        print("Generated embedding for search query in :", time.time()-start,'seconds.')
 
        start = time.time()
        r = retrive_docs.hybrid_search(top = top, k=k, query_embedding = vector_of_search_query)
 
        sources = []
        similiar_doc = []
        for doc in r:
            similiar_doc.append(doc["source"] + ": " + doc["content"].replace("\n", "").replace("\r", ""))
            sources.append(doc["source"])
        similiar_docs = "\n".join(similiar_doc)
        print("*"*100)
        print("Retrived similiar documents with text search in :", time.time()-start,'seconds.')
        #print("similiar_doc :", similiar_doc)
        #print("Retrived Docs are :",sources,"\n")
        #print("similiar_doc :", similiar_doc)
        #print("*"*100)
    return similiar_docs
   
 
def stream_response(stream_object):
    full_response = " "
    for chunk in stream_object:
        if len(chunk.choices) >0:
            if str(chunk.choices[0].delta.content) != "None":
                full_response += chunk.choices[0].delta.content
                print(chunk.choices[0].delta.content,end = '')
    return full_response
 
 
def generate_response_without_memory(user_query = " ",stream = True,max_tokens = 512,model = " "):
 
    similiar_docs = get_similiar_content(user_query = user_query)
    user_content = user_query + " \nSOURCES:\n" + similiar_docs
    chat_conversations = [{"role" : "system", "content" : system_message}]
    chat_conversations.append({"role": "user", "content": user_content })
    response = call_gpt_model(messages = chat_conversations,stream = stream,max_tokens=max_tokens)
    #response = stream_response(response)
    return response
 
 
system_message = pmt
chat_conversations_global_message = [{"role" : "system", "content" : system_message}]
 
 
def generate_response_with_memory(user_query = " ",keep_messages = 10,new_conversation = False,model=model_to_use,stream=False):
 
    #global chat_conversations_to_send
 
#    if new_conversation:
#        st.session_state.messages = []
 
   
    #print(CHAT_CONVERSATION_TO_SEND)
    #print(chat_conversations_to_send)
 
    query_for_retriver = generate_query_for_retriver(user_query=user_query,messages = st.session_state.messages[-keep_messages:])
    similiar_docs = get_similiar_content(query_for_retriver)
    #print("Query for Retriver :",query_for_retriver)
    similiar_docs = get_similiar_content(query_for_retriver)
    user_content = user_query + " \nSOURCES:\n" + similiar_docs
 
    chat_conversations_to_send = chat_conversations_global_message + st.session_state.messages[-keep_messages:] + [{"role":"user","content" : user_content}]
   
    response_from_model = call_gpt_model(messages = chat_conversations_to_send)
    #print("Response_from_model :",response_from_model)
    #chat_conversations_to_send = chat_conversations_to_send[1:]
    #st.session_state.messages[-1] = {"role": "user", "content": user_query}
    #st.session_state.messages.append({"role": "assistant", "content": response_from_model})
    #print("*"*100)
    #print("chat_conversations_to_send :", chat_conversations_to_send)
    #print("*"*100)
 
    return response_from_model
 
 
 
def make_dataframe(col,data):
    # Convert to DataFrame
    df = pd.DataFrame(data, columns=col)
    return df
 
def display_data_dataframe(col,data):
    # Convert to DataFrame
    df = pd.DataFrame(data, columns=col)

    st.dataframe(df, use_container_width=True)
    return df
   
def get_follow_up_questions(response):
    # Split the response on "Follow-up Questions:"
    parts = response.split("Follow-up Questions:")
   
    # If the split was successful and there are follow-up questions
    if len(parts) > 1:
        # Get the part after "Follow-up Questions:"
        follow_up_questions_text = parts[1].strip()
       
        # Split the follow-up questions by newline and remove any leading/trailing whitespace
        follow_up_questions = [question.strip() for question in follow_up_questions_text.split('\n') if question.strip()]
    else:
        # If no follow-up questions were found, return an empty list
        follow_up_questions = []
   
    print(follow_up_questions)
    return follow_up_questions
 
def handle_sql_error(error):
   
    # Extract the error code and message
    error_code = error.args[0]
    print("This is error code",error_code)
    error_message = str(error)
   
    # Comprehensive error handling dictionary
    error_translations = {
        # Connection Errors
        "08001": "Unable to establish a connection to the database. Please check your connection settings.",
        "08002": "Connection is already in use or has been closed.",
        "08003": "Connection does not exist.",
        "08004": "Server rejected the connection attempt.",
        "08007": "Connection failure during transaction processing.",
       
        # Authentication Errors
        "28000": "Invalid authorization specification. Please verify your username and password.",
       
        # SQL Statement Errors
        "42000": "Syntax error or access rule violation in SQL statement.",
        "42S02": "Base table or view not found.",
        "42S01": "Base table already exists.",
       
        # Constraint Violation Errors
        "23000": "Integrity constraint violation. This might be due to duplicate entries or foreign key constraints.",
        "23001": "Restrict violation.",
        "23505": "Unique constraint violation.",
       
        # Data Type and Conversion Errors
        "22003": "Numeric value out of range.",
        "22007": "Invalid datetime format.",
        "22008": "Datetime field overflow.",
       
        # Resource Errors
        "53000": "Insufficient resources to complete the operation.",
        "53100": "Disk full, unable to extend database files.",
       
        # Permission Errors
        "42501": "Insufficient database permissions.",
       
        # Generic Catch-all
        "default": "An unexpected database error occurred."
    }
 
    friendly_message = error_translations.get(str(error_code),
        f"Database error (Code: {error_code}). {error_translations['default']}")
   
    # Additional context from the original error message
    full_message = f"{friendly_message}"
    print(error_message)
    return full_message              
    
 
#Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []
    st.session_state.displaymessages=[]
else:
    # Display chat messages from history on app rerun
    for displaymessages in st.session_state.displaymessages:
        if displaymessages["role"] == "assistant":
            avatar = "🤖"
        else:
            avatar = "🧑‍💻"
        with st.chat_message(displaymessages["role"],avatar = avatar ):
            # if message["role"] == "assistant":
            #     full_table=message["content"]
            #     table = ast.literal_eval(full_table)  
            #     col=table[0]
            #     rows=table[1:]
            #     display_data_dataframe(col,rows)
            # else:
            #     st.markdown(message["content"])
            if displaymessages["role"] == "assistant":
                display_data_dataframe(displaymessages["content1"],displaymessages["content2"])
                #st.markdown(message["content"])
            else:
                st.markdown(displaymessages["content"])    
 

 
# User input
if prompt := st.chat_input("Ask about database..."):
    # Display user message in chat message container
    st.chat_message("user",avatar = "🧑‍💻").markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.session_state.displaymessages.append({"role": "user", "content": prompt})

    user_query = prompt + dropdown_3_prompt
    if use_memory:
        response = generate_response_with_memory(user_query= user_query,stream=True)
    else :
        response = generate_response_without_memory(user_query= user_query,stream=True)
 
    with st.chat_message("assistant",avatar = "🤖"):
        message_placeholder = st.empty()
        dataframe_placeholder = st.empty()
        full_response = " "
        # Simulate stream of response with milliseconds delay
        for chunk in response:
            if len(chunk.choices) >0:
                if str(chunk.choices[0].delta.content) != "None":
                    full_response += chunk.choices[0].delta.content
                    #message_placeholder.markdown(full_response + "▌")
        full_response = full_response.replace("$", "\$")
        print("GENERATED QUERY IS",full_response)
        response_table=[]
        rows=[]
        col=[]
    try:
        col,rows=execute_query_function(full_response)
        
        # response_table.append(col)
        # response_table.append(rows)
        response_table.append(full_response)
        print("This is inside try block",response_table)
        df=display_data_dataframe(col,rows)
        my_report = sv.analyze([df, "Train"])
        my_report.show_html('Report.html')
        #profile = ProfileReport(df)
        #profile
        # if st.button("Open D-Tale"):  
        #     d = dtale.show(df)  
        #     d_url = d._main_url()  
        #     st.write("D-Tale is now running. Open the web browser to view it.")  
        #     st.markdown(f"[Open D-Tale]({d_url})", unsafe_allow_html=True)  
    except pyodbc.Error as e:
 
        # if e.args:
        # # Convert to string to ensure consistent comparison
        #     error_code = str(e.args[0]).lower()
           
        #     # Specific error codes for retry
        #     retry_error_codes = ['42000', '42s02']
           
        #     if error_code in retry_error_codes:
               
        #             retry_pmt="Generated Query : "+str(response_table[0])+str(e)+" This is the error in your generated query refer the database schem again and regenerate the sql query only according to schema."
 
        #             response = generate_response_without_memory(user_query= retry_pmt,stream=True)
 
        #             response_table.append(response)
 
                   
        #             st.session_state.messages.append({"role": "assistant", "content":str(response_table)})
        #             st.session_state.displaymessages.append({"role": "assistant", "content1":col,"content2":rows})
        #             print(f"API call made due to SQL error {error_code}")
                   
                   
               
                   
        # # No specific error codes matched or no args
               
        #     else:
        err_msg=handle_sql_error(e)
        st.write(err_msg)
 
        print("An exception occurred:", e)
 
        print("This is user query",user_query)
 
        # print("This is inside exception block",response_table)    
 
    st.session_state.messages.append({"role": "assistant", "content":str(response_table)})
    st.session_state.displaymessages.append({"role": "assistant", "content1":col,"content2":rows})
        
    #     col,rows=execute_query_function(full_response)
       
    #     #print(response_table)
    # # agar isko with k andar likh rahe h to resonse ki copy ban rahi h next query m   
    #     display_data_dataframe(col,rows)
   
    # st.session_state.messages.append({"role": "assistant", "content":str(full_response)})
    # st.session_state.displaymessages.append({"role": "assistant", "content1":col,"content2":rows})



#  with st.chat_message("assistant",avatar = "🤖"):
#         message_placeholder = st.empty()
#         full_response = " "
#         # Simulate stream of response with milliseconds delay
#         for chunk in response:
#             if len(chunk.choices) >0:
#                 if str(chunk.choices[0].delta.content) != "None":
#                     full_response += chunk.choices[0].delta.content
#                     #message_placeholder.markdown(full_response + "▌")
#         full_response = full_response.replace("$", "\$")
#         print("Junaid Full Resp",full_response)
#         response_table=[]
#         rows=[]
#         col=[]
#     try:
#         col,rows=execute_query_function(full_response)
     
#         # response_table.append(col)
#         # response_table.append(rows)
#         response_table.append(full_response)
#         print("This is inside try block",response_table)
#         display_data_dataframe(col,rows)
#     except pyodbc.Error as e:
 
#         if e.args:
#         # Convert to string to ensure consistent comparison
#             error_code = str(e.args[0]).lower()
           
#             # Specific error codes for retry
#             retry_error_codes = ['42000', '42s02']
           
#             if error_code in retry_error_codes:
               
#                     retry_pmt="Generated Query : "+str(response_table[0])+str(e)+" This is the error in your generated query refer the database schem again and regenerate the sql query only according to schema."
 
#                     response = generate_response_without_memory(user_query= retry_pmt,stream=True)
 
#                     response_table.append(response)
 
                   
#                     st.session_state.messages.append({"role": "assistant", "content":str(response_table)})
#                     st.session_state.displaymessages.append({"role": "assistant", "content1":col,"content2":rows})
#                     print(f"API call made due to SQL error {error_code}")
                   
                   
               
                   
#         # No specific error codes matched or no args
               
#             else:
#                 err_msg=handle_sql_error(e)
#                 st.write(err_msg);
 
#         print("An exception occurred:", e)
 
#         print("This is user query",user_query)
 
#         print("This is inside exception block",response_table)    
 

    