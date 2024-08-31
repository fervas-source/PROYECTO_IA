import time

from openai import BaseModel, OpenAI
import pandas as pd
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from dotenv import load_dotenv
import os
from langchain_core.pydantic_v1 import Field
from langchain_core.messages import (
    BaseMessage,
    HumanMessage,
    SystemMessage,
)
import operator
import textwrap
from typing import Annotated, Dict, List, Optional, Sequence, TypedDict, Literal
from langgraph.graph import END, StateGraph
from langgraph.prebuilt import ToolNode
from langchain_core.tools import tool
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate
from langchain_core.pydantic_v1 import Field
from langchain_core.output_parsers import StrOutputParser
from langgraph.checkpoint.memory import MemorySaver

#from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import CharacterTextSplitter
from langchain.docstore.document import Document
import pdfplumber  
from langchain.output_parsers import PydanticOutputParser
from langchain_core.exceptions import OutputParserException


#from langchain_unstructured import UnstructuredLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
#from langchain_community.vectorstores import Chroma
#from langchain.retrievers import MultiVectorRetriever

from langchain_core.agents import AgentAction, AgentFinish



load_dotenv()

langchain_api_key = os.getenv('LANGCHAIN_API_KEY')
openai_api_key = os.getenv('OPENAI_API_KEY')
userdata = {
    'LANGCHAIN_API_KEY': langchain_api_key,
    'OPENAI_API_KEY': openai_api_key
}

os.environ["LANGCHAIN_API_KEY"] = langchain_api_key
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "DiplomadoProyFinal"
os.environ["LANGCHAIN_SESSION"] = "1"
os.environ['OPENAI_API_KEY'] = openai_api_key
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

openai = OpenAI()

drugfile = pd.read_csv('DrugData.csv')

#Definicion Clase Medicamento

from typing import Optional
from pydantic import BaseModel, Field

class Medication(BaseModel):
    id: Optional[int] = Field(None, description="Unique identifier for the drug")
    drug_name: Optional[str] = Field(None, description="Brand name of the drug")
    generic_name: Optional[str] = Field(None, description="Generic name of the drug")
    drug_class: Optional[str] = Field(None, description="Class or category of the drug")
    indications: Optional[str] = Field(None, description="Approved uses or indications for the drug")
    dosage_form: Optional[str] = Field(None, description="Physical form of the drug (e.g., tablet, capsule, injection)")
    strength: Optional[str] = Field(None, description="Strength or concentration of the active ingredient")
    route_of_administration: Optional[str] = Field(None, description="How the drug is administered (e.g., oral, intravenous)")
    mechanism_of_action: Optional[str] = Field(None, description="How the drug works or its mode of action")
    side_effects: Optional[str] = Field(None, description="Potential side effects of the drug")
    contraindications: Optional[str] = Field(None, description="Situations or conditions where the drug should not be used")
    interactions: Optional[str] = Field(None, description="Potential interactions with other drugs or substances")
    warnings_and_precautions: Optional[str] = Field(None, description="Important warnings and precautions for using the drug")
    pregnancy_category: Optional[str] = Field(None, description="Category indicating the potential risk during pregnancy")
    storage_conditions: Optional[str] = Field(None, description="Recommended storage conditions for the drug")
    manufacturer: Optional[str] = Field(None, description="Company that manufactures the drug")
    approval_date: Optional[str] = Field(None, description="Date the drug was approved for use")
    availability: Optional[str] = Field(None, description="Information about the availability of the drug")
    ndc: Optional[str] = Field(None, description="National Drug Code (NDC) for the drug")
    price: Optional[float] = Field(None, description="Price or cost of the drug")

class MedicationDictionary(BaseModel):
    medications: Dict[str, Medication] = {}

class MedicationList(BaseModel):
    medications: List[Medication] = []

# State Initialization
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]
    medication_info: list[Medication]
    prescription_details: str
    sender: str
    final_reply: str
    nearby_pharmacies: str
    location: str


class Medicamento_Protegido(BaseModel):
    principio_activo: str = Field(alias="Principio activo")
    registro_sanitario: str = Field(alias="Registro sanitario")
    titular_registro_sanitario: str = Field(alias="Titular Reg. Sanitario")
    especialidad_farmaceutica: str = Field(alias="Especialidad farmacéutica")
    forma_farmaceutica: str = Field(alias="Forma farmacéutica")
    dosis: str = Field(alias="Dosis")
    presentacion_por_envase: str = Field(alias="Presentación x envase")
    estupefaciente_o_psicotropico: bool = Field(alias="Estupefaciente o Psicotrópico")

# define a function to transform intermediate_steps from list
# of AgentAction to scratchpad string
def create_scratchpad(intermediate_steps: list[AgentAction]):
    research_steps = []
    for i, action in enumerate(intermediate_steps):
        if action.log != "TBD":
            # this was the ToolExecution
            research_steps.append(
                f"Tool: {action.tool}, input: {action.tool_input}\n"
                f"Output: {action.log}"
            )
    return "\n---\n".join(research_steps)

def generate_csv_vector_store(file_path, api_key):
    """
    Generate a vector store from a CSV file.

    Args:
        file_path (str): Path to the CSV file.
        api_key (str): OpenAI API key.

    Returns:
        FAISS: A FAISS vector store instance.
    """
    # Read the CSV file
    df = pd.read_csv(file_path)

    # Initialize an empty list to store the texts
    texts = []

    # Iterate over each row in the DataFrame
    for _, row in df.iterrows():
        # Concatenate the column names and values into a single string
        row_text = ' '.join([f"{col}: {str(value)}" for col, value in zip(df.columns, row.values) if pd.notnull(value)])
        texts.append(row_text)

    # Create a TextSplitter instance
    text_splitter = CharacterTextSplitter()

    # Split each text into chunks
    docs = []
    for text in texts:
        doc_chunks = text_splitter.split_text(text)
        docs.extend([Document(page_content=chunk) for chunk in doc_chunks])

    # Generate embeddings
    embeddings = OpenAIEmbeddings(api_key=api_key)

    # Create the vector store
    vectorstore = FAISS.from_documents(docs, embeddings)
    return vectorstore

def generate_pdf_vector_store(pdf_path):
    """
    Generate a vector store from a PDF file.

    Args:
        pdf_path (str): Path to the PDF file.

    Returns:
        FAISS: A FAISS vector store instance.
    """
    
    # Set up parameters
    chunk_size = 1000
    chunk_overlap = 200
    embedding_model = "text-embedding-ada-002"

    # Extract text from PDF using pdfplumber
    with pdfplumber.open(pdf_path) as pdf:
        text = ""
        for page in pdf.pages:
            text += page.extract_text() + "\n"

    # Create a single Document object
    doc = Document(page_content=text, metadata={"source": pdf_path})

    # Create text splitter with overlapping chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len
    )

    # Split the document into chunks
    chunks = text_splitter.split_documents([doc])

    # Initialize the embedding model
    #embeddings = OpenAIEmbeddings(model=embedding_model)

    # Generate embeddings
    embeddings = OpenAIEmbeddings(api_key= openai_api_key,model=embedding_model)

    # Create the FAISS vector store
    vectorstore = FAISS.from_documents(chunks, embeddings)

    # Save the vector store
    #index_name = os.path.splitext(os.path.basename(pdf_path))[0]
    #save_path = f"faiss_index_{index_name}"
    #vectorstore.save_local(save_path)

    #print(f"Vector store saved to {save_path}")
    return vectorstore



def generate_pdf_vector_store2(pdf_path):
    # Initialize an empty lists
    tables_texts= []
    chunks = []
    

    # Open the PDF with pdfplumber
    with pdfplumber.open(pdf_path) as pdf:
        for page_number, page in enumerate(pdf.pages, start=1):
            # Extract text content
            text_content = page.extract_text()

            # Extract tables from the page
            tables = page.extract_tables()
            field_names = ["Principio activo", "Registro sanitario", "Titular Reg. Sanitario", "Especialidad farmacéutica", "Forma farmacéutica", "Dosis", "Presentación x envase", "Estupefaciente o Psicotrópico"]
            table_metadata = {"field_names": field_names}

            for table in tables:
                # Convert table to a string, handling None values
                table_text = "\n".join(["\t".join([str(cell) if cell is not None else "" for cell in row]) for row in table])
                tables_texts.append(table_text)
                #print(table_text)
                input()


            # Chunk the  content
            for table in tables_texts:
                chunks.extend([Document(page_content=table, metadata = table_metadata)])

    # Generate embeddings
    embeddings = OpenAIEmbeddings(api_key= openai_api_key)
    # Create the vector store
    vectorstore = FAISS.from_documents(chunks, embeddings)

    return vectorstore


@tool
def vademecum_retriever_tool(question, k=1):
    """
    Retrieve documents from the vector store based on a given question.
    It will return documents closest to the query but may not be exact
    K must be 1 (k=1)

    Args:
        question (str): The question to retrieve documents for.
        k (int, optional): The number of documents to retrieve. Defaults to 1.

    Returns:
        list: A list of retrieved documents, each medication includes the following information
        Drug ID,Drug Name,Generic Name,Drug Class,Indications,Dosage Form,Strength,Route of Administration,
        Mechanism of Action,Side Effects,Contraindications,Interactions,Warnings and Precautions,Pregnancy Category,
        Storage Conditions,Manufacturer,Approval Date,Availability,NDC,Price
        """
    # Generate Retrieval
    retriever = vademecum_vectorstore.as_retriever(search_kwargs={"k": k})

    # Retrieve documents
    docs_response = retriever.invoke(question)

    return docs_response

parser_str = StrOutputParser()
parser_medicationlist = PydanticOutputParser(pydantic_object=Medication)

@tool
def forbiden_meds_retriever_tool(medications: List[str], k=3):
    """
    Retrieve a list of forbidden medications for the given list of medications.
    It will return documents closest to the query using each medication as a key.
    If a medication is not in the list, then the medication is allowed.

    Args:
        medications (List[str]): A list of medications to check.
        k (int, optional): The number of documents to retrieve for each medication. Defaults to 3.

    Returns:
        Dict[str, Union[List[str], str]]: A dictionary with medication names as keys and either a list of 
        "NOT ALOWED" or "ALLOWED" as values.
    """
    results = {}
    retriever = restricted_med_vectorstore.as_retriever(search_kwargs={"k": k})

    for medication in medications:
        docs_response = retriever.invoke(medication)
        if docs_response and any(medication.lower() in doc.page_content.lower() for doc in docs_response):
            results[medication] = "NOT ALLOWED"
        else:
            results[medication] = "ALLOWED"

    return results

#Create data vectors
vademecum_vectorstore = generate_csv_vector_store('DrugData.csv', userdata.get('OPENAI_API_KEY'))
restricted_med_vectorstore = generate_pdf_vector_store('Medicamentos Registro Sanitario Vigente Condición de Venta Receta Médica Retenida.pdf')

#restricted_med_vectorstore = generate_pdf_vector_store('Medicamentos Registro Sanitario Vigente Condición de Venta Receta Médica Retenida.pdf')

#----------VADEMECUM QUERY AGENT------------#
vademecum_prompt_template = PromptTemplate(
    template = """ 
You are a specialized agent tasked with answering queries related to medical prescriptions or specific medicaments.
Respond based on the data in messages. 
When asked about an specific medical drug or medicine use the vademecum_retriever_tool function to search for relevant information.
If the tool does not bring information about the specific medication (names or drug dont match) then your reply with the following string 'No tengo informacion sobre el medicamento.'
Only use the vademecum_retriever_tool function once for each medication.
If you already have retrieved the information about the medication fron the vademecum_retriever_tool, dont call it again. 
Don´t look for information about other medications than the ones mentioned in the initial query.
If the information about the medication is not available, respond with 'No tengo informacion sobre el medicamento.'
Your responses must be strictly based on the information you get. 
Do not use any other external knowledge . 
Structure your response as a json as defined in the output schema
When delivering your final answer, your are done.

Remember, your role is to ensure accuracy and relevance based solely on the information retrieved through the vademecum_retriever_tool tool. 
If the information is not available or the query cannot be answered, clearly indicate this limitation.

# Data:
    Messages: {messages}

    """,
    input_variables=["messages"],
    partial_variables={"format_instructions": parser_medicationlist.get_format_instructions()}
)
tools_vademecum = [vademecum_retriever_tool]
llm_vademecum = ChatOpenAI(model="gpt-4o", temperature=0, api_key=userdata.get('OPENAI_API_KEY')).bind_tools(tools_vademecum)
chain_vadmecum = vademecum_prompt_template | llm_vademecum | parser_str

def vademecum_agent(state: AgentState)->AgentState:
    messages = state['messages']
    chain = vademecum_prompt_template | llm_vademecum #| parser_medicationlist
    response = chain.invoke(messages)
    return {"messages": [response], "sender": "vademecum_agent"}

#----------</VADEMECUM QUERY AGENT>------------#


#----------<REVIEWER AGENT>------------#
reviewer_prompt_template = PromptTemplate(
    template="""
# Reviewer Agent Prompt Template

You are the Reviewer Agent in a medication and pharmacy information system. Your role is crucial in ensuring that the information provided to users is accurate, appropriate, and compliant with regulations. Follow these instructions carefully:

## Input:
You have compiled responses from other agents in the system, which includes:
1. Medication information output from the Vademecum Agent

## Task:
Your task is to review and validate the compiled information before it is sent back to the user.

## Instructions:
1. Ensure that no medical prescriptions are provided under any circumstances. In thar case reply with only with "No puedo prescribir medicamentos" and you are done.
    - This means the system can not answer to question about what medication given a symptom
    - This means the system can not answer to question about what medication given a disease
    - In these cases do not include details of any medications in the final reply
2. Extract all unique medication names from the Vademecum Agent's response.
3. If you have already called the forbiden_meds_retriever_tool for the lsit of medications, dont use it again.
4. For each medication:
   - If it's marked as "ALLOWED", include the information from the Vademecum Agent in your reply.
   - If it's marked as "ANOT LLOWED", do not include the detailed information in your reply, and flag the medication as "Medicamento protegido".
5. If information about a medication is not available, respond with 'No tengo información sobre el medicamento [nombre del medicamento].'
7. Once you have review the medications once, you are done

Remember, your primary role is to ensure the safety and compliance of the information provided to users. When in doubt, err on the side of caution.

## Validation Checklist:
- [ ] No medical prescriptions are given
- [ ] Forbidden medications are properly flagged and their details are not disclosed
- [ ] Medication information (if present and allowed) is factual and non-prescriptive
- [ ] Pharmacy information (if present) is current and relevant
- [ ] The overall response addresses the user's query without overstepping boundaries

# Data:
    Output from vademecum agent: {responses}

""",
    input_variables=["responses"],
)
tools_reviewer = [forbiden_meds_retriever_tool]  # Add any necessary tools here
llm_reviewer = ChatOpenAI(model="gpt-4o", temperature=0, api_key=userdata.get('OPENAI_API_KEY')).bind_tools(tools_reviewer)

def reviewer_agent(state: AgentState)->AgentState:
    messages = state['messages']
    chain = reviewer_prompt_template | llm_reviewer
    response = chain.invoke(messages)
    return {"messages": [response], "sender": "reviewer_agent"}

#----------</REVIEWER AGENT>------------#




# Define the workflow
workflow = StateGraph(AgentState)
workflow.add_node("vademecum_agent", vademecum_agent)
workflow.add_node("reviewer_agent", reviewer_agent)
tools = tools_vademecum + tools_reviewer

workflow.add_node("tool_node", ToolNode(tools))

# Conditional edges
def should_continue(state: dict) -> Literal["tools", "__end__","vademecum_agent"]:
    messages = state['messages']
    last_message = messages[-1]
    print("last message router ")
    print(last_message.content)
    print(last_message.tool_calls)

# Instructions:
    if "No puedo prescribir medicamentos, por favor consulta a un médico" in last_message.content:
        return "__end__"
    if "FINAL RESPONSE" in last_message.content:
        return "__end__"
    if "RE EVALUAR" in last_message.content:
        return "vademecum_agent"
    if last_message.tool_calls:
        return "tools"
    return "__end__"

workflow.add_conditional_edges("vademecum_agent", should_continue, {"tools": "tool_node", "__end__": "reviewer_agent"})
workflow.add_conditional_edges("reviewer_agent", should_continue, {"tools": "tool_node", "__end__": END})
workflow.add_conditional_edges(
    "tool_node",

    lambda x: x["sender"],
    {
        "vademecum_agent": "vademecum_agent",
        "reviewer_agent": "reviewer_agent",
    },
)

workflow.set_entry_point("vademecum_agent")

# Checkpointer
checkpointer = MemorySaver()

# Compile the workflow
app = workflow.compile(checkpointer=checkpointer)

# Draw the graph
#from IPython.display import Image, display
#display(Image(app.get_graph(xray=True).draw_mermaid_png()))

# Pregunta a Responder
#question = "paracetamol"
question = "¿que es la aspirina, tafirol, amoxicilina, Alprazolam?"
#question = "¿que tomo para la fiebre?"
#question = input("Ingrese la pregunta: ")

# Define la consulta inicial y el mensaje del sistema
system_message = """
"""

start_time = time.time()
app.invoke(
    {"messages": [SystemMessage(content=system_message), HumanMessage(content=question)]},
    config={"configurable": {"thread_id": 7}}
)
# Ejecución del workflow
"""
for event in app.stream(
    {"messages":
      [#SystemMessage(content=system_message, ),
      HumanMessage(content=question)]},
    config={"configurable": {"thread_id": 7}}
):
    for k, v in event.items():
        if k != "__end__":
            if 'messages' in v:
                for message in v['messages']:
                  if "FINAL RESPONSE" in message.content:
                    # Wrap the text to a maximum width of 80 characters
                    wrapped_text = textwrap.fill(message.content, width=80)
                    print(wrapped_text)
"""                    
response_time = time.time()
print(f'Tiempo para generar la respuesta: {response_time - start_time} segundos')

