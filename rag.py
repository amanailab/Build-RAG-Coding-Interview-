from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate


load_dotenv()  # Load environment variables from .env file

llm = ChatGroq(
    model = "openai/gpt-oss-20b",
    temperature = 0
)

loader = PyPDFLoader("data/Build_RAG_From_Scratch_Using_LangChain_AmanAI_Lab (1).pdf")

documents = loader.load()

splitter = RecursiveCharacterTextSplitter( chunk_size = 1000, chunk_overlap = 200)

chunks = splitter.split_documents(documents)

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

vectorstore = FAISS.from_documents(chunks, embeddings)


retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

prompt = ChatPromptTemplate.from_template("""
Answer the question using ONLY the provided context.
{context}
Question: {question}
""")

question = input("\n Enter your question: ")

docs = retriever.invoke(question)

context = "\n\n".join(doc.page_content for doc in docs)



response = llm.invoke(prompt.format(context=context, question=question))


print(response.content)