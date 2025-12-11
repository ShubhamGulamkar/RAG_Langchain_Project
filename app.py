import streamlit as st
from src.search import RAGSearch
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize the RAGSearch system
rag_search = RAGSearch()

# Streamlit UI
st.title("Modular RAG Search System")
st.write("This is a Retrieval-Augmented Generation system that allows you to query documents and get answers from a large corpus.")

# Create an input box for user query
query = st.text_input("Enter your question:")

# Button to get the answer
if st.button("Get Answer"):
    if query:
        # Query the RAG system
        summary = rag_search.search_and_summarize(query, top_k=3)
        st.subheader("Answer:")
        st.write(summary)
    else:
        st.write("Please enter a question to get an answer.")



# from src.data_loader import load_all_documents
# from src.vectorstore import FaissVectorStore
# from src.search import RAGSearch

# # Example usage
# if __name__ == "__main__":
    
#     docs = load_all_documents("data")
#     store = FaissVectorStore("faiss_store")
#     #store.build_from_documents(docs)
#     store.load()
#     #print(store.query("What is attention mechanism?", top_k=3))
#     rag_search = RAGSearch()
#     query = "what is Modular RAG?"
#     summary = rag_search.search_and_summarize(query, top_k=3)
#     print("Summary:", summary)