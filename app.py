import streamlit as st
from query import ask_question, process_uploaded_file

st.set_page_config(
    page_title="AlgoAsk - Trading Report Q&A",
    page_icon=":bar_chart:",
    layout="centered"
)

st.title("AlgoAsk: Trading Report Q&A Assistant")
st.markdown("Ask questions based on your uploaded trading report")

# Upload section
uploaded_file = st.file_uploader("Upload your trading report (CSV)", type=["csv"])

# Save and process uploaded file
if uploaded_file:
    file_path = f"data/{uploaded_file.name}"
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    st.success(f"File '{uploaded_file.name}' uploaded successfully!")

    # Process embeddings
    if st.button("Ingest File"):
        with st.spinner("Embedding and saving..."):
            process_uploaded_file(file_path)
        st.success("File embedded and ready for Q&A")

# Question input
question = st.text_input("Ask a question about your trading report:")
if question:
    with st.spinner("Thinking..."):
        answer = ask_question(question)
    st.markdown(f"**Answer:** {answer}")
    