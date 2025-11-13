
import streamlit as st
import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer
import sqlglot

# --- Step 1: Model Loading ---
@st.cache_resource
def load_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_name = "mrm8488/t5-small-finetuned-wikiSQL"
    try:
        tokenizer = T5Tokenizer.from_pretrained(model_name)
        model = T5ForConditionalGeneration.from_pretrained(model_name)
        model.to(device)
        return tokenizer, model, device
    except Exception as e:
        st.error(f"Error loading model: {e}")
        st.stop()

# --- Step 2: SQL Generation Function ---
def generate_sql(schema, question, tokenizer, model, device):
    if not tokenizer or not model:
        return "Model not loaded.", ""
    prompt = f"question: {question} context: {schema}"
    st.write("---")
    st.write("**Model Input Prompt (for debugging):**")
    st.info(prompt)
    try:
        inputs = tokenizer(prompt, return_tensors="pt", max_length=512, truncation=True).to(device)
        output = model.generate(**inputs, max_length=512, num_beams=5, early_stopping=True)
        generated_sql = tokenizer.decode(output[0], skip_special_tokens=True)
        return generated_sql, prompt
    except Exception as e:
        st.error(f"Error during SQL generation: {e}")
        return None, prompt

# --- Step 3: SQL Validation with sqlglot ---
def validate_sql(sql_query):
    if not sql_query:
        return "No SQL query generated.", "neutral"
    try:
        sqlglot.parse_one(sql_query)
        return "✅ Valid SQL Syntax", "success"
    except sqlglot.errors.ParseError as e:
        return f"❌ Invalid SQL Syntax: {e}", "error"
    except Exception as e:
        st.error(f"An unexpected error occurred during validation: {e}")
        return f"❓ Validation Error", "warning"

# --- Helper Function to parse schema input ---
def parse_schema_text(schema_text):
    return " ".join(schema_text.strip().split('
'))

# --- Streamlit App UI ---
def main():
    st.set_page_config(layout="wide", page_title="Text-to-SQL Chatbot")
    st.title("🤖 Text-to-SQL Chatbot")
    st.markdown("""
    This app uses a light Hugging Face model (`mrm8488/t5-small-finetuned-wikiSQL`)
    to generate SQL queries from natural language.
    """)

    with st.spinner("Loading AI Model (this may take a moment)..."):
        tokenizer, model, device = load_model()
    st.success("Model loaded successfully!")
    st.write(f"Running on device: **{device}**")

    col1, col2 = st.columns(2)
    with col1:
        st.header("1. Define Your Schema")
        st.markdown("""
        Enter your schema below. Use one line per table.
        **Format:** `table_name ( col1, col2, col3 )`
        """)
        example_schema_text = "employees ( id, name, department_id, salary )
departments ( id, name, location )"
        schema_text = st.text_area("Database Schema", value=example_schema_text, height=150, help="Example: employees ( id, name, salary )")
        st.header("2. Ask a Question")
        question = st.text_input("Natural Language Question", value="How many employees are there in the company?")
        generate_button = st.button("🚀 Generate SQL")

    with col2:
        st.header("3. Generated Output")
        if generate_button:
            if not schema_text or not question:
                st.warning("Please provide both a schema and a question.")
            else:
                with st.spinner("Generating SQL..."):
                    schema_string = parse_schema_text(schema_text)
                    sql_query, prompt = generate_sql(schema_string, question, tokenizer, model, device)
                    if sql_query:
                        st.subheader("Generated SQL Query")
                        st.code(sql_query, language="sql")
                        st.subheader("SQL Syntax Validation (using sqlglot)")
                        validation_message, v_type = validate_sql(sql_query)
                        if v_type == "success":
                            st.success(validation_message)
                        elif v_type == "error":
                            st.error(validation_message)
                        else:
                            st.warning(validation_message)
                    else:
                        st.error("Could not generate SQL query.")
        else:
            st.info("Click the 'Generate SQL' button to see the results.")

if __name__ == "__main__":
    main()
