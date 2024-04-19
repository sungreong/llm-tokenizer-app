import transformers
from dotenv import load_dotenv
from transformers import AutoTokenizer
import streamlit as st

st.set_page_config(layout="wide")

from huggingface_hub import login
import os, sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))
from utils.text_history import CustomTextHistory as TextHistory
import streamlit.components.v1 as components
import pandas as pd


@st.cache_resource(hash_funcs={AutoTokenizer: id})
def llm_tokenizer_model(tokenizer_selected_list):
    tokenizer_dict = {}
    for tokenizer_name in tokenizer_selected_list:
        try:
            tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, token=st.session_state.get("hf_token", ""))
            # st.write("Tokenizer Loaded: {}".format(tokenizer_name))
        except Exception as e:
            st.error(f"[{tokenizer_name}] Tokenizer Load Error: {e}")
        else:
            tokenizer_dict[tokenizer_name] = tokenizer
    return tokenizer_dict


def llm_tokenizer_app():
    st.title("LLM Tokenizer")
    st.write("This app is for tokenizing text using LLMs.")
    st.write("You can also use any tokenizer from Hugging Face's model hub.")
    st.write("transformer version == {}".format(transformers.__version__))
    st.write("streamlit version == {}".format(st.__version__))
    load_dotenv()

    st.session_state["hf_token"] = st.text_input(
        "Hugging Face Token",
        value=st.session_state.get("hf_token", ""),
        placeholder="Hugging Face Token (hf-xxx)",
        type="password",
    )
    if not st.session_state["hf_token"]:
        st.write("Please enter Hugging Face token.")
        st.stop()
    else:
        login(token=st.session_state["hf_token"])
        st.write("Hugging Face Token Loaded.")

    tokenizer_names = st.text_input(
        "Tokenizer Name",
        value="",
        placeholder="Tokenizer Name (e.g., google/gemma-2b,google/gemma-7b)",
    )
    if not tokenizer_names:
        st.write("Please enter tokenizer name.")
        st.stop()

    with st.form(key="tokenizer_form"):
        tokenizer_name_list = tokenizer_names.split(",")
        tokenizer_selected_list = st.multiselect(
            "Select Tokenizer",
            tokenizer_name_list,
        )
        sample_text = st.text_area(
            "Sample Text",
            value="",
            height=300,
            max_chars=None,
            key=None,
            help="Sample Text",
        )
        cols_per_row = st.number_input(
            "Columns per Row",
            min_value=2,
            max_value=10,
            value=3,
            step=1,
            key="cols_per_row",
        )
        submit_button = st.form_submit_button(label="Load Tokenizer")

    if submit_button:
        if len(tokenizer_selected_list) > 0:

            with st.spinner("Loading Tokenizer..."):
                llm_tokenizer_dict = llm_tokenizer_model(tokenizer_selected_list)
            tokenizer_selected_possible_list = list(llm_tokenizer_dict.keys())
            n = len(tokenizer_selected_possible_list)  # Total number of items in the list
            if n == 0:
                st.write("No tokenizer loaded.")
                st.stop()
            if n < cols_per_row:
                cols_per_row = n
            rows = (n + cols_per_row - 1) // cols_per_row  # Calculate the total number of rows needed
            token_sample_info = {}
            for i in range(rows):
                cols = st.columns(cols_per_row)  # Create a new row of columns
                for j in range(cols_per_row):
                    idx = i * cols_per_row + j  # Calculate the index in the list
                    if idx < n:  # Check if the index is within the bounds of the list
                        with cols[j]:  # Use the column
                            st.subheader(tokenizer_selected_possible_list[idx])
                            token_sample_info[tokenizer_selected_possible_list[idx]] = {}
                            active_tokenizer = llm_tokenizer_dict[tokenizer_selected_possible_list[idx]]
                            result = active_tokenizer.tokenize(sample_text)
                            st.write(f"token vocab : {active_tokenizer.vocab_size}")
                            st.write(f"token type: {active_tokenizer.__class__.__name__}")

                            st.write(f"sample token count: {len(result)}")

                            st.text_area(
                                "Tokenized Text",
                                value=result,
                                height=300,
                                max_chars=None,
                                key=f"{tokenizer_selected_possible_list[idx]}_tokenized_text",
                            )
                            token_to_ids = active_tokenizer.convert_tokens_to_ids(result)
                            st.text_area(
                                "Token to IDs",
                                value=token_to_ids,
                                height=100,
                                max_chars=None,
                                key=f"{tokenizer_selected_possible_list[idx]}_token_to_ids",
                            )
                            token_sample_info[tokenizer_selected_possible_list[idx]]["unique_token_count"] = len(
                                set(token_to_ids)
                            )
                            token_sample_info[tokenizer_selected_possible_list[idx]]["token_count"] = len(token_to_ids)
                            st.write(f"unique token count: {len(set(token_to_ids))}")
                            st.write("Special Tokens")
                            st.json(active_tokenizer.special_tokens_map)
                            st.write("Tokenizer Chat Template")
                            st.markdown(active_tokenizer.chat_template)
                            import torch

                            token_to_ids = torch.LongTensor([token_to_ids])
                            text_history = [
                                TextHistory(q, qt, system=True) for q, qt in zip(sample_text, token_to_ids)
                            ][0]
                            text_history_html = text_history.show_tokens_detail(
                                tokenizer=active_tokenizer,
                                show_legend=False,
                                to_html=True,
                            )
                            components.html(
                                text_history_html,
                                scrolling=True,
                                height=500,
                            )
            else:
                import pandas as pd

                token_count_df = pd.DataFrame(token_sample_info).T
                _, center_col, _ = st.columns([3, 6, 3])
                with center_col:
                    st.dataframe(token_count_df, width=1000)
        else:
            st.write("Please select tokenizer name.")
            st.stop()


if __name__ == "__main__":
    llm_tokenizer_app()
