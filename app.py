import transformers
from dotenv import load_dotenv
from transformers import AutoTokenizer
import streamlit as st

from huggingface_hub import login
import os, sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))
from utils.text_history import CustomTextHistory as TextHistory
from utils.decoder import incremental_decode as TextDecoder
from utils.huggingface_utils import repo_exists
import streamlit.components.v1 as components
import pandas as pd


@st.cache_resource(hash_funcs={AutoTokenizer: id})
def llm_tokenizer_model(tokenizer_selected_list):
    tokenizer_dict = {}
    for tokenizer_name in tokenizer_selected_list:
        try:
            if tokenizer_name == "OpenAI/GPT3.5":
                tokenizer_name = "DWDMaiMai/tiktoken_cl100k_base"
            tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, token=st.session_state.get("hf_token", ""))
            # st.write("Tokenizer Loaded: {}".format(tokenizer_name))
        except Exception as e:
            st.error(f"[{tokenizer_name}] Tokenizer Load Error: {e}")
        else:
            if tokenizer_name == "DWDMaiMai/tiktoken_cl100k_base":
                tokenizer_name = "OpenAI/GPT3.5"

            tokenizer_dict[tokenizer_name] = tokenizer
    return tokenizer_dict


def set_large_label_font():
    style = """
    <style>
    div[class*="stTextArea"] p {
    font-size: 30px
    }

    div[class*="stTextInput"] p {
    font-size: 30px
    }

    div[class*="stNumberInput"] p {
    font-size: 30px
    }
    div[class*="stMultiSelect"] p {
    font-size: 30px
    }
    </style>
    """
    st.markdown(style, unsafe_allow_html=True)


def display_header():
    st.title("LLM Tokenizer")
    st.markdown(
        """
    This app is designed for tokenizing text using various Language Model tokenizers available on the Hugging Face model hub.
    """
    )


def display_info():
    st.info(
        """
    🔍 You can try out different tokenizers from the Hugging Face model hub to see how they tokenize different texts.
    """
    )
    st.link_button("Go to Hugging Face", "https://huggingface.co/")


def display_versions():
    st.subheader("Environment Versions")
    st.write(f"**Transformer version:** `{transformers.__version__}`")
    st.write(f"**Streamlit version:** `{st.__version__}`")


def create_footer():
    footer = """
    <style>
    .footer {
        position: fixed;
        left: 0;
        bottom: 0;
        width: 100%;
        background-color: #f1f1f1;
        color: black;
        text-align: center;
        padding: 10px;
        font-size: 16px;
    }
    </style>
    <div class="footer">
        <p>Developed with ❤️ by srlee | <a href="mailto:leesungreong@gmail.com">Contact</a> | <a href="https://data-newbie.tistory.com/">Website</a></p>
    </div>
    """
    st.markdown(footer, unsafe_allow_html=True)


def hide_github_corner():
    st.markdown(
        """
        <style>
        .css-1jc7ptx, .e1ewe7hr3, .viewerBadge_container__1QSob,
        .styles_viewerBadge__1yB5_, .viewerBadge_link__1S137,
        .viewerBadge_text__1JaDK {
            display: none;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


# CSS를 이용하여 마지막 stActionButton 숨기기
def hide_last_action_button():
    hide_button_style = """
    <style>
    div[data-testid="stToolbarActions"] .stActionButton:last-child {
        display: none !important;
    }
    </style>
    """
    st.markdown(hide_button_style, unsafe_allow_html=True)


def llm_tokenizer_app():
    st.set_page_config(page_title="LLM Tokenizer APP", layout="wide")

    # 스타일 적용 함수 호출
    hide_last_action_button()
    hide_github_corner()

    set_large_label_font()
    create_footer()
    display_header()
    display_info()
    display_versions()
    load_dotenv()
    if "hf_token" not in st.session_state:
        st.session_state["hf_token"] = os.getenv("hf_token", "")
    if "tokenizer_names" not in st.session_state:
        st.session_state["tokenizer_names"] = ""
    token_input = st.text_input(
        "Hugging Face Token",
        value=st.session_state.get("hf_token", ""),
        placeholder="Hugging Face Token (hf-xxx)",
        type="password",
    )
    st.session_state["hf_token"] = token_input

    if not st.session_state["hf_token"]:
        st.write("Please enter Hugging Face token.")
        st.stop()
    else:
        login(token=st.session_state["hf_token"])
        st.write("Hugging Face Token Loaded.")

    tokenizer_names = st.text_input(
        "Tokenizer Names(, separated)",
        value=st.session_state["tokenizer_names"],
        placeholder="Tokenizer Name (e.g., google/gemma-2b,google/gemma-7b)",
    )
    st.session_state["tokenizer_names"] = tokenizer_names
    if not tokenizer_names:
        st.info("Please enter tokenizer name.")
        st.stop()

    with st.form(key="tokenizer_form"):
        tokenizer_name_list = list(set(tokenizer_names.split(",")))
        # check tokenizer name and change tokenizer NAME List
        checked_tokenizer_name_list = []
        for idx, tokenizer_name in enumerate(tokenizer_name_list):
            if repo_exists(tokenizer_name):
                checked_tokenizer_name_list.append(tokenizer_name)
            else:
                st.warning(f"Tokenizer Name [{tokenizer_name}] is not exists.")
        else:
            st.session_state["tokenizer_names"] = ",".join(checked_tokenizer_name_list)
        checked_tokenizer_name_list = checked_tokenizer_name_list + ["OpenAI/GPT3.5"]
        tokenizer_selected_list = st.multiselect(
            "Select Tokenizer",
            checked_tokenizer_name_list,
        )
        sample_text = st.text_area(
            "Sample Text",
            value="",
            height=300,
            max_chars=None,
            key=None,
            help="Sample Text",
        )
        add_special_tokens = st.checkbox("Add Special Tokens", value=True)
        cols_per_row = st.number_input(
            "Columns per Row",
            min_value=2,
            max_value=10,
            value=3,
            step=1,
            key="cols_per_row",
        )
        submit_button = st.form_submit_button(label="Load Tokenizer", type="primary")
    if submit_button:
        if len(tokenizer_selected_list) > 0:

            with st.spinner("Loading Tokenizer..."):
                llm_tokenizer_dict = llm_tokenizer_model(tokenizer_selected_list)
            tokenizer_selected_possible_list = list(set(list(llm_tokenizer_dict.keys())))
            n = len(tokenizer_selected_possible_list)  # Total number of items in the list
            if n == 0:
                st.write("No tokenizer loaded.")
                st.stop()
            if n < cols_per_row:
                cols_per_row = n
            rows = (n + cols_per_row - 1) // cols_per_row  # Calculate the total number of rows needed
            token_sample_info = {}
            st.info("Sample Text Length: {}".format(len(sample_text)) if sample_text else "No Sample Text Provided.")
            for i in range(rows):
                cols = st.columns(cols_per_row)  # Create a new row of columns
                for j in range(cols_per_row):
                    idx = i * cols_per_row + j  # Calculate the index in the list
                    if idx < n:  # Check if the index is within the bounds of the list
                        with cols[j]:  # Use the column
                            st.subheader(tokenizer_selected_possible_list[idx])
                            token_sample_info[tokenizer_selected_possible_list[idx]] = {}
                            active_tokenizer = llm_tokenizer_dict[tokenizer_selected_possible_list[idx]]
                            token_to_ids = active_tokenizer.encode(sample_text, add_special_tokens=add_special_tokens)

                            # token_ids_to_text = TextDecoder(active_tokenizer, token_to_ids)
                            st.write(f"token vocab : {active_tokenizer.vocab_size}")
                            st.write(f"token type: {active_tokenizer.__class__.__name__}")
                            st.write(f"sample token count: {len(token_to_ids)}")
                            tabs = st.tabs(
                                [
                                    f"Token Viewer(Visible)",
                                    f"Token Viewer(Token)",
                                    "Tokenized Text",
                                    "Token to IDs",
                                    "Info",
                                ],
                            )
                            import torch

                            with tabs[0]:
                                text_html = """
                                <style>
                                .custom-text {
                                    font-size: 30px;
                                }
                                </style>
                                <p class='custom-text'>Token Viewer(Visible)</p>
                                """
                                st.markdown(text_html, unsafe_allow_html=True)
                                text_history = [
                                    TextHistory(q, qt, system=True)
                                    for q, qt in zip(sample_text, torch.LongTensor([token_to_ids]))
                                ][0]
                                text_history_html = text_history.show_tokens_detail(
                                    tokenizer=active_tokenizer,
                                    show_legend=False,
                                    to_html=True,
                                    use_incremental_decode=True,
                                )
                                components.html(
                                    text_history_html,
                                    scrolling=True,
                                    height=500,
                                )
                            with tabs[1]:
                                text_html = """
                                <style>
                                .custom-text {
                                    font-size: 30px;
                                }
                                </style>
                                <p class='custom-text'>Token Viewer(Token)</p>
                                """
                                st.markdown(text_html, unsafe_allow_html=True)
                                text_history = [
                                    TextHistory(q, qt, system=True)
                                    for q, qt in zip(sample_text, torch.LongTensor([token_to_ids]))
                                ][0]
                                text_history_html = text_history.show_tokens_detail(
                                    tokenizer=active_tokenizer,
                                    show_legend=False,
                                    to_html=True,
                                    use_incremental_decode=False,
                                )
                                components.html(
                                    text_history_html,
                                    scrolling=True,
                                    height=500,
                                )
                            with tabs[2]:
                                st.text_area(
                                    "Tokenized Text",
                                    value=active_tokenizer.convert_ids_to_tokens(token_to_ids),
                                    height=300,
                                    max_chars=None,
                                    key=f"{tokenizer_selected_possible_list[idx]}_tokenized_text",
                                )
                            with tabs[3]:
                                st.text_area(
                                    "Token to IDs",
                                    value=token_to_ids,
                                    height=100,
                                    max_chars=None,
                                    key=f"{tokenizer_selected_possible_list[idx]}_token_to_ids",
                                )
                            with tabs[4]:
                                token_sample_info[tokenizer_selected_possible_list[idx]]["unique_token_count"] = len(
                                    set(token_to_ids)
                                )
                                token_sample_info[tokenizer_selected_possible_list[idx]]["token_count"] = len(
                                    token_to_ids
                                )
                                st.write(
                                    f"unique token count: {len(set(token_to_ids))}, token count: {len(token_to_ids)}"
                                )
                                st.write("Special Tokens")
                                st.json(active_tokenizer.special_tokens_map, expanded=False)
                                st.write("Tokenizer Chat Template")
                                st.markdown(active_tokenizer.chat_template, unsafe_allow_html=True)

            else:
                import pandas as pd

                token_count_df = pd.DataFrame(token_sample_info).T
                _, center_col, _ = st.columns([1, 8, 1])
                with center_col:
                    st.dataframe(token_count_df, width=1000, use_container_width=True)
        else:
            st.info("Please select tokenizer name.")
            st.stop()


if __name__ == "__main__":
    llm_tokenizer_app()
