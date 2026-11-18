"""Интерфейс «список покупок по блюду»."""

from dotenv import load_dotenv

load_dotenv()

import streamlit as st

from llm import (
    FORMAT_OPTIONS,
    MSG_NETWORK,
    PEOPLE_OPTIONS,
    ShoppingListError,
    generate_shopping_list,
)

st.set_page_config(page_title="Список покупок по блюду")
st.title("Список покупок по блюду")
st.write("Напишите блюдо, выберите число гостей и формат ответа.")

if "shopping_list" not in st.session_state:
    st.session_state.shopping_list = ""
if "form_error" not in st.session_state:
    st.session_state.form_error = ""

with st.form("shopping_form", clear_on_submit=False):
    dish = st.text_area(
        "Что приготовить",
        height=140,
        placeholder="Например: паста карбонара",
    )
    people = st.selectbox("На сколько человек", PEOPLE_OPTIONS, index=1)
    response_format = st.selectbox("Формат", FORMAT_OPTIONS)
    submitted = st.form_submit_button("Сгенерировать список покупок")

if submitted:
    st.session_state.form_error = ""
    st.session_state.shopping_list = ""
    try:
        st.session_state.shopping_list = generate_shopping_list(
            dish,
            people,
            response_format,
        )
    except ShoppingListError as error:
        st.session_state.form_error = error.message
    except Exception:
        st.session_state.form_error = MSG_NETWORK

if st.session_state.form_error:
    st.error(st.session_state.form_error)

st.text_area(
    "Список покупок",
    value=st.session_state.shopping_list,
    height=320,
    disabled=True,
)
