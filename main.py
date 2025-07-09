import streamlit as st
st.set_page_config(layout="wide",page_title="Seleziona Lingua / Select Language")  # deve essere prima di tutto

import estrai_grafico
import plot_extraction

st.markdown(
    "<div style='text-align: center;'>"
    "<button onclick=\"window.location.reload();\">"
    "<img src='https://flagcdn.com/24x18/it.png'/> Italiano</button> &nbsp;"
    "<button onclick=\"window.location.reload();\">"
    "<img src='https://flagcdn.com/24x18/gb.png'/> English</button>"
    "</div>",
    unsafe_allow_html=True
)

lang = st.radio(
    "Scegli la lingua / Choose language",
    ("Italiano", "English"),
    index=0
)

if lang == "Italiano":
    estrai_grafico.run()
else:
    plot_extraction.run()
