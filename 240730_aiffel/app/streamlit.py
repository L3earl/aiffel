# front 디자인은 대신해주는 api - 주로 대쉬보드
# graio라는 api도 같은 역할 


import streamlit as st
import requests
import json

st.title("This is my first toy project")
st.subheader("modu caculator")

x = st.text_area("x의 숫자를 입력해주세요.")
operator = st.selectbox("부등호", ("+", "-", "/", "*"))
y = st.text_area("y의 숫자를 입력해주세요.")

info_dict = {"x": x, "y": y, "operator": operator}

if st.button("Caculate"):
    result = requests.post(
        url="http://127.0.0.1:8000/calculator",
        data=json.dumps(info_dict),
        verify=False,
    )
    st.subheader(f"result: {result.text}")