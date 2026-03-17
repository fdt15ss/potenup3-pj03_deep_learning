# test.py
import streamlit as st 

st.title("session state의 중요성")

st.markdown("## Session State 사용")

if "count" not in st.session_state:
    st.session_state["count"] = 0
    
st.markdown("## Session State")
st.write(st.session_state)

mybutton = st.button(
    label="버튼",
    key="mybutton"
)

if mybutton:
    st.session_state["count"] += 1
    st.markdown(f"Count: {st.session_state["count"]}")



st.markdown("## Session State 없이 구현")
# count 초기 설정
count = 0
st.markdown(f"count 초기 설정: {count}")

# button 
mybutton = st.button(
    label="버튼"
)

if mybutton:
    count +=1 
    st.markdown(f"count = {count}")