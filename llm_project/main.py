# uv add openai python-dotenv streamlit
# uv add streamlit==1.49.1
# .env 파일 만들어서 OPENAI_API_KEY 추가해두기
# 서버 실행: streamlit run main.py
import streamlit as st 

pages = [
    st.Page(
        page="pages/chatbot_api.py",
        title="ChatBot",
        icon="😊",
        default=True
    ),
    st.Page(
        page="pages/chatbot_hw.py",
        title="츤데레 챗봇",
        icon="😊",
    ),
    st.Page(
        page="pages/chatbot_api_history.py",
        title="ChatBot History API",
        icon="😊",
    ),
    
    
]

nav = st.navigation(pages)
nav.run()