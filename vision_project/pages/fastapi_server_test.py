import streamlit as st  
import requests

st.title("/detect_image 요청")

st.markdown("## 목표")
with st.container(border=True):
    st.markdown("""
    Streamlit의 파일 업로더로 이미지를 입력하면, <br>
    그 이미지를 FastAPI(`/detect_image`)에 요청해서 Object Detection 정보를 응답받는다.
    """, unsafe_allow_html=True)
    st.markdown("* URL: `http://localhost:8080/detect_image`")
    st.markdown("* files: `{ 'file' :  이미지에 대한 정보}`")

left, right = st.columns(2)

with left:
    #############################################
    # 파일 업로드 UI
    #############################################
    st.markdown("## 파일 업로드")
    uploaded_file = st.file_uploader(
        label="이미지를 업로드해주세요",
        type=["jpg", "jpeg", "png"]
    )
    #############################################
    # 사용자가 파일 업로드 할 경우
    #############################################
    if uploaded_file:
        # 이미지 정보 출력하기 
        st.markdown(f"* 이미지 이름(uploaded_file.name): {uploaded_file.name}")
        st.markdown(f"* 이미지 크기(uploaded_file.size): {uploaded_file.size}")
        st.markdown(f"* 이미지 타입(uploaded_file.type): {uploaded_file.type}")

        # 이미지 데이터 
        image_bytes = uploaded_file.getvalue()
        st.markdown(f"uploaded_file.getvalue(): `{image_bytes[:10]}`...로 시작하는 바이너리 데이터")

        # 이미지 출력하기 
        with st.expander("이미지 보기", expanded=False):
            st.image(image_bytes)

        #############################################
        # FastAPI 요청보내기
        #############################################
        st.markdown("## FastAPI 요청보내기")
        st.markdown("* URL: `http://localhost:8080/detect_image`")
        st.markdown("* files: `{ 'file' : uploaded_file.getvalue() }`")
        st.code("""
    url = "http://localhost:8080/detect_image"
    files = { "file" : uploaded_file.getvalue() }
    response = requests.post(url, files=files)
                """)
        
        url = "http://localhost:8080/detect_image"
        files = { "file" : uploaded_file.getvalue() }
        response = requests.post(url, files=files)

        if response.status_code == 200:
            with st.container(border=True):
                st.write(response.json())

with right:
    st.code('''
#############################################
# 파일 업로드 UI
#############################################
st.markdown("## 파일 업로드")
uploaded_file = st.file_uploader(
    label="이미지를 업로드해주세요",
    type=["jpg", "jpeg", "png"]
)
#############################################
# 사용자가 파일 업로드 할 경우
#############################################
if uploaded_file:
    # 이미지 정보 출력하기 
    st.markdown(f"* 이미지 이름(uploaded_file.name): {uploaded_file.name}")
    st.markdown(f"* 이미지 크기(uploaded_file.size): {uploaded_file.size}")
    st.markdown(f"* 이미지 타입(uploaded_file.type): {uploaded_file.type}")

    # 이미지 데이터 
    image_bytes = uploaded_file.getvalue()
    st.markdown(f"uploaded_file.getvalue(): `{image_bytes[:10]}`...로 시작하는 바이너리 데이터")

    # 이미지 출력하기 
    with st.expander("이미지 보기", expanded=False):
        st.image(image_bytes)
        
    #############################################
    # FastAPI 요청보내기
    #############################################
    st.markdown("## FastAPI 요청보내기")
    st.markdown("* URL: `http://localhost:8080/detect_image`")
    st.markdown("* files: `{ 'file' : uploaded_file.getvalue() }`")
    st.code("""
url = "http://localhost:8080/detect_image"
files = { "file" : uploaded_file.getvalue() }
response = requests.post(url, files=files)
            """)
    
    url = "http://localhost:8080/detect_image"
    files = { "file" : uploaded_file.getvalue() }
    response = requests.post(url, files=files)

    if response.status_code == 200:
        with st.container(border=True):
            st.write(response.json())
'''
)





    
