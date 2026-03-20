@echo off

REM 🔽 프로젝트 루트 경로로 이동 (여기 수정!)
REM cd /d private_project\chatgpt-portable

REM 🔽 가상환경 활성화 (있으면)
if exist .venv\Scripts\activate (
    call .venv\Scripts\activate
)

REM 🔽 Streamlit 실행
streamlit run main.py

pause