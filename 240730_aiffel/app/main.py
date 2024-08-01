import threading
import subprocess


def run_streamlit():
    subprocess.run(["streamlit", "run", "streamlit.py"])


def run_fastapi():
    subprocess.run(["uvicorn", "api:app", "--reload", "--host=127.0.0.1"])


if __name__ == "__main__":
    streamlit_thread = threading.Thread(target=run_streamlit)
    streamlit_thread.start()

    fastapi_thread = threading.Thread(target=run_fastapi)
    fastapi_thread.start()

    # 모두 종료될 때까지 대기
    streamlit_thread.join()
    fastapi_thread.join()