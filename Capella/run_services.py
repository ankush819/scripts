import subprocess
import sys
import time

def run_services():
    try:
        # Start FastAPI server
        api_process = subprocess.Popen(
            ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        print("Started FastAPI server on http://localhost:8000")
        
        # Wait a moment for API to start
        time.sleep(2)
        
        # Start Streamlit app
        streamlit_process = subprocess.Popen(
            ["streamlit", "run", "src/app.py", "--server.port", "8501"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        print("Started Streamlit app on http://localhost:8501")
        
        # Keep running until interrupted
        api_process.wait()
        streamlit_process.wait()
        
    except KeyboardInterrupt:
        print("\nShutting down services...")
        api_process.terminate()
        streamlit_process.terminate()
        sys.exit(0)

if __name__ == "__main__":
    run_services() 