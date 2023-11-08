# ChatBot with documents

- Provide a pdf to this chatbot and ask a question. It answers the question based on the pdf.

## Running the app - Method 1

- Intialize a virtual environment. run `python -m venv .venv`
- activate the virtual environment
    - Windows - run `.venv/Scripts/activate`
    - Linux - run `source .venv/bin/activate`
- Install dependencies from requirements.txt. run `pip install -r requirements.txt` (takes about ~6 minutes)
- Run the streamlit app. run `streamlit run app.py`
- Open `localhost:8501` in browser

## Running the app - Method 2

- Install docker from official website and make sure the docker is running
- run `cd other`
- run `docker-compose up` (takes about ~6 minutes)
- Open `localhost:8501` in browser
