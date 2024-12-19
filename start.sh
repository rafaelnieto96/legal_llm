python -m venv venv

venv\Scripts\activate

pip install -r requirements.txt

python src/app.py





python -m src.training.train

streamlit run src/app.py