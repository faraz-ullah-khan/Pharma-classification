- program.py file trains the ML model and create a pkl file for it called as model.pkl. Run `python program1.py` to create the pkl file.
- app.py file is for running streamlit local server. This file uses the previously created pkl model file to predict the result. Run `streamlit run pharma.py` to host the local server.
  The model is more bayased towards no 'no disease'