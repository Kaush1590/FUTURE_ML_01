import streamlit as st

# Application navigation and page routing
mainpage = st.Page("mainpage.py", title = "Sales Forecast Dashboard", icon = "📈")
dataset = st.Page("dataset_info.py", title = "Dataset Overview", icon = "📋")
model_comparison = st.Page("comparison.py", title = "Model Comparison", icon = "💻")
pg = st.navigation(pages = [mainpage, dataset, model_comparison])
pg.run()