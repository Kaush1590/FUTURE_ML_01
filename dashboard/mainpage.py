import io
import pathlib
import pickle
import plotly.graph_objects as go
import streamlit as st

# For loading application state
@st.cache_data
def load_state(path):
    with open(path, "rb") as f:
        return pickle.load(f)

# Provides download button for interactive plotly graphs
def download_button_graph(fig, filename, key):
    buffer = io.StringIO()
    fig.write_html(buffer)
    st.download_button(
        label = "Download interactive graph (HTML)",
        data = buffer.getvalue(),
        file_name = f"{filename}.html",
        mime = "text/html",
        key = key
    )

# Provides download button for dataframe
def download_button_dataframe(dataframe, label, filename_no_extension, key):
    st.download_button(
        label = f"Download {label}",
        data = dataframe.to_csv(index = False),
        file_name = f"{filename_no_extension}.csv",
        mime = "text/csv",
        key = key
    )

# Load application states
try:
    state_path = pathlib.Path(__file__).parent.parent / "store" / "state_dump.pkl"
    state_holder = load_state(state_path)
    
    # Load trained model predictions
    monthly_sales = state_holder["monthly_sales"]
    forecast_all = state_holder["forecast"]
    category_forecast = state_holder["category_forecast_df"]
    errors = state_holder["errors"]

except FileNotFoundError:
    st.error("Required files are not found. Please run the training notebook first.")
    st.stop()
except Exception as e:
    st.error(f"Errors encountered while starting project: {e}")
    st.stop()

# Page configuration
page_title = "Sales Forecast Dashboard"
st.set_page_config(
    page_title = page_title,
    page_icon = "📈",
    layout = "wide"
)
st.sidebar.header(page_title)

st.title(page_title)

# Category Forecast Selection
st.subheader("Category Selection")
categories = ["All Categories"] + sorted(category_forecast["Category"].unique().tolist())
st.selectbox(
    label = "Select product category",
    options = categories,
    key = "selected_category"
)

# Choose forecast source
if st.session_state.selected_category == "All Categories":
    forecast = forecast_all.copy()
    actuals = monthly_sales.rename(
        columns={"Order Date": "ds", "Sales": "actual"}
    )
else:
    forecast = category_forecast[
        category_forecast["Category"] == st.session_state.selected_category
    ].copy()
    actuals = None  # category actuals optional

st.subheader("What-If Growth Scenario")
scenario_map = {
    "Conservative (-10%)": -0.10,
    "Baseline (0%)": 0.00,
    "Optimistic (+10%)": 0.10,
    "Aggressive (+20%)": 0.20
}
st.selectbox(
    label = "Select expected growth scenario",
    options = list(scenario_map.keys()),
    index = 1,
    key = "selected_scenario"
)

# Apply growth adjustment to forecast
growth_adjustment = scenario_map[st.session_state.selected_scenario]
forecast["yhat_adj"] = forecast["yhat"] * (1 + growth_adjustment)
forecast["yhat_lower_adj"] = forecast["yhat_lower"] * (1 + growth_adjustment)
forecast["yhat_upper_adj"] = forecast["yhat_upper"] * (1 + growth_adjustment)

# KPI Summary
future_6m = forecast.tail(6)
forecast_6m_total = future_6m["yhat_adj"].sum()
best_model = errors["MAE"].idxmin()
best_mae = errors.loc[best_model, "MAE"]

c1, c2, c3, c4 = st.columns(4)
c1.metric("Category", st.session_state.selected_category)
c2.metric("Next 6 Months Forecast", f"{forecast_6m_total:,.0f}")
c3.metric("Growth Assumption", f"{growth_adjustment*100:.0f}%")
c4.metric("Model Used", f"{best_model} (MAE {best_mae:.2f})")
st.divider()

# Scenario-Based Forecast Plot
st.subheader("Scenario-Based Forecast")

fig = go.Figure()

# Plot historical actuals (only for overall forecast)
if actuals is not None:
    fig.add_trace(go.Scatter(
        x = actuals["ds"],
        y = actuals["actual"],
        mode = "lines",
        name = "Actual Sales",
        line = dict(color = "black")
    ))

# Plot adjusted forecast
fig.add_trace(go.Scatter(
    x = forecast["ds"],
    y = forecast["yhat_adj"],
    mode = "lines",
    name = f"Forecast ({st.session_state.selected_scenario})",
    line = dict(color = "#0068C9")
))

fig.add_trace(go.Scatter(
    x = forecast["ds"],
    y = forecast["yhat_lower_adj"],
    mode = "lines",
    line = dict(width = 0),
    showlegend = False
))

# Confidence interval
fig.add_trace(go.Scatter(
    x = forecast["ds"],
    y = forecast["yhat_upper_adj"],
    mode = "lines",
    fill = "tonexty",
    fillcolor = "rgba(0,104,201,0.2)",
    name = "Confidence Interval",
    line = dict(width = 0)
))

fig.update_layout(
    hovermode = "x unified",
    template = "plotly_white",
    xaxis_title = "Date",
    yaxis_title = "Sales"
)

st.plotly_chart(
    figure_or_data = fig, 
    width = "stretch"
)

download_button_graph(
    fig = fig,
    filename = "scenerio_based_forecast",
    key = "scenerio"
)
if st.session_state.scenerio:
    st.toast("Download Started")

# Forecast Table (Next 6 Months)
st.subheader("Forecast Table (Next 6 Months)")
table = future_6m[[
    "ds", "yhat_adj", "yhat_lower_adj", "yhat_upper_adj"
]].copy()

table.columns = ["Month", "Forecast", "Lower Bound", "Upper Bound"]

st.dataframe(
    data = table, 
    width = "stretch"
)

download_button_dataframe(table, "Forecast", f"forecast_{st.session_state.selected_category.replace(' ', '_')}", "forecast")
if st.session_state.forecast:
    st.toast("Download Started")

# Business Insights
st.subheader("Business Insight")
if growth_adjustment >= 0.15:
    st.success("Aggressive growth scenario selected. Plan for higher inventory and operational capacity.")
elif growth_adjustment > 0:
    st.info("Moderate growth expected. Gradual scaling recommended.")
elif growth_adjustment < 0:
    st.warning("Conservative scenario selected. Focus on cost control and promotions.")
else:
    st.info("Baseline scenario selected. Maintain current strategy.")