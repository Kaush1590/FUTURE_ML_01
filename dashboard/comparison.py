import io
import pathlib
import pickle
import plotly.express as px
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
        file_name = filename,
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

# Load Application State
try:
    state_path = pathlib.Path(__file__).parent.parent / "store" / "state_dump.pkl"
    state_holder = load_state(state_path)

    # Forecast outputs
    baseline_test = state_holder["baseline_forecasts"]
    errors_table = state_holder["errors"]
    future_forecast = state_holder["forecast"].tail(6)

    # Forecast outputs
    forecast_6m = future_forecast["yhat"].sum()
    avg_sales = state_holder["monthly_sales"]["Sales"].mean()

    # KPI calculations
    prophet_mae = errors_table.loc["Prophet", "MAE"]
    prophet_mape = errors_table.loc["Prophet", "MAPE"]
except FileNotFoundError as e:
    st.error("Required files are not found. Please run the training notebook first.")
    st.stop()
except Exception as e:
    st.error(f"Errors encountered while starting project: {e}")
    st.stop()

# Identify Best Model Based on MAE
best_model = errors_table["MAE"].idxmin()
best_mae = errors_table.loc[best_model, "MAE"]
best_mape = errors_table.loc[best_model, "MAPE"]

# Page configuration
page_title = "Model Comparison"
st.set_page_config(
    page_title = page_title,
    page_icon = "💻",
    layout = "wide"
)
st.sidebar.header(page_title)
st.title(page_title)

# KPI Summary
c1, c2, c3, c4 = st.columns(4)
c1.metric("Average Monthly Sales", f"{avg_sales:.0f}")
c2.metric("Forecast (Next 6 Months)", f"{forecast_6m:.0f}")
c3.metric("Best MAE", f"{best_mae:.2f}")
c4.metric("Best MAPE", f"{best_mape:.2%}")

# Error Summary
with st.expander("Errors Summary"):
    st.dataframe(
        data = errors_table.style.format({
            "MAE": "{:.2f}",
            "MAPE": "{:.2%}"
        }),
        width = "stretch"
    )
    
    download_button_dataframe(errors_table, "Errors Table", "errors", "error_df")
    if st.session_state.error_df:
        st.toast("Download Started")


st.success(f"Selected Model: {best_model}")

baseline, prophet, comparisons = st.tabs(["Baseline Forecast", "Prophet", "Error Comparison"])

# Baseline Forecast Models
with baseline:
    naive, ma, seasonal = st.tabs(tabs = ["Naïve Forecast", "Moving Average Forecast", "Seasonal Naïve Forecast"])
    
    with naive:
        df_naive = (
            baseline_test[["Order Date", "Sales", "naive_forecast"]]
            .reset_index(drop = True)
            .rename(columns = {"naive_forecast": "Naïve Forecast"})
        )
        df_naive = df_naive.set_index("Order Date")

        st.subheader("Baseline vs Actual (Test Period)")
        plot_naive = px.line(
            data_frame = df_naive,
            color_discrete_sequence = ["#0068C9", "#83C9FF"]
        )

        st.plotly_chart(
            figure_or_data = plot_naive,
            width = "stretch"
        )

        download_button_graph(plot_naive, "naive_forecast_graph.html", "naive_graph")
        if st.session_state.naive_graph:
            st.toast("Download Started")

        st.dataframe(df_naive)
        download_button_dataframe(df_naive, "Naive Forecast", "naive_forecast_table", "naive_table")
        if st.session_state.naive_table:
            st.toast("Download Started")

    with ma:
        df_ma = (
            baseline_test[["Order Date", "Sales", "ma_forecast"]]
            .reset_index(drop = True)
            .rename(columns = {"ma_forecast": "Moving Average Forecast"})
        )
        df_ma = df_ma.set_index("Order Date")

        st.subheader("Baseline vs Actual (Test Period)")
        plot_ma = px.line(
            data_frame = df_ma,
            color_discrete_sequence = ["#0068C9", "#83C9FF"]
        )

        st.plotly_chart(
            figure_or_data = plot_ma,
            width = "stretch"
        )
        
        download_button_graph(plot_ma, "moving_average_forecast_graph.html", "ma_graph")
        if st.session_state.ma_graph:
            st.toast("Download Started")

        st.dataframe(df_ma)
        download_button_dataframe(df_ma, "Moving Average Forecast", "moving_average_forecast_table", "ma_table")
        if st.session_state.ma_table:
            st.toast("Download Started")

    with seasonal:
        df_seasonal = (
            baseline_test[["Order Date", "Sales", "seasonal_naive_forecast"]]
            .reset_index(drop = True)
            .rename(columns = {"seasonal_naive_forecast": "Seasonal Naïve Forecast"})
        )
        df_seasonal = df_seasonal.set_index("Order Date")

        st.subheader("Baseline vs Actual (Test Period)")
        plot_seasonal = px.line(
            data_frame = df_seasonal,
            color_discrete_sequence = ["#0068C9", "#83C9FF"]
        )
        st.plotly_chart(
            figure_or_data = plot_seasonal,
            width = "stretch"
        )
        
        download_button_graph(plot_seasonal, "seasonal_naive_forecast_graph.html", "seasonal_graph")
        if st.session_state.seasonal_graph:
            st.toast("Download Started")

        st.dataframe(df_seasonal)
        download_button_dataframe(df_seasonal, "Seasonal Naïve Forecast", "seasonal_naive_forecast_table", "seasonal_table")
        if st.session_state.seasonal_table:
            st.toast("Download Started")

# Prophet Model Evaluation
with prophet:
    df_prophet = (
        state_holder["test"]
        .merge(
            state_holder["prophet_forecasts"][["ds", "yhat"]],
            left_on = "Order Date",
            right_on = "ds",
            how = "inner"
        )
        .drop(columns = "ds")
        .rename(columns = {"yhat": "Prophet Forecast"})
    )
    df_prophet = df_prophet.set_index("Order Date")

    st.subheader("Prophet vs Actual (Test Period)")
    plot_prophet = px.line(
        data_frame = df_prophet,
        color_discrete_sequence = ["#0068C9", "#83C9FF"]
    )

    st.plotly_chart(
        plot_prophet,
        width = "stretch"
    )

    download_button_graph(plot_prophet, "prophet_forecast_graph.html", "prophet_graph")
    if st.session_state.prophet_graph:
        st.toast("Download Started")

    st.dataframe(df_prophet)
    download_button_dataframe(df_prophet, "Prophet Forecast", "prophet_forecast_table", "prophet_table")
    if st.session_state.prophet_table:
        st.toast("Download Started")

# Model Comparison
with comparisons:
    st.subheader("MAE by Model (Lower is Better)")
    chart_df_a = errors_table["MAE"].reset_index()
    chart_df_a = chart_df_a.set_index("Model")
    plot_a = px.bar(
        data_frame = chart_df_a,
        text_auto = ".2f",
        color_discrete_sequence = ["#0068C9"]
    )
    st.plotly_chart(
        figure_or_data = plot_a,
        width = "stretch"
    )
    download_button_graph(plot_a, "mae_graph.html", "mae_graph")
    if st.session_state.mae_graph:
        st.toast("Download Started")

    plot_df = (
        state_holder["test"][["Order Date", "Sales"]]
        .rename(columns = {"Order Date": "ds", "Sales": "actual"})
        .merge(
            state_holder["prophet_forecasts"][["ds", "yhat", "yhat_lower", "yhat_upper"]],
            on = "ds",
            how = "inner"
        )
    )

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x = plot_df["ds"],
            y = plot_df["actual"],
            mode = "lines",
            name = "Actual Sales",
            line = dict(color = "black")
        )
    )

    fig.add_trace(
        go.Scatter(
            x = plot_df["ds"],
            y = plot_df["yhat"],
            mode = "lines",
            name = "Prophet Forecast",
            line = dict(color = "#1f77b4", dash = "solid")
        )
    )

    fig.add_trace(
        go.Scatter(
            x = plot_df["ds"],
            y = plot_df["yhat_lower"],
            mode = "lines",
            line = dict(width = 0),
            showlegend = False
        )
    )

    fig.add_trace(
        go.Scatter(
            x = plot_df["ds"],
            y = plot_df["yhat_upper"],
            mode = "lines",
            fill = "tonexty",
            fillcolor = "rgba(31, 119, 180, 0.2)",
            name = "Confidence Interval",
            line = dict(width = 0)
        )
    )

    fig.update_layout(
        title = "Prophet Forecast with Confidence Interval",
        xaxis_title = "Date",
        yaxis_title = "Sales",
        hovermode = "x unified",
        template = "plotly_white",
        legend = dict(orientation = "h", y = -0.25)
    )

    st.plotly_chart(
        figure_or_data = fig, 
        width = "stretch"
    )
    download_button_graph(fig, "prophet_forecast.html", "forecast_graph")
    if st.session_state.forecast_graph:
        st.toast("Download Started")

    with st.expander("Other Error Metrics"):
        st.subheader("MSE by Model (Lower is Better)")
        chart_df_b = errors_table["MSE"].reset_index()
        chart_df_b = chart_df_b.set_index("Model")
        plot_b = px.bar(
            data_frame = chart_df_b,
            text_auto = ".2f",
            color_discrete_sequence = ["#0068C9"]
        )
        st.plotly_chart(
            figure_or_data = plot_b,
            width = "stretch"
        )
        download_button_graph(plot_b, "mse_graph.html", "mse_graph")
        if st.session_state.mse_graph:
            st.toast("Download Started")

        st.subheader("MAPE % by Model (Lower is Better)")
        chart_df_c = errors_table["MAPE"].reset_index()
        chart_df_c = chart_df_c.set_index("Model")
        chart_df_c = chart_df_c["MAPE"]*100
        plot_c = px.bar(
            data_frame = chart_df_c,
            text_auto = ".2f",
            color_discrete_sequence = ["#0068C9"]
        )
        st.plotly_chart(
            figure_or_data = plot_c,
            width = "stretch"
        )
        download_button_graph(plot_c, "mape_graph.html", "mape_graph")
        if st.session_state.mape_graph:
            st.toast("Download Started")

        st.subheader("RMSE by Model (Lower is Better)")
        chart_df_d = errors_table["RMSE"].reset_index()
        chart_df_d = chart_df_d.set_index("Model")
        plot_d = px.bar(
            data_frame = chart_df_d,
            text_auto = ".2f",
            color_discrete_sequence = ["#0068C9"]
        )
        st.plotly_chart(
            figure_or_data = plot_d,
            width = "stretch"
        )
        download_button_graph(plot_d, "rmse_graph.html", "rmse_graph")
        if st.session_state.rmse_graph:
            st.toast("Download Started")