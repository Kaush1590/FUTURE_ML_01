import io
import pandas as pd
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

# Load Application State
try:
    state_path = pathlib.Path(__file__).parent.parent / "store" / "state_dump.pkl"
    state_holder = load_state(state_path)
    dataframe = state_holder["dataframe"].set_index("Row ID")
    categorical_info = state_holder["categorical_info"]
except FileNotFoundError as e:
    st.error(f"Required files not found in the project. Please generate the files by running training notebook before proceeding.")
    st.stop()
except Exception as e:
    st.error(f"Errors encountered while starting project: {e}")
    st.stop()

# Dataset summary
summary = pd.DataFrame({
    "Column": dataframe.columns,
    "Dtype": dataframe.dtypes.astype(str),
    "Missing %": (
        dataframe.isna().mean() * 100
    ).round(2)
})

# Page configuration
page_title = "Dataset Overview"
st.set_page_config(
    page_title = page_title,
    page_icon = "📋",
    layout = "wide"
)
st.sidebar.header(page_title)
st.title(page_title)

# High-Level Dataset Metrics
c1, c2, c3, c4, c5, c6 = st.columns(6)
c1.metric("Rows", dataframe.shape[0])
c2.metric("Columns", dataframe.shape[1])
c3.metric("Categorical Cols", len(state_holder["categorical_cols"]))
c4.metric("Sales Outliers", len(state_holder["outliers"]))
c5.metric("Missing Values", sum(state_holder["missing_values"].values()))
c6.metric("Duplicate Count", len(state_holder["duplicate_entries"]))

# Column summary
with st.expander("Column Summary"):
    st.dataframe(summary)
    download_button_dataframe(summary, "Column Summary", "column_summary", "column_summary")
    if st.session_state.column_summary:
        st.toast("Download Started")

dataset, categorical, eda, addin = st.tabs(tabs = ["Dataset Contents", "Categorical Overview", "Exploratory Data Analysis (EDA)", "Additional Information"])

# Dataset Contents Tab
with dataset:
    st.multiselect(
        label = "Select columns to display",
        options = dataframe.columns,
        default = dataframe.columns,
        key = "selected_cols_data"
    )

    st.dataframe(
        data = dataframe[st.session_state.selected_cols_data],
        width = "stretch",
        height = "stretch"
    )

# Categorical Overview Tab
with categorical:
    st.dataframe(
        data = categorical_info,
        width = "stretch"
    )
    download_button_dataframe(categorical_info, "Categorical Information", "categorical_info", "categorical_info")
    if st.session_state.categorical_info:
        st.toast("Download started")

# Exploratory Data Analysis (EDA)
with eda:
    c1, c2 = st.columns(2)
    c1.metric("Starting date: ", str(dataframe['Order Date'].min()))
    c2.metric("Ending date: ", str(dataframe['Order Date'].max()))

    with st.expander("Monthly Stats", expanded = True):
        st.subheader("Monthly Sales Trend")
        monthly_sales_graph = px.line(
            data_frame = state_holder["monthly_sales"],
            x = "Order Date",
            y = "Sales",
            color_discrete_sequence = ["#0068C9"]
        )

        st.plotly_chart(
            figure_or_data = monthly_sales_graph,
            width = "stretch"
        )
        download_button_graph(monthly_sales_graph, "monthly_sales", "monthly_sales_graph")
        if st.session_state.monthly_sales_graph:
            st.toast("Download Started")

        st.subheader("Average Sales by Month")
        monthly_avg_graph = px.line(
            data_frame = state_holder["monthly_avg"],
            x = "Month",
            y = "Sales",
            color_discrete_sequence = ["#0068C9"]
        )
        
        st.plotly_chart(
            figure_or_data = monthly_avg_graph,
            width = "stretch"
        )
        download_button_graph(monthly_avg_graph, "monthly_avg", "monthly_avg_graph")
        if st.session_state.monthly_avg_graph:
            st.toast("Download Started")

        st.subheader("Box plot for Monthly Sales")
        monthly_sales_box = px.box(
            data_frame = state_holder["monthly_sales"],
            x = "Sales",
            color_discrete_sequence = ["#0068C9"]
        )

        st.plotly_chart(
            figure_or_data = monthly_sales_box,
            width = "stretch"
        )
        download_button_graph(monthly_sales_box, "monthly_sales_box", "monthly_sales_box")
        if st.session_state.monthly_sales_box:
            st.toast("Download Started")

        st.subheader("Sales Outliers (Transaction Level)")
        df_outliers = state_holder["outliers"]    
        st.multiselect(
            label = "Select columns to display",
            options = df_outliers.columns,
            default = df_outliers.columns,
            key = "selected_cols_outliers"
        )

        st.dataframe(
            data = df_outliers[st.session_state.selected_cols_outliers],
            width = "stretch"
        )
        download_button_dataframe(df_outliers, "Detected Outliers", "outliers", "outliers")
        if st.session_state.outliers:
            st.toast("Download Started")

    with st.expander("Daily Stats"):
        st.subheader("Daily Sales Trend")
        df_daily = state_holder["time_series_daily"]
        time_series_daily_graph = px.line(
            data_frame = df_daily,
            x = "Order Date",
            y = "Sales",
            color_discrete_sequence = ["#0068C9"]
        )

        st.plotly_chart(
            figure_or_data = time_series_daily_graph,
            width = "stretch"
        )
        download_button_graph(time_series_daily_graph, "time_series_daily", "time_series_daily_graph")
        if st.session_state.time_series_daily_graph:
            st.toast("Download Started")

        st.subheader("Rolling Mean and Volatility (30-Day Window)")
        fig = go.Figure()

        fig.add_trace(
            go.Scatter(
                x = df_daily["Order Date"],
                y = df_daily["Sales"],
                mode = "lines",
                name = "Daily Sales",
                line = dict(color = "#0068C9", dash = "solid")
            )
        )

        fig.add_trace(
            go.Scatter(
                x = df_daily["Order Date"],
                y = df_daily["Rolling_Mean_30"],
                mode = "lines",
                name = "30-Day Rolling Mean",
                line = dict(color = "#68C900", dash = "solid")
            )
        )

        fig.add_trace(
            go.Scatter(
                x = df_daily["Order Date"],
                y = df_daily["Rolling_Std_30"],
                mode = "lines",
                name = "30-Day Rolling Std",
                line = dict(color = "#C90068", dash = "solid")
            )
        )

        st.plotly_chart(
            figure_or_data = fig,
            width = "stretch"
        )
        download_button_graph(fig, "rolling_mean_and_volatility", "rolling_mean_and_volatility")
        if st.session_state.rolling_mean_and_volatility:
            st.toast("Download Started")

        st.subheader("Average Sales by Day of Week")
        df_week = df_daily.groupby("DayOfWeek")["Sales"].mean().reindex(
            ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        )
        df_week_graph = px.bar(
            df_week,
            y = "Sales",
            color_discrete_sequence = ["#0068C9"]
        )
        df_week_graph.update_layout(xaxis_title = "Day of Week", yaxis_title = "Average Sales")

        st.plotly_chart(
            figure_or_data = df_week_graph,
            width = "stretch"
        )
        download_button_graph(df_week_graph, "df_week_graph", "df_week_graph")
        if st.session_state.df_week_graph:
            st.toast("Download Started")

    with st.expander("Total Sales by Feature"):
        st.subheader("Total Sales by Category")
        category_sales_graph = px.bar(
            data_frame = state_holder["category_sales"],
            x = "Category",
            y = "Sales",
            color_discrete_sequence = ["#0068C9"]
        )

        st.plotly_chart(
            figure_or_data = category_sales_graph,
            width = "stretch"
        )
        download_button_graph(category_sales_graph, "category_sales_graph", "category_sales_graph")
        if st.session_state.category_sales_graph:
            st.toast("Download Started")

        st.subheader("Total Sales by Region")
        region_sales_graph = px.bar(
            data_frame = state_holder["region_sales"],
            x = "Region",
            y = "Sales",
            color_discrete_sequence = ["#0068C9"]
        )

        st.plotly_chart(
            figure_or_data = region_sales_graph,
            width = "stretch"
        )
        download_button_graph(region_sales_graph, "region_sales_graph", "region_sales_graph")
        if st.session_state.region_sales_graph:
            st.toast("Download Started")

        st.subheader("Total Sales by Segment")
        segment_sales_graph = px.bar(
            data_frame = state_holder["segment_sales"],
            x = "Segment",
            y = "Sales",
            color_discrete_sequence = ["#0068C9"]
        )

        st.plotly_chart(
            figure_or_data = segment_sales_graph,
            width = "stretch"
        )
        download_button_graph(segment_sales_graph, "segment_sales_graph", "segment_sales_graph")
        if st.session_state.segment_sales_graph:
            st.toast("Download Started")

# Additional Information
with addin:
    st.subheader("Invalid Sales")
    invalid = state_holder["invalid_sales"]
    if len(invalid) > 0:
        st.warning(f"{len(invalid)} invalid sales entries are found in the dataset")
        st.dataframe(
            data = invalid,
            width = "stretch"
        )
        download_button_dataframe(invalid, "Invalid Sales Records", "invalid_sales", "invalid_sales")
        if st.session_state.invalid_sales:
            st.toast("Download Started")
    else:
        st.success("No invalid sales entries are found in the dataset")

    st.subheader("Duplicate Entries")
    duplicate_entries = state_holder["duplicate_entries"]
    if len(duplicate_entries) > 0:
        st.warning(f"{len(duplicate_entries)} duplicate entries are found in the dataset")
        st.dataframe(
            data = duplicate_entries,
            width = "stretch"
        )
        download_button_dataframe(duplicate_entries, "Duplicate Entries", "duplicate_entries", "duplicate_entries")
        if st.session_state.duplicate_entries:
            st.toast("Download Started")
    else:
        st.success("No duplicate entries are found in the dataset")

    st.subheader("Missing Dates")
    missing_dates = state_holder["missing_dates"]

    if(len(missing_dates) > 0):
        st.warning(f"{len(missing_dates)} missing dates are found in the dataset")
        st.dataframe(
            data = missing_dates,
            width = "stretch"
        )
        download_button_dataframe(missing_dates, "Missing Dates", "missing_dates", "missing_dates")
        if st.session_state.missing_dates:
            st.toast("Download Started")
    else:
        st.success("No missing dates are found in the dataset")