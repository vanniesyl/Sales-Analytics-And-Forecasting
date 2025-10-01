import streamlit as st
import plotly.express as px
import pandas as pd
import os
import warnings
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from prophet import Prophet
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')

st.set_page_config(page_title="Superstore!!!", page_icon=":bar_chart:", layout="wide")

st.title(" :bar_chart: Interactive Dashboard")
st.markdown('<style>div.block-container{padding-top:1rem;}</style>', unsafe_allow_html=True)

fl = st.file_uploader(":file_folder: Upload a file", type=(["csv", "txt", "xlsx", "xls"]))
if fl is not None:
    filename = fl.name
    st.write(filename)
    df = pd.read_csv(fl, encoding="ISO-8859-1")
else:
    os.chdir(r"D:\Interactive-Dashboard-1\interactive_Dashboard")
    df = pd.read_csv("Superstore.csv", encoding="ISO-8859-1")

# Check if the "Category" column exists in the DataFrame
if "Category" in df.columns:
    category_col = "Category"
elif "item type" in df.columns:
    category_col = "Item Type"
else:
    st.error("Neither 'Category' nor 'Item type' column exists in the dataset. Please upload a valid dataset.")
    st.stop()

# Create filter options for Region, State, and City
region = st.sidebar.multiselect("Pick your Region", df["Region"].unique())
if not region:
    df2 = df.copy()
else:
    df2 = df[df["Region"].isin(region)]

state = st.sidebar.multiselect("Pick the State", df2["State"].unique())
if not state:
    df3 = df2.copy()
else:
    df3 = df2[df2["State"].isin(state)]

city = st.sidebar.multiselect("Pick the City", df3["City"].unique())

# Filter the data based on Region, State, and City
if not region and not city:
    filtered_df = df
elif not city:
    filtered_df = df[df["Region"].isin(region)]
elif not region:
    filtered_df = df[df["State"].isin(state)]
elif city:
    filtered_df = df[df["Region"].isin(region) & df["City"].isin(city)]
elif region:
    filtered_df = df[df["Region"].isin(region) & df["State"].isin(state)]
else:
    filtered_df = df[df["Region"].isin(region) & df["State"].isin(state) & df["City"].isin(city)]

col1, col2 = st.columns((2))
df["Order Date"] = pd.to_datetime(df["Order Date"], errors='coerce')

# Getting the min and max date 
startDate = pd.to_datetime(df["Order Date"]).min()
endDate = pd.to_datetime(df["Order Date"]).max()

with col1:
    date1 = pd.to_datetime(st.date_input("Start Date", startDate))

with col2:
    date2 = pd.to_datetime(st.date_input("End Date", endDate))

df = df[(df["Order Date"] >= date1) & (df["Order Date"] <= date2)].copy()

st.sidebar.header("Choose your filter: ")

category_df = filtered_df.groupby(by=[category_col], as_index=False)["Sales"].sum()

with col1:
    st.subheader(f"{category_col} wise Sales")
    fig = px.bar(category_df, x=category_col, y="Sales", text=['${:,.2f}'.format(x) for x in category_df["Sales"]],
                 template="seaborn")
    st.plotly_chart(fig, use_container_width=True, height=200)

with col2:
    st.subheader("Region wise Sales")
    fig = px.pie(filtered_df, values="Sales", names="Region", hole=0.5)
    fig.update_traces(text=filtered_df["Region"], textposition="outside")
    st.plotly_chart(fig, use_container_width=True)

cl1, cl2 = st.columns((2))
with cl1:
    with st.expander(f"{category_col}_ViewData"):
        st.write(category_df.style.background_gradient(cmap="Blues"))
        csv = category_df.to_csv(index=False).encode('utf-8')
        st.download_button("Download Data", data=csv, file_name=f"{category_col}.csv", mime="text/csv",
                           help=f'Click here to download the data as a CSV file')

with cl2:
    with st.expander("Region_ViewData"):
        region = filtered_df.groupby(by="Region", as_index=False)["Sales"].sum()
        st.write(region.style.background_gradient(cmap="Oranges"))
        csv = region.to_csv(index=False).encode('utf-8')
        st.download_button("Download Data", data=csv, file_name="Region.csv", mime="text/csv",
                           help='Click here to download the data as a CSV file')

filtered_df["month_year"] = filtered_df["Order Date"].dt.to_period("M")
st.subheader('Time Series Analysis')

linechart = pd.DataFrame(filtered_df.groupby(filtered_df["month_year"].dt.strftime("%Y : %b"))["Sales"].sum()).reset_index()
fig2 = px.line(linechart, x="month_year", y="Sales", labels={"Sales": "Amount"}, height=500, width=1000, template="gridon")
st.plotly_chart(fig2, use_container_width=True)

with st.expander("View Data of TimeSeries:"):
    st.write(linechart.T.style.background_gradient(cmap="Blues"))
    csv = linechart.to_csv(index=False).encode("utf-8")
    st.download_button('Download Data', data=csv, file_name="TimeSeries.csv", mime='text/csv')

# Future Prediction
st.subheader("Hierarchical view of Sales using TreeMap")
fig3 = px.treemap(filtered_df, path=["Region", category_col, "Sub-Category"], values="Sales", hover_data=["Sales"],
                  color="Sub-Category")
fig3.update_layout(width=800, height=650)
st.plotly_chart(fig3, use_container_width=True)

chart1, chart2 = st.columns((2))
with chart1:
    st.subheader('Segment wise Sales')
    fig = px.pie(filtered_df, values="Sales", names="Segment", template="plotly_dark")
    fig.update_traces(text=filtered_df["Segment"], textposition="inside")
    st.plotly_chart(fig, use_container_width=True)

with chart2:
    st.subheader(f"{category_col} wise Sales")
    fig = px.pie(filtered_df, values="Sales", names=category_col, template="gridon")
    fig.update_traces(text=filtered_df[category_col], textposition="inside")
    st.plotly_chart(fig, use_container_width=True)

import plotly.figure_factory as ff
st.subheader(":point_right: Month wise Sub-Category Sales Summary")
with st.expander("Summary_Table"):
    df_sample = df[0:5][["Region", "State", "City", category_col, "Sales", "Profit", "Quantity"]]
    fig = ff.create_table(df_sample, colorscale="Cividis")
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("Month wise sub-Category Table")
    filtered_df["month"] = filtered_df["Order Date"].dt.month_name()
    sub_category_Year = pd.pivot_table(data=filtered_df, values="Sales", index=["Sub-Category"], columns="month")
    st.write(sub_category_Year.style.background_gradient(cmap="Blues"))

# Create a scatter plot
data1 = px.scatter(filtered_df, x="Sales", y="Profit", size="Quantity")
data1['layout'].update(
    title=dict(
        text="Relationship between Sales and Profits using Scatter Plot.",
        font=dict(size=20)
    ),
    xaxis=dict(
        title=dict(
            text="Sales",
            font=dict(size=19)
        )
    ),
    yaxis=dict(
        title=dict(
            text="Profit",
            font=dict(size=19)
        )
    )
)

st.plotly_chart(data1, use_container_width=True)
st.subheader("📈 Sales Forecasting")

# User input for forecast horizon
periods_input = st.number_input(
    "Enter how many periods to forecast:",
    min_value=1,
    value=6
)

freq_input = st.selectbox(
    "Select forecast frequency:",
    options=["Days", "Months", "Years"]
)

# Convert frequency to Prophet format
freq_map = {
    "Days": "D",
    "Months": "M",
    "Years": "Y"
}
freq = freq_map[freq_input]

# Prepare data
sales_df = filtered_df[["Order Date", "Sales"]].rename(columns={"Order Date": "ds", "Sales": "y"})
sales_df = sales_df.dropna(subset=['ds', 'y'])

# Optional: sort by date
sales_df = sales_df.sort_values('ds')
# Fit the model
model = Prophet(mcmc_samples=0)
model.fit(sales_df)

# Make future dataframe according to user choice
future = model.make_future_dataframe(periods=periods_input, freq=freq)
forecast = model.predict(future)

# Save forecast for table
linechart_combined = forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]].set_index("ds")

# Show forecast plots
st.subheader("🔮 Forecast Plot")
fig1 = model.plot(forecast)
st.pyplot(fig1)

st.subheader("📊 Forecast Components")
fig2 = model.plot_components(forecast)
st.pyplot(fig2)

# Expander with forecast data
with st.expander("View Forecast Data Table:"):
    st.write(linechart_combined.T.style.background_gradient(cmap="Blues"))


# Download original DataSet
csv = df.to_csv(index=False).encode('utf-8')
st.download_button('Download Data', data=csv, file_name="Data.csv", mime="text/csv")

# Predict future sales
# Prepare data for prediction
try:
    # Exclude rows with NaT values in 'Order Date'
    filtered_df = filtered_df.dropna(subset=['Order Date'])

    # Add month and year as features
    filtered_df['Order Date Ordinal'] = filtered_df['Order Date'].apply(lambda x: x.toordinal())
    filtered_df['Month'] = filtered_df['Order Date'].dt.month
    filtered_df['Year'] = filtered_df['Order Date'].dt.year
except Exception as e:
    st.error(f"Error: {e}")
    st.error("Failed to process date values. Please check your date data.")
    st.stop()

# Split data into features and target variable
X = filtered_df[['Order Date Ordinal', 'Month', 'Year']]
y = filtered_df['Sales']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train RandomForestRegressor model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predict future sales
# Generate future dates until 2025
future_dates = pd.date_range(start=pd.to_datetime(filtered_df['Order Date']).max(), end='2025-01-01')

# Convert future dates to ordinal, month, and year
future_dates_df = pd.DataFrame({
    'Order Date': future_dates,
    'Order Date Ordinal': future_dates.map(lambda x: x.toordinal()),
    'Month': future_dates.map(lambda x: x.month),
    'Year': future_dates.map(lambda x: x.year)
})

# Predict sales for future dates
future_sales = model.predict(future_dates_df[['Order Date Ordinal', 'Month', 'Year']])

# Combine actual and future sales data
filtered_df['Type'] = 'Actual'
future_sales_df = pd.DataFrame({'Order Date': future_dates, 'Sales': future_sales, 'Type': 'Predicted'})

# Concatenate the actual and future sales DataFrame
combined_df = pd.concat([filtered_df[['Order Date', 'Sales', 'Type']], future_sales_df])

# Convert Order Date to month_year format
combined_df['month_year'] = combined_df['Order Date'].dt.to_period("M")

# Plot the combined data using Plotly
linechart_combined = combined_df.groupby(['month_year'])['Sales'].sum().reset_index()

# Ensure all date data is in string format for Plotly
linechart_combined['month_year'] = linechart_combined['month_year'].astype(str)

fig_combined = px.line(linechart_combined, x="month_year", y="Sales",  labels={"Sales": "Amount"}, 
                       height=500, width=1000, template="gridon")
fig_combined.update_layout(title="Actual and Predicted Sales Over Time")

# Display the plot in Streamlit
st.plotly_chart(fig_combined, use_container_width=True)
if 'linechart_combined' in locals():
    st.write(linechart_combined.T.style.background_gradient(cmap="Blues"))
    csv = linechart_combined.to_csv(index=False).encode("utf-8")
    st.download_button('Download Data', data=csv, file_name="PredictedTimeSeries.csv", mime='text/csv')

# Show predicted sales values in a table
st.subheader('Predicted Sales for Future Dates')
st.write(future_sales_df)
