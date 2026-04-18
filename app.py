import streamlit as st
import joblib
import numpy as np
import pandas as pd

# load models
demand_model = joblib.load("models/demand_model.pkl")
stockout_model = joblib.load("models/stockout_model.pkl")

st.title("Demand Forecasting & Inventory Optimization")

# sidebar inputs
st.sidebar.header("User Inputs")

month = st.sidebar.slider("Month",1,12)

price = st.sidebar.slider(
    "Unit Price",
    10,
    500,
    100
)

sub_category = st.sidebar.selectbox(
    "Sub Category",
    [
        "Accessories","Appliances","Art","Binders","Bookcases",
        "Chairs","Copiers","Envelopes","Fasteners","Furnishings",
        "Labels","Machines","Paper","Phones","Storage",
        "Supplies","Tables"
    ]
)

current_stock = st.sidebar.slider("Current Stock",1,50)


ordering_cost = st.sidebar.number_input(
    "Ordering Cost",
    value=100
)

holding_cost = st.sidebar.number_input(
    "Holding Cost",
    value=5
)

lead_time = st.sidebar.number_input(
    "Lead Time (days)",
    value=5
)

Safety_Stock = st.sidebar.number_input(
    "Safety Stock (units)",
    value=20
)

# -----------------------------
# DEMAND PREDICTION
# -----------------------------

# create input structure matching training data
input_data = {
    "Month": month,
    "Unit_Price": price
}

sub_categories = [
"Accessories","Appliances","Art","Binders","Bookcases",
"Chairs","Copiers","Envelopes","Fasteners","Furnishings",
"Labels","Machines","Paper","Phones","Storage",
"Supplies","Tables"
]

for cat in sub_categories:
    input_data[f"Sub-Category_{cat}"] = 1 if cat == sub_category else 0

input_df = pd.DataFrame([input_data])

# predict demand
demand = demand_model.predict(input_df)

predicted_demand = int(demand[0])

st.subheader("Predicted Monthly Demand")

st.write(predicted_demand,"units")

# -----------------------------
# INVENTORY OPTIMIZATION
# -----------------------------

annual_demand = predicted_demand * 12

# EOQ
eoq = np.sqrt((2 * annual_demand * ordering_cost) / holding_cost)

# reorder point
daily_demand = predicted_demand / 30
rop = (daily_demand * lead_time) + Safety_Stock

st.subheader("Inventory Optimization")

st.write("Annual Demand:",annual_demand)

st.write("EOQ:",int(eoq))

st.write("Reorder Point:",int(rop))

# -----------------------------
# STOCKOUT PREDICTION
# -----------------------------

# using predicted demand as input
stockout = stockout_model.predict(
    [[predicted_demand,current_stock]]
)

st.subheader("Stockout Risk")

if stockout[0]==1:
    st.error("High Risk of Stockout")
else:
    st.success("Stock Level is Safe")