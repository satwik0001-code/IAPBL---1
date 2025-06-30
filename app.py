import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(
    page_title="Fitness App Subscription Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

@st.cache_data
def load_data():
    # Read everything as string first to avoid dtype issues
    df = pd.read_excel("Fitness App Subscription_DAIDM_GJ25NS003 .xlsx", dtype=str)
    df.columns = df.columns.str.strip()
    return df

df = load_data()

# --- Sidebar Debug Info ---
with st.sidebar:
    st.write("🗂 **Columns in your dataset:**")
    st.write(df.columns.tolist())
    st.write("**Data Types Before Cleaning:**")
    st.write(df.dtypes)

# ---- COLUMN DEFINITIONS (update here if your sheet changes) ----
age_col = 'Age'
bmi_col = 'BMI'
active_min_col = 'Daily_Active_Minutes'
steps_col = 'Steps_Per_Day'
workouts_col = 'Workout_Sessions_Per_Week'
calories_col = 'Calories_Burned_Per_Day'
sleep_col = 'Hours_of_Sleep'
screen_time_col = 'Screen_Time_Minutes'
days_active_col = 'Days_Active_Per_Month'
subscribed_col = 'Subscribed'

# ---- DATA CLEANING ----
numeric_columns = [
    age_col, bmi_col, active_min_col, steps_col,
    workouts_col, calories_col, sleep_col, screen_time_col, days_active_col
]
for col in numeric_columns:
    df[col] = pd.to_numeric(df[col], errors='coerce')

# Standardize Subscribed col
df[subscribed_col] = df[subscribed_col].astype(str).str.strip().str.title()
df[subscribed_col] = df[subscribed_col].replace({'Yes': 'Yes', 'No': 'No'})

# Keep only Yes/No values
df = df[df[subscribed_col].isin(['Yes', 'No'])]

# Drop rows with NaN in essential columns
df = df.dropna(subset=numeric_columns + [subscribed_col])

with st.sidebar:
    st.write("**Data Types After Cleaning:**")
    st.write(df.dtypes)
    st.write(f"**Rows after cleaning:** {len(df)}")

# --- Robust Age/BMI slider bounds
age_vals = df[age_col].dropna()
bmi_vals = df[bmi_col].dropna()

if age_vals.empty or bmi_vals.empty:
    st.error("No valid data remains after cleaning. Please check your dataset for valid Age and BMI values.")
    st.stop()

age_min, age_max = int(age_vals.min()), int(age_vals.max())
bmi_min, bmi_max = float(bmi_vals.min()), float(bmi_vals.max())

# ---- FILTERS ----
st.sidebar.header("Filter Data")
age = st.sidebar.slider(
    "Age",
    age_min, age_max,
    (age_min, age_max)
)
bmi = st.sidebar.slider(
    "BMI",
    bmi_min, bmi_max,
    (bmi_min, bmi_max)
)
subscription = st.sidebar.selectbox("Subscribed", ["Both", "Yes", "No"])

filtered_df = df[
    (df[age_col] >= age[0]) & (df[age_col] <= age[1]) &
    (df[bmi_col] >= bmi[0]) & (df[bmi_col] <= bmi[1])
]
if subscription != "Both":
    filtered_df = filtered_df[filtered_df[subscribed_col] == subscription]

# ---- DASHBOARD ----
st.title("Fitness App Subscription Analytics Dashboard")
st.markdown(
    "This dashboard provides key insights into user behavior, health, and engagement with the fitness app to help management make data-driven decisions."
)

tabs = st.tabs([
    "Overview",
    "Engagement & Health",
    "Subscription Analysis",
    "Custom Analysis"
])

# ----------------- TAB 1: OVERVIEW -----------------
with tabs[0]:
    st.subheader("General Overview")
    st.markdown(
        "Below are some quick stats and charts to understand your user base and app subscription trends."
    )
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Users", len(filtered_df))
    col2.metric("Subscribers", filtered_df[subscribed_col].eq('Yes').sum())
    col3.metric("Non-Subscribers", filtered_df[subscribed_col].eq('No').sum())
    sub_rate = filtered_df[subscribed_col].eq('Yes').mean() * 100
    col4.metric("Subscription Rate (%)", f"{sub_rate:.1f}%")

    # Pie Chart
    st.markdown(
        "**Subscription Status Distribution**  This pie chart shows the proportion of subscribed vs non-subscribed users."
    )
    sub_counts = filtered_df[subscribed_col].value_counts()
    fig1, ax1 = plt.subplots()
    ax1.pie(sub_counts, labels=sub_counts.index, autopct='%1.1f%%', startangle=90)
    ax1.axis('equal')
    st.pyplot(fig1)

    # KPI Table
    st.markdown("**Key Numbers Table**")
    st.dataframe(filtered_df.groupby(subscribed_col).agg(
        Users=(subscribed_col, 'count'),
        Avg_Age=(age_col, 'mean'),
        Avg_BMI=(bmi_col, 'mean'),
        Avg_Steps=(steps_col, 'mean'),
        Avg_Sleep=(sleep_col, 'mean'),
        Avg_Workouts=(workouts_col, 'mean'),
        Avg_Active_Min=(active_min_col, 'mean'),
        Avg_Cals=(calories_col, 'mean'),
        Avg_Screen=(screen_time_col, 'mean'),
        Avg_Days_Active=(days_active_col, 'mean')
    ).reset_index(), use_container_width=True)

# --------------- TAB 2: ENGAGEMENT & HEALTH --------------
with tabs[1]:
    st.subheader("User Engagement & Health Metrics")
    st.markdown("Understand user engagement patterns and health metrics to improve product offering.")

    # Steps Distribution
    st.markdown("**Steps Per Day Distribution**  Higher step count often correlates with higher engagement and health awareness.")
    fig2, ax2 = plt.subplots()
    sns.histplot(filtered_df, x=steps_col, hue=subscribed_col, multiple='stack', bins=15, ax=ax2)
    st.pyplot(fig2)

    # Workout Sessions Per Week
    st.markdown("**Workout Sessions Per Week**  See the frequency of users' workout habits.")
    fig3, ax3 = plt.subplots()
    sns.histplot(filtered_df, x=workouts_col, hue=subscribed_col, multiple='stack', bins=10, ax=ax3)
    st.pyplot(fig3)

    # Sleep Hours
    st.markdown("**Sleep Hours Analysis**  Compare sleep habits between subscribers and non-subscribers.")
    fig4, ax4 = plt.subplots()
    sns.boxplot(x=subscribed_col, y=sleep_col, data=filtered_df, ax=ax4)
    st.pyplot(fig4)

    # BMI by Subscription
    st.markdown("**BMI Comparison**  Healthier users might be more interested in the app, let's check.")
    fig5, ax5 = plt.subplots()
    sns.violinplot(x=subscribed_col, y=bmi_col, data=filtered_df, ax=ax5)
    st.pyplot(fig5)

    # Daily Active Minutes
    st.markdown("**Daily Active Minutes**  Do more active users subscribe?")
    fig6, ax6 = plt.subplots()
    sns.boxplot(x=subscribed_col, y=active_min_col, data=filtered_df, ax=ax6)
    st.pyplot(fig6)

# --------------- TAB 3: SUBSCRIPTION ANALYSIS ---------------
with tabs[2]:
    st.subheader("Detailed Subscription Analysis")
    st.markdown("Analyze drivers of subscription, conversion funnels, and micro-segments.")

    # Correlation Matrix
    st.markdown("**Correlation Matrix**  Shows the relationship between numerical features and subscription status.")
    numeric_cols = filtered_df.select_dtypes(include=np.number).columns
    if len(numeric_cols) > 1:
        corr = filtered_df[numeric_cols].corr()
        fig7, ax7 = plt.subplots()
        sns.heatmap(corr, annot=True, cmap='coolwarm', ax=ax7)
        st.pyplot(fig7)

    # Sub Rate by Age Group
    st.markdown("**Subscription Rate by Age Group**")
    filtered_df['Age Group'] = pd.cut(
        filtered_df[age_col],
        bins=[0,18,25,35,50,100],
        labels=['<18','18-25','26-35','36-50','50+']
    )
    age_group = filtered_df.groupby('Age Group')[subscribed_col].value_counts(normalize=True).unstack().fillna(0)
    fig8, ax8 = plt.subplots()
    age_group['Yes'].plot(kind='bar', ax=ax8)
    ax8.set_ylabel("Subscription Rate")
    st.pyplot(fig8)

    # Subscription vs Steps
    st.markdown("**Steps vs Subscription**  Do highly active users subscribe more?")
    fig9, ax9 = plt.subplots()
    sns.boxplot(x=subscribed_col, y=steps_col, data=filtered_df, ax=ax9)
    st.pyplot(fig9)

    # Scatter Plot - BMI vs Steps
    st.markdown("**BMI vs Steps: Segmentation**  Are there health-conscious clusters?")
    fig10, ax10 = plt.subplots()
    sns.scatterplot(x=bmi_col, y=steps_col, hue=subscribed_col, data=filtered_df, ax=ax10)
    st.pyplot(fig10)

    # Calories Burned
    st.markdown("**Calories Burned Per Day**  Do those burning more calories subscribe more?")
    fig11, ax11 = plt.subplots()
    sns.boxplot(x=subscribed_col, y=calories_col, data=filtered_df, ax=ax11)
    st.pyplot(fig11)

    # Screen Time
    st.markdown("**Screen Time Minutes**  Do those with less screen time subscribe more?")
    fig12, ax12 = plt.subplots()
    sns.boxplot(x=subscribed_col, y=screen_time_col, data=filtered_df, ax=ax12)
    st.pyplot(fig12)

    # Days Active Per Month
    st.markdown("**Days Active Per Month**  More days active, more likely to subscribe?")
    fig13, ax13 = plt.subplots()
    sns.boxplot(x=subscribed_col, y=days_active_col, data=filtered_df, ax=ax13)
    st.pyplot(fig13)

# --------------- TAB 4: CUSTOM ANALYSIS & DEEP DIVES ---------------
with tabs[3]:
    st.subheader("Custom Analysis and Deep Dives")
    st.markdown("For advanced users, explore the data with custom groupings and filters.")

    # Pivot Table
    st.markdown("**Custom Pivot Table**  Select your own fields to group and summarize.")
    group_field = st.selectbox("Group By", options=['Age Group', subscribed_col])
    value_fields = [steps_col, bmi_col, sleep_col, age_col, active_min_col, calories_col, screen_time_col, days_active_col, workouts_col]
    value_field = st.selectbox("Value Field", options=value_fields)
    agg_func = st.selectbox("Aggregation", options=['mean','sum','count','min','max'])
    pivot_df = filtered_df.groupby(group_field)[value_field].agg(agg_func)
    st.dataframe(pivot_df)

    # Pairplot
    st.markdown("**Pairwise Feature Relationships**  Explore relationships between all numeric variables.")
    import warnings
    warnings.filterwarnings("ignore")
    if len(filtered_df) <= 500 and len(numeric_cols) >= 2:
        pairplot_fig = sns.pairplot(filtered_df[numeric_cols.to_list() + [subscribed_col]], hue=subscribed_col)
        st.pyplot(pairplot_fig)
    else:
        st.info("Pairplot disabled for large data (>500 rows) or not enough numeric columns.")

    # Raw Data Table
    st.markdown("**Raw Data View (after filtering)**")
    st.dataframe(filtered_df, use_container_width=True)

st.markdown("---")
st.caption("Dashboard created with Streamlit. © 2025 YourNameHere")

