import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Set Streamlit page config
st.set_page_config(
    page_title="Fitness App Subscription Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 1. LOAD DATA
@st.cache_data
def load_data():
    df = pd.read_excel("Fitness App Subscription_DAIDM_GJ25NS003 .xlsx")
    return df

df = load_data()

# 2. SIDEBAR FILTERS
st.sidebar.header("Filter Data")
gender = st.sidebar.multiselect("Gender", df['Gender'].unique(), default=list(df['Gender'].unique()))
age = st.sidebar.slider("Age", int(df['Age'].min()), int(df['Age'].max()), (int(df['Age'].min()), int(df['Age'].max())))
city = st.sidebar.multiselect("City", df['City'].unique(), default=list(df['City'].unique()))
subscription = st.sidebar.selectbox("Subscribed", ["Both", "Yes", "No"])

filtered_df = df[
    (df['Gender'].isin(gender)) &
    (df['Age'] >= age[0]) & (df['Age'] <= age[1]) &
    (df['City'].isin(city))
]
if subscription != "Both":
    filtered_df = filtered_df[filtered_df['Subscribed'] == subscription]

# 3. HEADER
st.title("Fitness App Subscription Analytics Dashboard")
st.markdown("This dashboard provides key insights into user behavior, demographics, and engagement with the fitness app to help management make data-driven decisions.")

# 4. TABS
tabs = st.tabs([
    "Overview",
    "Demographics",
    "Engagement & Health",
    "Subscription Analysis",
    "Custom Analysis"
])

# --------------------------- TAB 1: OVERVIEW ---------------------------

with tabs[0]:
    st.subheader("General Overview")
    st.markdown("Below are some quick stats and charts to understand your user base and app subscription trends.")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Users", len(filtered_df))
    col2.metric("Subscribers", filtered_df['Subscribed'].eq('Yes').sum())
    col3.metric("Non-Subscribers", filtered_df['Subscribed'].eq('No').sum())
    sub_rate = filtered_df['Subscribed'].eq('Yes').mean() * 100
    col4.metric("Subscription Rate (%)", f"{sub_rate:.1f}%")

    # Pie Chart
    st.markdown("**Subscription Status Distribution**  
    This pie chart shows the proportion of subscribed vs non-subscribed users.")
    sub_counts = filtered_df['Subscribed'].value_counts()
    fig1, ax1 = plt.subplots()
    ax1.pie(sub_counts, labels=sub_counts.index, autopct='%1.1f%%', startangle=90)
    ax1.axis('equal')
    st.pyplot(fig1)

    # Bar Chart
    st.markdown("**Subscription by Gender**  
    See which gender is more likely to subscribe.")
    fig2, ax2 = plt.subplots()
    sns.countplot(x='Gender', hue='Subscribed', data=filtered_df, ax=ax2)
    st.pyplot(fig2)

    # KPI Table
    st.markdown("**Key Numbers Table**")
    st.dataframe(filtered_df.groupby('Subscribed').agg(
        Users=('Subscribed', 'count'),
        Avg_Age=('Age', 'mean'),
        Avg_BMI=('BMI', 'mean'),
        Avg_Steps=('Avg Steps per day', 'mean'),
        Avg_Sleep=('Sleep Hours', 'mean')
    ).reset_index(), use_container_width=True)

# ---------------------- TAB 2: DEMOGRAPHICS ----------------------------

with tabs[1]:
    st.subheader("Demographic Insights")
    st.markdown("Demographic breakdown of your user base to identify target segments.")

    # Age Distribution
    st.markdown("**Age Distribution**  
    Analyze the age profile of users and subscribers.")
    fig3, ax3 = plt.subplots()
    sns.histplot(filtered_df, x='Age', hue='Subscribed', multiple='stack', bins=15, ax=ax3)
    st.pyplot(fig3)

    # Gender-Age Boxplot
    st.markdown("**Age vs Gender**  
    Compare age ranges between genders and subscription status.")
    fig4, ax4 = plt.subplots()
    sns.boxplot(x='Gender', y='Age', hue='Subscribed', data=filtered_df, ax=ax4)
    st.pyplot(fig4)

    # City vs Subscription Rate
    st.markdown("**Subscription Rate by City**  
    Which cities have the highest conversion rates?")
    city_sub = filtered_df.groupby('City')['Subscribed'].value_counts(normalize=True).unstack().fillna(0)
    fig5, ax5 = plt.subplots(figsize=(8,4))
    city_sub['Yes'].sort_values(ascending=False).plot(kind='bar', ax=ax5)
    ax5.set_ylabel("Subscription Rate")
    st.pyplot(fig5)

    # Gender & City Heatmap
    st.markdown("**Gender Distribution across Cities**")
    gender_city = pd.crosstab(filtered_df['City'], filtered_df['Gender'])
    fig6, ax6 = plt.subplots(figsize=(8,5))
    sns.heatmap(gender_city, annot=True, fmt="d", cmap="YlGnBu", ax=ax6)
    st.pyplot(fig6)

# ---------------------- TAB 3: ENGAGEMENT & HEALTH ----------------------------

with tabs[2]:
    st.subheader("User Engagement & Health Metrics")
    st.markdown("Understand user engagement patterns and health metrics to improve product offering.")

    # Steps Distribution
    st.markdown("**Average Daily Steps**  
    Higher step count often correlates with higher engagement and health awareness.")
    fig7, ax7 = plt.subplots()
    sns.histplot(filtered_df, x='Avg Steps per day', hue='Subscribed', multiple='stack', bins=15, ax=ax7)
    st.pyplot(fig7)

    # Sleep Hours
    st.markdown("**Sleep Hours Analysis**  
    Compare sleep habits between subscribers and non-subscribers.")
    fig8, ax8 = plt.subplots()
    sns.boxplot(x='Subscribed', y='Sleep Hours', data=filtered_df, ax=ax8)
    st.pyplot(fig8)

    # BMI by Subscription
    st.markdown("**BMI Comparison**  
    Healthier users might be more interested in the app, let's check.")
    fig9, ax9 = plt.subplots()
    sns.violinplot(x='Subscribed', y='BMI', data=filtered_df, ax=ax9)
    st.pyplot(fig9)

    # Activity Level by Subscription
    if 'Activity Level' in filtered_df.columns:
        st.markdown("**Activity Level Distribution**  
        See if more active users are likely to subscribe.")
        fig10, ax10 = plt.subplots()
        sns.countplot(x='Activity Level', hue='Subscribed', data=filtered_df, ax=ax10)
        st.pyplot(fig10)

# ---------------------- TAB 4: SUBSCRIPTION ANALYSIS ----------------------------

with tabs[3]:
    st.subheader("Detailed Subscription Analysis")
    st.markdown("Analyze drivers of subscription, conversion funnels, and micro-segments.")

    # Correlation Matrix
    st.markdown("**Correlation Matrix**  
    Shows the relationship between numerical features and subscription status.")
    numeric_cols = filtered_df.select_dtypes(include=np.number).columns
    corr = filtered_df[numeric_cols].corr()
    fig11, ax11 = plt.subplots()
    sns.heatmap(corr, annot=True, cmap='coolwarm', ax=ax11)
    st.pyplot(fig11)

    # Sub Rate by Age Group
    st.markdown("**Subscription Rate by Age Group**")
    filtered_df['Age Group'] = pd.cut(filtered_df['Age'], bins=[0,18,25,35,50,100],
                                      labels=['<18','18-25','26-35','36-50','50+'])
    age_group = filtered_df.groupby('Age Group')['Subscribed'].value_counts(normalize=True).unstack().fillna(0)
    fig12, ax12 = plt.subplots()
    age_group['Yes'].plot(kind='bar', ax=ax12)
    ax12.set_ylabel("Subscription Rate")
    st.pyplot(fig12)

    # Subscription vs Steps
    st.markdown("**Steps vs Subscription**  
    Do highly active users subscribe more?")
    fig13, ax13 = plt.subplots()
    sns.boxplot(x='Subscribed', y='Avg Steps per day', data=filtered_df, ax=ax13)
    st.pyplot(fig13)

    # CrossTab Table: City, Gender, Subscription
    st.markdown("**User Count by City & Gender (Subscribed/Not)**")
    crosstab = pd.crosstab([filtered_df['City'], filtered_df['Gender']], filtered_df['Subscribed'])
    st.dataframe(crosstab)

    # Scatter Plot - BMI vs Steps
    st.markdown("**BMI vs Steps: Segmentation**  
    Are there health-conscious clusters?")
    fig14, ax14 = plt.subplots()
    sns.scatterplot(x='BMI', y='Avg Steps per day', hue='Subscribed', data=filtered_df, ax=ax14)
    st.pyplot(fig14)

# ---------------------- TAB 5: CUSTOM ANALYSIS & DEEP DIVES ----------------------------

with tabs[4]:
    st.subheader("Custom Analysis and Deep Dives")
    st.markdown("For advanced users, explore the data with custom groupings and filters.")

    # Pivot Table
    st.markdown("**Custom Pivot Table**  
    Select your own fields to group and summarize.")
    group_field = st.selectbox("Group By", options=['City','Gender','Age Group','Activity Level'])
    value_field = st.selectbox("Value Field", options=numeric_cols)
    agg_func = st.selectbox("Aggregation", options=['mean','sum','count','min','max'])
    pivot_df = filtered_df.groupby(group_field)[value_field].agg(agg_func)
    st.dataframe(pivot_df)

    # Pairplot
    st.markdown("**Pairwise Feature Relationships**  
    Explore relationships between all numeric variables.")
    import warnings
    warnings.filterwarnings("ignore")
    st.pyplot(sns.pairplot(filtered_df[numeric_cols.union(['Subscribed'])], hue='Subscribed'))

    # Raw Data Table
    st.markdown("**Raw Data View (after filtering)**")
    st.dataframe(filtered_df, use_container_width=True)

# -------------------- END OF APP --------------------
st.markdown("---")
st.caption("Dashboard created with Streamlit. © 2025 YourNameHere")

