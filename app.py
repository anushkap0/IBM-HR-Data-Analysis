import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

st.set_page_config(page_title="HR Attrition Dashboard", layout="wide")
sns.set(style="whitegrid")

st.title("🏢 HR Attrition Analysis Dashboard")
st.caption("Exploratory analysis of employee attrition, based on the IBM HR dataset.")

# ---------- Load data ----------
uploaded = st.sidebar.file_uploader("Upload IBM HR Data CSV", type="csv")

@st.cache_data
def load_data(file):
    df = pd.read_csv(file)

    # Attrition binary flag
    df["Attrition_Binary"] = df["Attrition"].apply(lambda x: 0 if x == "Current employee" else 1)

    # Clean MonthlyIncome (strip $ and ,)
    if "MonthlyIncome" in df.columns:
        df["MonthlyIncome"] = (
            df["MonthlyIncome"]
            .astype(str)
            .str.replace(",", "", regex=False)
            .str.replace("$", "", regex=False)
        )
        df["MonthlyIncome"] = pd.to_numeric(df["MonthlyIncome"], errors="coerce")

    for col in ["Age", "MonthlyIncome", "JobInvolvement", "WorkLifeBalance"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    job_map = {1: "Low", 2: "Medium", 3: "High", 4: "Very High"}
    wlb_map = {1: "Bad", 2: "Good", 3: "Better", 4: "Best"}

    if "JobInvolvement" in df.columns:
        df["JobInvolvementLabel"] = df["JobInvolvement"].map(job_map)
    if "WorkLifeBalance" in df.columns:
        df["WorkLifeBalanceLabel"] = df["WorkLifeBalance"].map(wlb_map)

    # Drop known bad rows seen in the source notebook
    if "Department" in df.columns:
        df = df[df["Department"] != "1296"]
    if "Gender" in df.columns:
        df = df[df["Gender"] != "2"]

    return df

if uploaded is None:
    st.info("Upload the IBM HR Data CSV from the sidebar to begin. (Expected columns: Attrition, Age, MonthlyIncome, Department, Gender, EducationField, JobInvolvement, WorkLifeBalance, etc.)")
    st.stop()

df = load_data(uploaded)

# ---------- Sidebar filters ----------
st.sidebar.header("Filters")

dept_options = sorted(df["Department"].dropna().unique()) if "Department" in df.columns else []
selected_depts = st.sidebar.multiselect("Department", dept_options, default=dept_options)

gender_options = sorted(df["Gender"].dropna().unique()) if "Gender" in df.columns else []
selected_genders = st.sidebar.multiselect("Gender", gender_options, default=gender_options)

if "Age" in df.columns and df["Age"].notna().any():
    age_min, age_max = int(df["Age"].min()), int(df["Age"].max())
    age_range = st.sidebar.slider("Age range", age_min, age_max, (age_min, age_max))
else:
    age_range = None

filtered = df.copy()
if selected_depts:
    filtered = filtered[filtered["Department"].isin(selected_depts)]
if selected_genders:
    filtered = filtered[filtered["Gender"].isin(selected_genders)]
if age_range:
    filtered = filtered[(filtered["Age"] >= age_range[0]) & (filtered["Age"] <= age_range[1])]

# ---------- KPI row ----------
col1, col2, col3, col4 = st.columns(4)
col1.metric("Employees", f"{len(filtered):,}")
col2.metric("Attrition Rate", f"{round(filtered['Attrition_Binary'].mean() * 100, 2)}%")
col3.metric("Avg Age", f"{filtered['Age'].mean():.1f}" if "Age" in filtered.columns else "—")
col4.metric("Avg Monthly Income", f"${filtered['MonthlyIncome'].mean():,.0f}" if "MonthlyIncome" in filtered.columns else "—")

st.divider()

# ---------- Data preview ----------
with st.expander("📋 Data preview & summary"):
    st.dataframe(filtered.head(20))
    st.write("Shape:", filtered.shape)
    st.write("Missing values per column:")
    st.dataframe(filtered.isnull().sum().rename("missing").to_frame())
    st.write("Descriptive statistics:")
    st.dataframe(filtered.describe(include="all"))

st.divider()

# ---------- Breakdown tables ----------
st.subheader("Attrition Breakdowns")
bt_cols = st.columns(3)

with bt_cols[0]:
    if "Department" in filtered.columns:
        st.write("**By Department**")
        st.dataframe(filtered.groupby(["Department", "Attrition"]).size().unstack(fill_value=0))

with bt_cols[1]:
    if "Gender" in filtered.columns:
        st.write("**By Gender**")
        st.dataframe(filtered.groupby(["Gender", "Attrition"]).size().unstack(fill_value=0))

with bt_cols[2]:
    if "EducationField" in filtered.columns:
        st.write("**By Education Field**")
        st.dataframe(filtered.groupby(["EducationField", "Attrition"]).size().unstack(fill_value=0))

st.divider()

# ---------- Charts ----------
st.subheader("Visualizations")

def show_fig(fig):
    st.pyplot(fig)
    plt.close(fig)

c1, c2 = st.columns(2)

with c1:
    if "Department" in filtered.columns:
        dept_attr = pd.crosstab(filtered["Department"], filtered["Attrition"], normalize="index") * 100
        fig, ax = plt.subplots(figsize=(7, 4))
        dept_attr.plot(kind="bar", ax=ax)
        ax.set_ylabel("Percentage")
        ax.set_title("Department-wise Attrition (%)")
        plt.xticks(rotation=20)
        plt.tight_layout()
        show_fig(fig)

with c2:
    if "Age" in filtered.columns:
        fig, ax = plt.subplots(figsize=(7, 4))
        sns.histplot(filtered["Age"].dropna(), bins=20, kde=True, color="steelblue", ax=ax)
        ax.set_title("Age Distribution")
        plt.tight_layout()
        show_fig(fig)

c3, c4 = st.columns(2)

with c3:
    if "MonthlyIncome" in filtered.columns:
        fig, ax = plt.subplots(figsize=(7, 4))
        sns.boxplot(data=filtered, x="Attrition", y="MonthlyIncome", ax=ax)
        ax.set_title("Monthly Income by Attrition")
        plt.xticks(rotation=20)
        plt.tight_layout()
        show_fig(fig)

with c4:
    if "Department" in filtered.columns:
        fig, ax = plt.subplots(figsize=(7, 4))
        sns.countplot(data=filtered, x="Department", hue="Attrition", ax=ax)
        ax.set_title("Department-wise Attrition (Counts)")
        plt.xticks(rotation=20)
        plt.tight_layout()
        show_fig(fig)

c5, c6 = st.columns(2)

with c5:
    if "Gender" in filtered.columns:
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.countplot(data=filtered, x="Gender", hue="Attrition", ax=ax)
        ax.set_title("Gender-wise Attrition")
        plt.tight_layout()
        show_fig(fig)

with c6:
    if "JobInvolvementLabel" in filtered.columns:
        fig, ax = plt.subplots(figsize=(7, 4))
        sns.countplot(data=filtered, x="JobInvolvementLabel", hue="Attrition",
                      order=["Low", "Medium", "High", "Very High"], ax=ax)
        ax.set_title("Job Involvement by Attrition")
        plt.xticks(rotation=20)
        plt.tight_layout()
        show_fig(fig)

c7, c8 = st.columns(2)

with c7:
    if "WorkLifeBalanceLabel" in filtered.columns:
        fig, ax = plt.subplots(figsize=(7, 4))
        sns.countplot(data=filtered, x="WorkLifeBalanceLabel", hue="Attrition",
                      order=["Bad", "Good", "Better", "Best"], ax=ax)
        ax.set_title("Work-Life Balance by Attrition")
        plt.xticks(rotation=20)
        plt.tight_layout()
        show_fig(fig)

with c8:
    if "EducationField" in filtered.columns:
        fig, ax = plt.subplots(figsize=(8, 4))
        sns.countplot(data=filtered, x="EducationField", hue="Attrition", ax=ax)
        ax.set_title("Education Field vs Attrition")
        plt.xticks(rotation=30)
        plt.tight_layout()
        show_fig(fig)

st.subheader("Correlation Heatmap")
num_df = filtered.select_dtypes(include=np.number)
if not num_df.empty:
    fig, ax = plt.subplots(figsize=(12, 8))
    sns.heatmap(num_df.corr(), cmap="coolwarm", center=0, ax=ax)
    ax.set_title("Correlation Heatmap")
    plt.tight_layout()
    show_fig(fig)

st.divider()

# ---------- Download cleaned data ----------
st.subheader("Download cleaned dataset")
csv = filtered.to_csv(index=False).encode("utf-8")
st.download_button("Download cleaned CSV", csv, "ibm_hr_attrition_cleaned.csv", "text/csv")
