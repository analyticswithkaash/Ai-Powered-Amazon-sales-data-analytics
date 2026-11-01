
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from scipy.stats import pearsonr, spearmanr
import statsmodels.formula.api as smf
from pathlib import Path

st.set_page_config(
    page_title="Amazon Analytics Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

@st.cache_data
def load_data():
    file_path = Path(__file__).resolve().parent / "amazon_cleaned.csv"
    return pd.read_csv(file_path)

df = load_data()

st.markdown("""
<style>
.stApp {
    background-color: #101827;
    color: #F8FAFC;
}
[data-testid="stSidebar"] {
    background-color: #172235;
}
[data-testid="stMetric"] {
    background: linear-gradient(135deg, #1E293B, #26364F);
    border: 1px solid #40516B;
    border-left: 4px solid #FF9900;
    border-radius: 14px;
    padding: 22px 18px;
    box-shadow: 0 5px 18px rgba(0, 0, 0, 0.25);
}
[data-testid="stMetricLabel"] {
    color: #CBD5E1;
}
[data-testid="stMetricValue"] {
    color: #FFFFFF;
    font-size: 28px;
    font-weight: 700;
}
h1, h2, h3 {
    color: #F8FAFC;
}
</style>
""", unsafe_allow_html=True)

st.title("Amazon Sales Data Analytics Dashboard")
st.caption("Data Analytics & Machine Learning Internship Project")

st.divider()

st.success("Dataset loaded successfully!")

st.write("Total Products:", len(df))
st.write("Total Columns:", len(df.columns))

st.subheader("Dataset Preview")
st.dataframe(df.head(10), width="stretch")




st.markdown("""
<style>
[data-testid="stSidebar"] {
    background-color: #172235;
}

[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3,
[data-testid="stSidebar"] p,
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] span {
    color: #F8FAFC !important;
}

[data-testid="stSidebar"] [data-testid="stWidgetLabel"] {
    color: #FFFFFF !important;
    font-weight: 600;
}

[data-testid="stSidebar"] [data-baseweb="select"] > div {
    background-color: #26364F !important;
    border: 1px solid #52647E !important;
    color: #FFFFFF !important;
}

[data-testid="stSidebar"] [data-baseweb="tag"] {
    background-color: #40516B !important;
}

[data-testid="stSidebar"] [data-baseweb="tag"] span {
    color: #FFFFFF !important;
}

[data-testid="stSidebar"] [data-testid="stCaptionContainer"] p {
    color: #CBD5E1 !important;
}
</style>
""", unsafe_allow_html=True)


st.sidebar.markdown("### 📌 Quick Navigation")

st.sidebar.markdown("""
- [📊 Executive Overview](#executive-overview)
- [🧩 Task 1 — Product Segmentation](#task-1-product-segmentation)
- [📉 Task 2 — Discount vs Rating](#task-2-discount-vs-rating-analysis)
- [💬 Task 3 — Sentiment Analysis](#task-3-review-based-sentiment-analysis)
- [🏆 Task 4 — Product Ranking](#task-4-product-ranking)
- [🔍 Task 5 — Anomaly Detection](#task-5-anomaly-outlier-detection)
- [🤖 Task 6 — Rating Prediction](#task-6-predictive-rating-analysis)
""")

st.sidebar.divider()

st.sidebar.title("Dashboard Controls")

def group_category(category):
    category = str(category).split("|")[0].strip()

    if category in [
        "Electronics",
        "Home&Kitchen",
        "Computers&Accessories",
        "OfficeProducts"
    ]:
        return category

    return "Other"

df["category_grouped"] = df["category"].apply(group_category)

categories = sorted(df["category_grouped"].unique().tolist())

selected_categories = st.sidebar.multiselect(
    "Select Product Categories",
    options=categories,
    default=categories
)

filtered_df = df[df["category_grouped"].isin(selected_categories)].copy()

st.sidebar.divider()
st.sidebar.caption(f"Showing {len(filtered_df):,} of {len(df):,} products")

st.subheader("Executive Overview")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Total Products", f"{len(filtered_df):,}")

with col2:
    st.metric("Average Rating", f"{filtered_df['rating'].mean():.2f} / 5")

with col3:
    st.metric("Average Discount", f"{filtered_df['discount_percentage'].mean():.1f}%")

with col4:
    st.metric("Total Rating Counts", f"{filtered_df['rating_count'].sum():,}")


st.divider()
st.header("Task 1 — Product Segmentation")

@st.cache_data
def load_segmentation():
    return pd.read_csv(
        Path(__file__).resolve().parent / "task_1_product_segmentation.csv"
    )

segmentation_df = load_segmentation()

segmentation_df["category_grouped"] = segmentation_df[
    "category"
].apply(group_category)

segmentation_df = segmentation_df[
    segmentation_df["category_grouped"].isin(selected_categories)
].copy()

st.markdown("### K-Means Customer Demand Segments")

segment_count = segmentation_df["Cluster"].nunique()
segmented_products = len(segmentation_df)

seg_col1, seg_col2, seg_col3 = st.columns(3)

with seg_col1:
    st.metric("Segmented Products", f"{segmented_products:,}")

with seg_col2:
    st.metric("Active Clusters", segment_count)

with seg_col3:
    st.metric(
        "Average Rating",
        f"{segmentation_df['rating'].mean():.2f}"
        if segmented_products else "N/A"
    )

if not segmentation_df.empty:

    cluster_summary = (
        segmentation_df.groupby(
            ["Cluster", "Segment_Name"],
            as_index=False
        )
        .agg(
            Products=("product_id", "size"),
            Median_Price=("discounted_price", "median"),
            Median_Discount=("discount_percentage", "median"),
            Median_Rating=("rating", "median"),
            Median_Rating_Count=("rating_count", "median")
        )
    )

    cluster_summary["Cluster_Label"] = (
        "Cluster "
        + cluster_summary["Cluster"].astype(str)
    )

    chart_col1, chart_col2 = st.columns(2)

    with chart_col1:
        fig_cluster = px.bar(
            cluster_summary,
            x="Cluster_Label",
            y="Products",
            color="Segment_Name",
            title="Product Distribution Across Clusters",
            text="Products",
            hover_data=["Segment_Name"]
        )

        fig_cluster.update_layout(
            template="plotly_dark",
            paper_bgcolor="#101827",
            plot_bgcolor="#172235",
            height=450,
            showlegend=False
        )

        st.plotly_chart(fig_cluster, width="stretch")

    with chart_col2:
        fig_popularity = px.bar(
            cluster_summary,
            x="Cluster_Label",
            y="Median_Rating_Count",
            color="Segment_Name",
            title="Median Popularity by Cluster",
            hover_data=["Segment_Name"]
        )

        fig_popularity.update_layout(
            template="plotly_dark",
            paper_bgcolor="#101827",
            plot_bgcolor="#172235",
            height=450,
            showlegend=False
        )

        st.plotly_chart(fig_popularity, width="stretch")

    chart_col3, chart_col4 = st.columns(2)

    with chart_col3:
        fig_price = px.bar(
            cluster_summary,
            x="Cluster_Label",
            y="Median_Price",
            color="Segment_Name",
            title="Median Discounted Price by Cluster",
            labels={"Median_Price": "Price (INR)"}
        )

        fig_price.update_layout(
            template="plotly_dark",
            paper_bgcolor="#101827",
            plot_bgcolor="#172235",
            height=450,
            showlegend=False
        )

        st.plotly_chart(fig_price, width="stretch")

    with chart_col4:
        fig_discount = px.bar(
            cluster_summary,
            x="Cluster_Label",
            y="Median_Discount",
            color="Segment_Name",
            title="Median Discount Percentage by Cluster"
        )

        fig_discount.update_layout(
            template="plotly_dark",
            paper_bgcolor="#101827",
            plot_bgcolor="#172235",
            height=450,
            showlegend=False
        )

        st.plotly_chart(fig_discount, width="stretch")

    st.markdown("### Cluster Performance Comparison")

    st.dataframe(
        cluster_summary[
            [
                "Cluster",
                "Segment_Name",
                "Products",
                "Median_Price",
                "Median_Discount",
                "Median_Rating",
                "Median_Rating_Count"
            ]
        ].round(2),
        width="stretch",
        hide_index=True
    )

    st.caption(
        "Clusters and segment names are loaded from the original "
        "Task 1 internship results. Charts respond to category filters."
    )

else:
    st.info("No products match the selected category filters.")






st.markdown("### Elbow Method & Silhouette Analysis")

st.caption(
    "Original Task 1 clustering diagnostics used to select K = 5."
)

chart_path = (
    Path(__file__).resolve().parent
    / "task1_elbow_silhouette.png"
)

if chart_path.exists():
    from PIL import Image

    with Image.open(chart_path) as img:
        st.image(
            img.copy(),
            caption="Original K-Means clustering diagnostic chart",
            width="stretch"
        )
else:
    st.warning("Original clustering chart file not found.")

st.metric("Optimal Number of Clusters", "5")

st.info(
    "K = 5 was selected in the original Task 1 analysis "
    "using Silhouette Score and Elbow Method diagnostics."
)


st.markdown("### K-Means vs Hierarchical Clustering")

km_col1, km_col2, km_col3 = st.columns(3)

with km_col1:
    st.metric("K-Means Silhouette Score", "0.2360")

with km_col2:
    st.metric("Hierarchical Silhouette Score", "0.1613")

with km_col3:
    st.metric("Adjusted Rand Index", "0.3692")

score_df = pd.DataFrame({
    "Algorithm": ["K-Means", "Hierarchical"],
    "Silhouette Score": [0.2360, 0.1613]
})

fig_scores = px.bar(
    score_df,
    x="Algorithm",
    y="Silhouette Score",
    color="Algorithm",
    text="Silhouette Score",
    title="Clustering Quality Comparison"
)

fig_scores.update_traces(texttemplate="%{text:.4f}", textposition="outside")

fig_scores.update_layout(
    template="plotly_dark",
    paper_bgcolor="#101827",
    plot_bgcolor="#172235",
    height=400,
    showlegend=False,
    yaxis_range=[0, 0.30]
)

st.plotly_chart(fig_scores, width="stretch")

st.markdown("#### Cluster Assignment Comparison")

cluster_matrix = pd.crosstab(
    segmentation_df["Cluster"],
    segmentation_df["Hierarchical_Cluster"]
)

fig_agreement = px.imshow(
    cluster_matrix,
    text_auto=True,
    aspect="auto",
    color_continuous_scale="Blues",
    labels={
        "x": "Hierarchical Cluster",
        "y": "K-Means Cluster",
        "color": "Products"
    },
    title="K-Means vs Hierarchical Cluster Assignments"
)

fig_agreement.update_layout(
    template="plotly_dark",
    paper_bgcolor="#101827",
    plot_bgcolor="#172235",
    height=450
)

st.plotly_chart(fig_agreement, width="stretch")

st.info(
    "K-Means achieved a higher Silhouette Score than Hierarchical "
    "Clustering in the original analysis. The Adjusted Rand Index "
    "indicates moderate agreement between the two clustering solutions."
)

st.caption(
    "Silhouette Scores and ARI are original full-dataset Task 1 results. "
    "The cluster assignment heatmap responds to dashboard category filters."
)


st.subheader("Product Distribution by Category")

category_counts = (
    filtered_df["category_grouped"]
    .value_counts()
    .reset_index()
)

category_counts.columns = ["Category", "Products"]

fig = px.bar(
    category_counts,
    x="Category",
    y="Products",
    color="Category",
    text="Products",
    color_discrete_sequence=[
        "#FF9900",
        "#00B4D8",
        "#8B5CF6",
        "#10B981",
        "#F43F5E"
    ]
)

fig.update_traces(
    textposition="outside",
    hovertemplate="<b>%{x}</b><br>Products: %{y:,}<extra></extra>"
)

fig.update_layout(
    template="plotly_dark",
    paper_bgcolor="#101827",
    plot_bgcolor="#172235",
    showlegend=False,
    height=450,
    xaxis_title="Product Category",
    yaxis_title="Number of Products",
    margin=dict(l=20, r=20, t=40, b=40)
)

st.plotly_chart(fig, width="stretch")


st.divider()
st.subheader("Price & Discount Analysis")

left_col, right_col = st.columns(2)

with left_col:
    st.markdown("#### Product Price Distribution")

    fig_price = px.histogram(
        filtered_df,
        x="discounted_price",
        nbins=40,
        color_discrete_sequence=["#FF9900"],
        labels={"discounted_price": "Discounted Price (₹)"}
    )

    fig_price.update_layout(
        template="plotly_dark",
        paper_bgcolor="#101827",
        plot_bgcolor="#172235",
        height=420,
        xaxis_title="Discounted Price (₹)",
        yaxis_title="Number of Products",
        bargap=0.05
    )

    st.plotly_chart(fig_price, width="stretch")

with right_col:
    st.markdown("#### Discount vs Customer Rating")

    fig_discount = px.scatter(
        filtered_df,
        x="discount_percentage",
        y="rating",
        color="category_grouped",
        hover_data=["product_name"],
        opacity=0.65,
        labels={
            "discount_percentage": "Discount (%)",
            "rating": "Customer Rating",
            "category_grouped": "Category"
        },
        color_discrete_sequence=[
            "#FF9900",
            "#00B4D8",
            "#8B5CF6",
            "#10B981",
            "#F43F5E"
        ]
    )

    fig_discount.update_layout(
        template="plotly_dark",
        paper_bgcolor="#101827",
        plot_bgcolor="#172235",
        height=420,
        xaxis_title="Discount (%)",
        yaxis_title="Customer Rating"
    )

    st.plotly_chart(fig_discount, width="stretch")


st.divider()
st.divider()

st.divider()
st.subheader("Task 1 — Key Findings & Conclusion")

st.markdown("""
#### Key Findings

1. **Optimal Clusters:** K-Means identified five product segments using discounted price, discount percentage, rating, and rating count.

2. **Clustering Performance:** K-Means achieved a Silhouette Score of approximately **0.236**, outperforming Hierarchical Clustering (**0.161**). The Adjusted Rand Index was approximately **0.369**, indicating partial agreement between the two methods.

3. **Customer Demand:** The Popular High-Discount segment showed the highest median rating count (**20,850**) and a median customer rating of **4.20**.

4. **Price Segmentation:** The Premium-Priced segment had the highest median discounted price (**₹8,799**), distinguishing it from lower-priced segments.

5. **Commercial Potential:** The Popular High-Discount segment appears most promising based on customer ratings and review volume, which serve as demand proxies rather than actual sales figures.

#### Business Conclusion

K-Means segmentation provides a useful framework for identifying distinct Amazon product groups. The Popular High-Discount segment is a strong candidate for marketing and promotional prioritization. However, the modest Silhouette Score indicates overlapping segments, and actual profitability cannot be established without sales, cost, and margin data.
""")

st.header("Task 2 — Discount vs Rating Analysis")

st.subheader("Pearson & Spearman Correlation Analysis")

correlation_df = filtered_df[
    ["discount_percentage", "rating"]
].dropna()

if (
    len(correlation_df) >= 3
    and correlation_df["discount_percentage"].nunique() > 1
    and correlation_df["rating"].nunique() > 1
):
    pearson_r, pearson_p = pearsonr(
        correlation_df["discount_percentage"],
        correlation_df["rating"]
    )

    spearman_r, spearman_p = spearmanr(
        correlation_df["discount_percentage"],
        correlation_df["rating"]
    )

    p_col1, p_col2, p_col3, p_col4 = st.columns(4)

    with p_col1:
        st.metric("Pearson r", f"{pearson_r:.4f}")

    with p_col2:
        st.metric("Pearson p-value", f"{pearson_p:.2e}")

    with p_col3:
        st.metric("Spearman rho", f"{spearman_r:.4f}")

    with p_col4:
        st.metric("Spearman p-value", f"{spearman_p:.2e}")

    st.caption(
        f"Correlation calculated using {len(correlation_df):,} "
        "products from the selected categories."
    )

    if pearson_p < 0.05:
        st.info(
            "Pearson correlation is statistically significant "
            "at the 5% significance level."
        )
    else:
        st.info(
            "Pearson correlation is not statistically significant "
            "at the 5% significance level."
        )

    st.caption(
        "Correlation measures association, not causation. "
        "Statistical significance does not imply a strong relationship."
    )

else:
    st.warning(
        "Insufficient variation or observations to calculate correlations."
    )


st.divider()
st.subheader("Category-wise Correlation Comparison")

correlation_results = []

for category, group in filtered_df.groupby("category_grouped"):
    valid = group[["discount_percentage", "rating"]].dropna()

    if (
        len(valid) >= 3
        and valid["discount_percentage"].nunique() > 1
        and valid["rating"].nunique() > 1
    ):
        pearson_value, pearson_pvalue = pearsonr(
            valid["discount_percentage"],
            valid["rating"]
        )

        spearman_value, spearman_pvalue = spearmanr(
            valid["discount_percentage"],
            valid["rating"]
        )

        correlation_results.append({
            "Category": category,
            "Products": len(valid),
            "Pearson r": pearson_value,
            "Pearson p-value": pearson_pvalue,
            "Spearman rho": spearman_value,
            "Spearman p-value": spearman_pvalue
        })

if correlation_results:
    category_corr_df = pd.DataFrame(correlation_results)

    chart_data = category_corr_df.melt(
        id_vars=["Category"],
        value_vars=["Pearson r", "Spearman rho"],
        var_name="Correlation Method",
        value_name="Coefficient"
    )

    fig_category_corr = px.bar(
        chart_data,
        x="Category",
        y="Coefficient",
        color="Correlation Method",
        barmode="group",
        text_auto=".3f",
        color_discrete_map={
            "Pearson r": "#FF9900",
            "Spearman rho": "#00B4D8"
        }
    )

    fig_category_corr.update_layout(
        template="plotly_dark",
        paper_bgcolor="#101827",
        plot_bgcolor="#172235",
        height=450,
        xaxis_title="Product Category",
        yaxis_title="Correlation Coefficient",
        yaxis_range=[-1, 1]
    )

    st.plotly_chart(fig_category_corr, width="stretch")

    st.dataframe(
        category_corr_df.round(4),
        width="stretch",
        hide_index=True
    )

else:
    st.warning("Insufficient data for category-wise correlation.")


st.divider()
st.subheader("OLS Regression Analysis")

regression_df = filtered_df[
    [
        "rating",
        "discount_percentage",
        "discounted_price",
        "rating_count",
        "category_grouped"
    ]
].dropna().copy()

if len(regression_df) >= 20 and regression_df["rating"].nunique() > 1:

    regression_df["log_price"] = np.log1p(
        regression_df["discounted_price"]
    )

    regression_df["log_rating_count"] = np.log1p(
        regression_df["rating_count"]
    )

    regression_model = smf.ols(
        formula=(
            "rating ~ discount_percentage + log_price "
            "+ log_rating_count + C(category_grouped)"
        ),
        data=regression_df
    ).fit(cov_type="HC3")

    reg_col1, reg_col2, reg_col3, reg_col4 = st.columns(4)

    with reg_col1:
        st.metric("R²", f"{regression_model.rsquared:.4f}")

    with reg_col2:
        st.metric("Adjusted R²", f"{regression_model.rsquared_adj:.4f}")

    with reg_col3:
        st.metric("AIC", f"{regression_model.aic:.2f}")

    with reg_col4:
        st.metric("Observations", f"{int(regression_model.nobs):,}")

    confidence = regression_model.conf_int()

    coefficients_df = pd.DataFrame({
        "Feature": regression_model.params.index,
        "Coefficient": regression_model.params.values,
        "P-value": regression_model.pvalues.values,
        "CI Lower": confidence.iloc[:, 0].values,
        "CI Upper": confidence.iloc[:, 1].values
    })

    coefficients_df["Significance"] = coefficients_df["P-value"].apply(
        lambda p: "Significant" if p < 0.05 else "Not Significant"
    )

    st.markdown("#### Regression Coefficients")

    st.dataframe(
        coefficients_df.round(6),
        width="stretch",
        hide_index=True
    )

    fig_coeff = px.bar(
        coefficients_df,
        x="Feature",
        y="Coefficient",
        color="Significance",
        color_discrete_map={
            "Significant": "#FF9900",
            "Not Significant": "#00B4D8"
        },
        hover_data=["P-value", "CI Lower", "CI Upper"]
    )

    fig_coeff.update_layout(
        template="plotly_dark",
        paper_bgcolor="#101827",
        plot_bgcolor="#172235",
        height=450,
        xaxis_title="Regression Feature",
        yaxis_title="Coefficient"
    )

    st.plotly_chart(fig_coeff, width="stretch")

    st.caption(
        "OLS regression uses HC3 heteroskedasticity-robust "
        "standard errors. Results update with category filters."
    )

else:
    st.warning("Insufficient data for regression analysis.")


st.divider()
st.subheader("Category Interaction Regression")

if (
    len(regression_df) >= 30
    and regression_df["category_grouped"].nunique() >= 2
):
    interaction_model = smf.ols(
        formula=(
            "rating ~ discount_percentage * C(category_grouped) "
            "+ log_price + log_rating_count"
        ),
        data=regression_df
    ).fit(cov_type="HC3")

    int_col1, int_col2, int_col3 = st.columns(3)

    with int_col1:
        st.metric(
            "Interaction Adjusted R²",
            f"{interaction_model.rsquared_adj:.4f}"
        )

    with int_col2:
        st.metric(
            "Interaction AIC",
            f"{interaction_model.aic:.2f}"
        )

    with int_col3:
        st.metric(
            "Change in Adjusted R²",
            f"{interaction_model.rsquared_adj - regression_model.rsquared_adj:+.4f}"
        )

    interaction_conf = interaction_model.conf_int()

    interaction_results = pd.DataFrame({
        "Feature": interaction_model.params.index,
        "Coefficient": interaction_model.params.values,
        "P-value": interaction_model.pvalues.values,
        "CI Lower": interaction_conf.iloc[:, 0].values,
        "CI Upper": interaction_conf.iloc[:, 1].values
    })

    interaction_results["Significance"] = interaction_results[
        "P-value"
    ].apply(
        lambda p: "Significant" if p < 0.05 else "Not Significant"
    )

    st.markdown("#### Interaction Model Coefficients")

    st.dataframe(
        interaction_results.round(6),
        width="stretch",
        hide_index=True
    )

    interaction_only = interaction_results[
        interaction_results["Feature"].str.contains(
            "discount_percentage:",
            regex=False
        )
    ]

    if not interaction_only.empty:
        fig_interaction = px.bar(
            interaction_only,
            x="Feature",
            y="Coefficient",
            color="Significance",
            color_discrete_map={
                "Significant": "#FF9900",
                "Not Significant": "#00B4D8"
            },
            hover_data=["P-value", "CI Lower", "CI Upper"]
        )

        fig_interaction.update_layout(
            template="plotly_dark",
            paper_bgcolor="#101827",
            plot_bgcolor="#172235",
            height=450,
            xaxis_title="Category Interaction",
            yaxis_title="Interaction Coefficient"
        )

        st.plotly_chart(fig_interaction, width="stretch")

    st.caption(
        "Interaction coefficients represent differences in discount "
        "slopes relative to the reference category. "
        "HC3 robust standard errors are used."
    )

else:
    st.warning(
        "Select at least two categories with sufficient observations "
        "for interaction regression."
    )


st.divider()
st.subheader("Influential Observations — Cook's Distance")

if len(regression_df) >= 30:

    influence = regression_model.get_influence()
    cooks_distance = influence.cooks_distance[0]

    cooks_threshold = 4 / len(regression_df)

    influence_df = regression_df.copy()
    influence_df["Cooks_Distance"] = cooks_distance
    influence_df["Influential"] = (
        influence_df["Cooks_Distance"] > cooks_threshold
    )

    influential_count = int(influence_df["Influential"].sum())

    cook_col1, cook_col2, cook_col3 = st.columns(3)

    with cook_col1:
        st.metric("Total Observations", f"{len(influence_df):,}")

    with cook_col2:
        st.metric("Cook's Threshold", f"{cooks_threshold:.6f}")

    with cook_col3:
        st.metric("Influential Observations", influential_count)

    fig_cooks = px.scatter(
        influence_df.reset_index(drop=True).reset_index(),
        x="index",
        y="Cooks_Distance",
        color="Influential",
        color_discrete_map={
            True: "#FF9900",
            False: "#00B4D8"
        },
        labels={
            "index": "Observation Index",
            "Cooks_Distance": "Cook's Distance"
        }
    )

    fig_cooks.add_hline(
        y=cooks_threshold,
        line_dash="dash",
        line_color="#F43F5E",
        annotation_text="Cook's Threshold"
    )

    fig_cooks.update_layout(
        template="plotly_dark",
        paper_bgcolor="#101827",
        plot_bgcolor="#172235",
        height=450
    )

    st.plotly_chart(fig_cooks, width="stretch")

else:
    st.warning("Insufficient observations for influence analysis.")


st.markdown("#### Regression After Removing Influential Observations")

if len(regression_df) >= 30:

    clean_regression_df = influence_df.loc[
        ~influence_df["Influential"]
    ].copy()

    if len(clean_regression_df) >= 20:

        clean_model = smf.ols(
            formula=(
                "rating ~ discount_percentage + log_price "
                "+ log_rating_count + C(category_grouped)"
            ),
            data=clean_regression_df
        ).fit(cov_type="HC3")

        comparison_df = pd.DataFrame({
            "Metric": [
                "Observations",
                "R²",
                "Adjusted R²",
                "AIC",
                "Discount Coefficient",
                "Discount p-value"
            ],
            "Original Model": [
                regression_model.nobs,
                regression_model.rsquared,
                regression_model.rsquared_adj,
                regression_model.aic,
                regression_model.params["discount_percentage"],
                regression_model.pvalues["discount_percentage"]
            ],
            "After Removal": [
                clean_model.nobs,
                clean_model.rsquared,
                clean_model.rsquared_adj,
                clean_model.aic,
                clean_model.params["discount_percentage"],
                clean_model.pvalues["discount_percentage"]
            ]
        })

        st.dataframe(
            comparison_df,
            width="stretch",
            hide_index=True
        )

        before_col, after_col = st.columns(2)

        with before_col:
            st.metric(
                "Original Adjusted R²",
                f"{regression_model.rsquared_adj:.4f}"
            )

        with after_col:
            st.metric(
                "After Removal Adjusted R²",
                f"{clean_model.rsquared_adj:.4f}",
                delta=f"{clean_model.rsquared_adj - regression_model.rsquared_adj:+.4f}"
            )

        st.caption(
            "Influential observations are identified using Cook's Distance "
            "greater than 4/n. The model is refitted using HC3 robust "
            "standard errors. Removing influential observations is a "
            "sensitivity analysis, not automatic evidence that those "
            "observations are invalid."
        )

    else:
        st.warning("Insufficient observations after filtering.")


st.divider()

st.divider()
st.subheader("Task 2 — Key Findings & Conclusion")

st.markdown("""
#### Key Findings

1. **Discount–Rating Relationship:** Pearson correlation was **-0.1622**, while Spearman correlation was **-0.1499**, indicating a weak negative association between discount percentage and product rating.

2. **Statistical Significance:** Both correlations were statistically significant, although their magnitudes were small.

3. **Baseline Regression:** The OLS model achieved an Adjusted R² of approximately **0.1163**, suggesting limited explanatory power.

4. **Discount Effect:** After controlling for price, rating count, and product category, the discount coefficient was approximately **-0.001273** (p ≈ 0.0027).

5. **Category Differences:** Home & Kitchen showed a stronger negative discount–rating relationship. The category interaction model achieved an Adjusted R² of approximately **0.1207**.

6. **Influential Observations:** Cook's Distance identified **71 influential observations**. Removing these increased Adjusted R² to approximately **0.1483**, while the discount coefficient remained negative.

#### Business Conclusion

Higher discounts are weakly associated with lower customer ratings in this dataset, even after accounting for selected product characteristics. However, the models explain only a limited proportion of rating variation. Discounting should therefore not be treated as a direct cause of customer dissatisfaction. Product quality, customer expectations, and other unobserved factors may also influence ratings.
""")

st.header("Task 3 — Review-Based Sentiment Analysis")

st.markdown("### Customer Review Sentiment Overview")

task3_df = pd.read_csv(
    Path(__file__).resolve().parent / "task_3_sentiment_analysis.csv"
)

task3_df["sentiment"] = task3_df["sentiment"].astype(str).str.strip().str.title()

sentiment_counts = task3_df["sentiment"].value_counts()

total_reviews = len(task3_df)
positive_reviews = int(sentiment_counts.get("Positive", 0))
negative_reviews = int(sentiment_counts.get("Negative", 0))
neutral_reviews = int(sentiment_counts.get("Neutral", 0))

t3_col1, t3_col2, t3_col3, t3_col4 = st.columns(4)

t3_col1.metric("Total Reviews", f"{total_reviews:,}")
t3_col2.metric("Positive Reviews", f"{positive_reviews:,}")
t3_col3.metric("Negative Reviews", f"{negative_reviews:,}")
t3_col4.metric("Neutral Reviews", f"{neutral_reviews:,}")

st.subheader("Sentiment Distribution")

sentiment_summary = pd.DataFrame({
    "Sentiment": ["Positive", "Negative", "Neutral"],
    "Count": [positive_reviews, negative_reviews, neutral_reviews]
})

sentiment_summary["Percentage"] = (
    sentiment_summary["Count"] / total_reviews * 100
).round(2)

t3_chart1, t3_chart2 = st.columns(2)

with t3_chart1:
    fig_t3_donut = px.pie(
        sentiment_summary,
        names="Sentiment",
        values="Count",
        hole=0.55,
        title="Customer Review Sentiment",
        color="Sentiment",
        color_discrete_map={
            "Positive": "#22C55E",
            "Negative": "#EF4444",
            "Neutral": "#F59E0B"
        }
    )
    fig_t3_donut.update_layout(template="plotly_dark")
    st.plotly_chart(fig_t3_donut, width="stretch")

with t3_chart2:
    fig_t3_bar = px.bar(
        sentiment_summary,
        x="Sentiment",
        y="Percentage",
        color="Sentiment",
        text="Percentage",
        title="Sentiment Percentage Comparison",
        color_discrete_map={
            "Positive": "#22C55E",
            "Negative": "#EF4444",
            "Neutral": "#F59E0B"
        }
    )
    fig_t3_bar.update_traces(texttemplate="%{text:.2f}%")
    fig_t3_bar.update_layout(
        template="plotly_dark",
        showlegend=False,
        yaxis_title="Percentage (%)"
    )
    st.plotly_chart(fig_t3_bar, width="stretch")

st.dataframe(sentiment_summary, width="stretch")


st.divider()
st.subheader("Category-wise Sentiment Analysis")

category_sentiment = pd.crosstab(
    task3_df["main_category"],
    task3_df["sentiment"]
)

category_sentiment = category_sentiment.reindex(
    columns=["Positive", "Negative", "Neutral"],
    fill_value=0
)

category_sentiment_long = (
    category_sentiment
    .reset_index()
    .melt(
        id_vars="main_category",
        var_name="Sentiment",
        value_name="Count"
    )
)

fig_t3_category = px.bar(
    category_sentiment_long,
    x="main_category",
    y="Count",
    color="Sentiment",
    barmode="group",
    title="Category-wise Sentiment Distribution",
    color_discrete_map={
        "Positive": "#22C55E",
        "Negative": "#EF4444",
        "Neutral": "#F59E0B"
    }
)

fig_t3_category.update_layout(
    template="plotly_dark",
    xaxis_title="Product Category",
    yaxis_title="Review Count"
)

st.plotly_chart(fig_t3_category, width="stretch")

category_sentiment_pct = (
    category_sentiment
    .div(category_sentiment.sum(axis=1), axis=0)
    .mul(100)
    .fillna(0)
    .round(2)
)

category_sentiment_pct_long = (
    category_sentiment_pct
    .reset_index()
    .melt(
        id_vars="main_category",
        var_name="Sentiment",
        value_name="Percentage"
    )
)

fig_t3_category_pct = px.bar(
    category_sentiment_pct_long,
    x="main_category",
    y="Percentage",
    color="Sentiment",
    barmode="stack",
    title="Category-wise Sentiment Percentage",
    color_discrete_map={
        "Positive": "#22C55E",
        "Negative": "#EF4444",
        "Neutral": "#F59E0B"
    }
)

fig_t3_category_pct.update_layout(
    template="plotly_dark",
    xaxis_title="Product Category",
    yaxis_title="Percentage (%)"
)

st.plotly_chart(fig_t3_category_pct, width="stretch")

st.dataframe(
    category_sentiment.assign(
        Total=category_sentiment.sum(axis=1)
    ),
    width="stretch"
)


st.divider()
st.subheader("Rating vs Sentiment Analysis")

task3_rating_df = task3_df.copy()
task3_rating_df["rating"] = pd.to_numeric(
    task3_rating_df["rating"], errors="coerce"
)
task3_rating_df["sentiment_score"] = pd.to_numeric(
    task3_rating_df["sentiment_score"], errors="coerce"
)

task3_rating_df = task3_rating_df.dropna(
    subset=["rating", "sentiment_score"]
)

fig_t3_rating = px.box(
    task3_rating_df,
    x="sentiment",
    y="rating",
    color="sentiment",
    category_orders={
        "sentiment": ["Positive", "Neutral", "Negative"]
    },
    title="Product Rating Distribution by Review Sentiment",
    color_discrete_map={
        "Positive": "#22C55E",
        "Negative": "#EF4444",
        "Neutral": "#F59E0B"
    }
)

fig_t3_rating.update_layout(
    template="plotly_dark",
    xaxis_title="Review Sentiment",
    yaxis_title="Product Rating",
    showlegend=False
)

st.plotly_chart(fig_t3_rating, width="stretch")

fig_t3_scatter = px.scatter(
    task3_rating_df,
    x="rating",
    y="sentiment_score",
    color="sentiment",
    hover_data=["product_name"],
    title="Product Rating vs Sentiment Score",
    color_discrete_map={
        "Positive": "#22C55E",
        "Negative": "#EF4444",
        "Neutral": "#F59E0B"
    }
)

fig_t3_scatter.update_layout(
    template="plotly_dark",
    xaxis_title="Product Rating",
    yaxis_title="Sentiment Score"
)

st.plotly_chart(fig_t3_scatter, width="stretch")

st.divider()
st.subheader("Rating and Sentiment Mismatch Detection")

high_rating_negative = task3_rating_df[
    (task3_rating_df["rating"] >= 4.0) &
    (task3_rating_df["sentiment"] == "Negative")
].copy()

low_rating_positive = task3_rating_df[
    (task3_rating_df["rating"] <= 3.5) &
    (task3_rating_df["sentiment"] == "Positive")
].copy()

mismatch_col1, mismatch_col2 = st.columns(2)

mismatch_col1.metric(
    "High-Rated Negative Sentiment",
    len(high_rating_negative)
)

mismatch_col2.metric(
    "Low-Rated Positive Sentiment",
    len(low_rating_positive)
)

st.caption(
    "Mismatch thresholds: rating >= 4.0 with Negative sentiment, "
    "or rating <= 3.5 with Positive sentiment. "
    "These are exploratory rules, not confirmed data errors."
)

st.markdown("#### High-Rated Products with Negative Sentiment")

st.dataframe(
    high_rating_negative[
        ["product_name", "main_category", "rating",
         "sentiment_score", "sentiment"]
    ].sort_values("rating", ascending=False),
    width="stretch"
)

st.markdown("#### Low-Rated Products with Positive Sentiment")

st.dataframe(
    low_rating_positive[
        ["product_name", "main_category", "rating",
         "sentiment_score", "sentiment"]
    ].sort_values("rating", ascending=True),
    width="stretch"
)


st.divider()
st.subheader("Top Positive & Negative Products")

top_positive = (
    task3_rating_df[
        task3_rating_df["sentiment"] == "Positive"
    ]
    .sort_values("sentiment_score", ascending=False)
    .head(10)
)

top_negative = (
    task3_rating_df[
        task3_rating_df["sentiment"] == "Negative"
    ]
    .sort_values("sentiment_score", ascending=True)
    .head(10)
)

top_col1, top_col2 = st.columns(2)

with top_col1:
    st.markdown("#### Top 10 Positive Products")
    st.dataframe(
        top_positive[
            ["product_name", "main_category", "rating", "sentiment_score"]
        ],
        width="stretch",
        hide_index=True
    )

with top_col2:
    st.markdown("#### Top 10 Negative Products")
    st.dataframe(
        top_negative[
            ["product_name", "main_category", "rating", "sentiment_score"]
        ],
        width="stretch",
        hide_index=True
    )

top_comparison = pd.concat([
    top_positive.assign(Sentiment="Positive"),
    top_negative.assign(Sentiment="Negative")
])

fig_t3_top = px.bar(
    top_comparison,
    x="sentiment_score",
    y="product_name",
    color="Sentiment",
    orientation="h",
    hover_data=["rating", "main_category"],
    title="Top Positive and Negative Product Sentiment Scores",
    color_discrete_map={
        "Positive": "#22C55E",
        "Negative": "#EF4444"
    }
)

fig_t3_top.update_layout(
    template="plotly_dark",
    height=750,
    xaxis_title="Sentiment Score",
    yaxis_title="Product",
    yaxis={"automargin": True}
)

st.plotly_chart(fig_t3_top, width="stretch")

st.caption(
    "Products are ranked by their available review sentiment scores. "
    "Tied scores may occur."
)


st.divider()
st.subheader("Task 3 — Key Findings & Conclusion")

positive_pct = positive_reviews / total_reviews * 100
negative_pct = negative_reviews / total_reviews * 100
neutral_pct = neutral_reviews / total_reviews * 100

category_positive_pct = (
    category_sentiment_pct["Positive"]
    .sort_values(ascending=False)
)

highest_positive_category = category_positive_pct.index[0]
highest_positive_percentage = category_positive_pct.iloc[0]

valid_rating_sentiment = task3_rating_df[
    task3_rating_df["sentiment"].isin(["Positive", "Negative"])
]

rating_sentiment_corr = valid_rating_sentiment[
    ["rating", "sentiment_score"]
].corr().iloc[0, 1]

total_mismatches = (
    len(high_rating_negative) + len(low_rating_positive)
)

st.markdown("#### Sentiment Analysis Summary")

st.write(
    f"Out of {total_reviews:,} product review records, "
    f"{positive_reviews:,} ({positive_pct:.2f}%) are Positive, "
    f"{negative_reviews:,} ({negative_pct:.2f}%) are Negative, "
    f"and {neutral_reviews:,} ({neutral_pct:.2f}%) are Neutral."
)

st.write(
    f"The category with the highest positive sentiment percentage is "
    f"{highest_positive_category}, at {highest_positive_percentage:.2f}%."
)

st.write(
    f"The Pearson correlation between product rating and sentiment score "
    f"is {rating_sentiment_corr:.4f}, based on records with "
    f"Positive or Negative sentiment and valid numeric values."
)

st.write(
    f"Using the selected rating thresholds, {total_mismatches} "
    f"rating–sentiment mismatches were identified, including "
    f"{len(high_rating_negative)} high-rated Negative cases and "
    f"{len(low_rating_positive)} low-rated Positive cases."
)

st.markdown("#### Business Interpretation")

st.info(
    "Review sentiment provides an additional perspective on customer "
    "experience beyond aggregate product ratings. Categories with "
    "relatively higher negative sentiment may warrant closer review. "
    "Rating–sentiment mismatches are useful investigation candidates, "
    "but do not necessarily indicate incorrect ratings or data errors."
)

st.success(
    "Task 3 Completed: Sentiment distribution, category comparison, "
    "rating relationships, mismatch detection, product rankings, "
    "and automated insights are integrated."
)


st.divider()
st.header("Task 4 — Product Ranking")

st.markdown("### Product Ranking Overview")

task4_df = pd.read_csv(
    Path(__file__).resolve().parent / "task_4_product_ranking.csv"
)

task4_df["rank"] = pd.to_numeric(task4_df["rank"], errors="coerce")
task4_df["new_rank"] = pd.to_numeric(task4_df["new_rank"], errors="coerce")
task4_df["ranking_score"] = pd.to_numeric(
    task4_df["ranking_score"], errors="coerce"
)
task4_df["new_ranking_score"] = pd.to_numeric(
    task4_df["new_ranking_score"], errors="coerce"
)

total_products_t4 = len(task4_df)
ranking_changes = int(
    (task4_df["rank"] != task4_df["new_rank"]).sum()
)
ranking_change_pct = ranking_changes / total_products_t4 * 100

t4_col1, t4_col2, t4_col3, t4_col4 = st.columns(4)

t4_col1.metric("Total Products", f"{total_products_t4:,}")
t4_col2.metric("Ranking Changes", f"{ranking_changes:,}")
t4_col3.metric("Ranking Change %", f"{ranking_change_pct:.2f}%")
t4_col4.metric(
    "Average Product Rating",
    f'{task4_df["rating"].mean():.2f}'
)

st.subheader("Top 10 Products — Original vs Updated Ranking")

top_original_t4 = task4_df.nsmallest(10, "rank").copy()
top_updated_t4 = task4_df.nsmallest(10, "new_rank").copy()

t4_chart_col1, t4_chart_col2 = st.columns(2)

with t4_chart_col1:
    fig_t4_original = px.bar(
        top_original_t4.sort_values("ranking_score"),
        x="ranking_score",
        y="product_name",
        orientation="h",
        color="ranking_score",
        color_continuous_scale="Blues",
        title="Top 10 — Original Ranking",
        hover_data=["rating", "rating_count", "rank"]
    )
    fig_t4_original.update_layout(
        template="plotly_dark",
        height=600,
        yaxis_title="Product",
        xaxis_title="Original Ranking Score",
        yaxis={"automargin": True}
    )
    st.plotly_chart(fig_t4_original, width="stretch")

with t4_chart_col2:
    fig_t4_updated = px.bar(
        top_updated_t4.sort_values("new_ranking_score"),
        x="new_ranking_score",
        y="product_name",
        orientation="h",
        color="new_ranking_score",
        color_continuous_scale="Oranges",
        title="Top 10 — Updated Ranking",
        hover_data=["rating", "rating_count", "new_rank"]
    )
    fig_t4_updated.update_layout(
        template="plotly_dark",
        height=600,
        yaxis_title="Product",
        xaxis_title="Updated Ranking Score",
        yaxis={"automargin": True}
    )
    st.plotly_chart(fig_t4_updated, width="stretch")

st.markdown("#### Top 10 Products — Updated Ranking")

st.dataframe(
    top_updated_t4[
        [
            "product_name",
            "category",
            "rating",
            "rating_count",
            "weighted_rating",
            "ranking_score",
            "rank",
            "new_ranking_score",
            "new_rank"
        ]
    ].sort_values("new_rank"),
    width="stretch",
    hide_index=True
)


st.divider()
st.subheader("Task 4 — Category-wise Ranking Analysis")

task4_category_df = task4_df.copy()

task4_category_df["main_category"] = (
    task4_category_df["category"]
    .astype(str)
    .str.split("|", regex=False)
    .str[0]
    .str.strip()
)

task4_category_summary = (
    task4_category_df
    .groupby("main_category")
    .agg(
        Total_Products=("product_id", "count"),
        Average_Rating=("rating", "mean"),
        Average_Weighted_Rating=("weighted_rating", "mean"),
        Average_Updated_Rank=("new_rank", "mean")
    )
    .reset_index()
)

fig_t4_category = px.bar(
    task4_category_summary.sort_values("Average_Weighted_Rating"),
    x="Average_Weighted_Rating",
    y="main_category",
    orientation="h",
    color="Total_Products",
    title="Average Weighted Rating by Category",
    color_continuous_scale="Blues"
)

fig_t4_category.update_layout(
    template="plotly_dark",
    height=500,
    xaxis_title="Average Weighted Rating",
    yaxis_title="Category"
)

st.plotly_chart(fig_t4_category, width="stretch")

st.dataframe(
    task4_category_summary.round(3),
    width="stretch",
    hide_index=True
)

st.divider()
st.subheader("Task 4 — Original vs Updated Rank Movement")

task4_movement_df = task4_df.copy()

task4_movement_df["Rank_Change"] = (
    task4_movement_df["rank"] -
    task4_movement_df["new_rank"]
)

task4_movement_df["Movement"] = task4_movement_df["Rank_Change"].apply(
    lambda x: "Improved" if x > 0
    else "Declined" if x < 0
    else "Unchanged"
)

movement_counts = (
    task4_movement_df["Movement"]
    .value_counts()
    .reset_index()
)

movement_counts.columns = ["Movement", "Products"]

t4_move_col1, t4_move_col2 = st.columns(2)

with t4_move_col1:
    fig_t4_movement = px.pie(
        movement_counts,
        names="Movement",
        values="Products",
        hole=0.5,
        title="Product Rank Movement Distribution",
        color="Movement",
        color_discrete_map={
            "Improved": "#22C55E",
            "Declined": "#EF4444",
            "Unchanged": "#F59E0B"
        }
    )
    fig_t4_movement.update_layout(template="plotly_dark")
    st.plotly_chart(fig_t4_movement, width="stretch")

with t4_move_col2:
    fig_t4_rank_scatter = px.scatter(
        task4_movement_df,
        x="rank",
        y="new_rank",
        color="Movement",
        hover_data=["product_name", "rating", "rating_count"],
        title="Original Rank vs Updated Rank",
        color_discrete_map={
            "Improved": "#22C55E",
            "Declined": "#EF4444",
            "Unchanged": "#F59E0B"
        }
    )

    fig_t4_rank_scatter.update_layout(
        template="plotly_dark",
        xaxis_title="Original Rank",
        yaxis_title="Updated Rank"
    )

    st.plotly_chart(fig_t4_rank_scatter, width="stretch")

st.markdown("#### Biggest Rank Improvements")

st.dataframe(
    task4_movement_df.nlargest(10, "Rank_Change")[
        [
            "product_name",
            "rating",
            "rating_count",
            "rank",
            "new_rank",
            "Rank_Change"
        ]
    ],
    width="stretch",
    hide_index=True
)

st.markdown("#### Biggest Rank Declines")

st.dataframe(
    task4_movement_df.nsmallest(10, "Rank_Change")[
        [
            "product_name",
            "rating",
            "rating_count",
            "rank",
            "new_rank",
            "Rank_Change"
        ]
    ],
    width="stretch",
    hide_index=True
)


st.divider()
st.subheader("Task 4 — Ranking Methodology")

st.markdown("""
The product ranking analysis uses the original Task 4 dataset,
which contains weighted ratings, original ranking scores,
updated ranking scores, and their corresponding ranks.

The dashboard compares the saved ranking results without
recalculating or modifying the original scoring methodology.
""")

st.divider()
st.subheader("Task 4 — Key Findings & Conclusion")

total_t4 = len(task4_movement_df)

improved_t4 = int(
    (task4_movement_df["Rank_Change"] > 0).sum()
)

declined_t4 = int(
    (task4_movement_df["Rank_Change"] < 0).sum()
)

unchanged_t4 = int(
    (task4_movement_df["Rank_Change"] == 0).sum()
)

changed_t4 = improved_t4 + declined_t4

rank_correlation_t4 = (
    task4_movement_df[["rank", "new_rank"]]
    .corr(method="spearman")
    .iloc[0, 1]
)

top10_original_ids = set(
    task4_movement_df.nsmallest(10, "rank")["product_id"]
)

top10_updated_ids = set(
    task4_movement_df.nsmallest(10, "new_rank")["product_id"]
)

top10_overlap = len(
    top10_original_ids.intersection(top10_updated_ids)
)

best_updated_product = (
    task4_movement_df
    .sort_values("new_rank")
    .iloc[0]
)

st.markdown("#### Key Findings")

st.write(
    f"**1. Ranking Changes:** {changed_t4:,} out of "
    f"{total_t4:,} products changed their positions "
    f"({changed_t4 / total_t4 * 100:.2f}%)."
)

st.write(
    f"**2. Rank Movement:** {improved_t4:,} products improved, "
    f"{declined_t4:,} declined, and "
    f"{unchanged_t4:,} remained unchanged."
)

st.write(
    f"**3. Ranking Agreement:** Spearman rank correlation "
    f"between original and updated ranks is "
    f"{rank_correlation_t4:.4f}."
)

st.write(
    f"**4. Top 10 Stability:** {top10_overlap} products "
    f"appear in the Top 10 under both ranking methods."
)

st.write(
    f"**5. Highest Updated Rank:** "
    f"{best_updated_product['product_name']} "
    f"holds the best updated rank of "
    f"{best_updated_product['new_rank']:.0f}."
)

st.markdown("#### Business Interpretation")

st.info(
    "The original and updated ranking methods produce different "
    "product priorities. Comparing rank movements helps identify "
    "products that gain or lose visibility under the revised "
    "scoring approach. Ranking results can support product "
    "shortlisting and merchandising decisions, but should be "
    "interpreted alongside customer demand, product quality, "
    "and business performance."
)

st.markdown("#### Final Conclusion")

st.success(
    "Task 4 Completed: Product ranking overview, original versus "
    "updated Top 10 comparison, category-wise ranking analysis, "
    "rank movement visualization, and automated findings "
    "have been integrated successfully."
)


st.divider()
st.markdown('<div id="task-5-anomaly-outlier-detection"></div>', unsafe_allow_html=True)
st.header("Task 5 — Anomaly & Outlier Detection")

st.markdown("### Anomaly Detection Overview")

task5_df = pd.read_csv(
    Path(__file__).resolve().parent / "task_5_final_anomaly_report.csv"
)

task5_df["Anomaly_Type"] = (
    task5_df["Anomaly_Type"].astype(str).str.strip()
)

task5_counts = task5_df["Anomaly_Type"].value_counts()

task5_total = len(task5_df)
task5_iqr_only = int(task5_counts.get("IQR Only", 0))
task5_both = int(task5_counts.get("Both Methods", 0))

task5_shared = (
    task5_df["Shared_Review_Flag"]
    .astype(str)
    .str.lower()
    .eq("true")
    .sum()
)

task5_col1, task5_col2, task5_col3, task5_col4 = st.columns(4)

task5_col1.metric("Flagged Records", f"{task5_total:,}")
task5_col2.metric("IQR Only", f"{task5_iqr_only:,}")
task5_col3.metric("Both Methods", f"{task5_both:,}")
task5_col4.metric("Shared Review Flags", f"{task5_shared:,}")

st.caption(
    "These metrics describe the saved anomaly report. "
    "Shared review flags are not additional anomaly records."
)

st.subheader("Anomaly Detection Method Comparison")

task5_method_summary = (
    task5_df["Anomaly_Type"]
    .value_counts()
    .rename_axis("Anomaly Type")
    .reset_index(name="Products")
)

task5_method_summary["Percentage"] = (
    task5_method_summary["Products"] / task5_total * 100
).round(2)

task5_chart1, task5_chart2 = st.columns(2)

with task5_chart1:
    fig_t5_donut = px.pie(
        task5_method_summary,
        names="Anomaly Type",
        values="Products",
        hole=0.55,
        title="Anomaly Classification Distribution",
        color="Anomaly Type",
        color_discrete_map={
            "IQR Only": "#F59E0B",
            "Both Methods": "#EF4444"
        }
    )

    fig_t5_donut.update_layout(template="plotly_dark")

    st.plotly_chart(
        fig_t5_donut,
        width="stretch"
    )

with task5_chart2:
    fig_t5_bar = px.bar(
        task5_method_summary,
        x="Anomaly Type",
        y="Products",
        color="Anomaly Type",
        text="Products",
        title="IQR Only vs Both Methods",
        color_discrete_map={
            "IQR Only": "#F59E0B",
            "Both Methods": "#EF4444"
        }
    )

    fig_t5_bar.update_layout(
        template="plotly_dark",
        showlegend=False,
        yaxis_title="Flagged Records"
    )

    st.plotly_chart(
        fig_t5_bar,
        width="stretch"
    )

st.markdown("#### Anomaly Detection Summary")

st.dataframe(
    task5_method_summary,
    width="stretch",
    hide_index=True
)


st.divider()
st.subheader("Task 5 — Price, Discount & Rating Analysis")

task5_analysis_df = task5_df.copy()

task5_numeric_cols = [
    "actual_price",
    "discounted_price",
    "discount_percentage",
    "rating",
    "rating_count",
    "IF_Anomaly_Score"
]

for col in task5_numeric_cols:
    task5_analysis_df[col] = pd.to_numeric(
        task5_analysis_df[col],
        errors="coerce"
    )

t5_price_col, t5_discount_col = st.columns(2)

with t5_price_col:
    fig_t5_price = px.box(
        task5_analysis_df,
        x="Anomaly_Type",
        y="discounted_price",
        color="Anomaly_Type",
        points="outliers",
        title="Discounted Price Distribution by Anomaly Type",
        color_discrete_map={
            "IQR Only": "#F59E0B",
            "Both Methods": "#EF4444"
        }
    )

    fig_t5_price.update_layout(
        template="plotly_dark",
        xaxis_title="Anomaly Classification",
        yaxis_title="Discounted Price (INR)",
        showlegend=False
    )

    st.plotly_chart(fig_t5_price, width="stretch")

with t5_discount_col:
    fig_t5_discount = px.scatter(
        task5_analysis_df,
        x="discount_percentage",
        y="rating",
        color="Anomaly_Type",
        hover_data=["product_name", "discounted_price"],
        title="Discount Percentage vs Product Rating",
        color_discrete_map={
            "IQR Only": "#F59E0B",
            "Both Methods": "#EF4444"
        }
    )

    fig_t5_discount.update_layout(
        template="plotly_dark",
        xaxis_title="Discount Percentage (%)",
        yaxis_title="Customer Rating"
    )

    st.plotly_chart(fig_t5_discount, width="stretch")

st.divider()
st.subheader("Task 5 — Category-wise Anomaly Analysis")

task5_analysis_df["main_category"] = (
    task5_analysis_df["category"]
    .astype(str)
    .str.split("|", regex=False)
    .str[0]
    .str.strip()
)

task5_category_summary = (
    task5_analysis_df
    .groupby(["main_category", "Anomaly_Type"])
    .agg(
        Products=("product_name", "size"),
        Average_Price=("discounted_price", "mean"),
        Average_Discount=("discount_percentage", "mean"),
        Average_Rating=("rating", "mean")
    )
    .reset_index()
)

fig_t5_category = px.bar(
    task5_category_summary,
    x="main_category",
    y="Products",
    color="Anomaly_Type",
    barmode="group",
    title="Category-wise Anomaly Classification",
    color_discrete_map={
        "IQR Only": "#F59E0B",
        "Both Methods": "#EF4444"
    }
)

fig_t5_category.update_layout(
    template="plotly_dark",
    xaxis_title="Product Category",
    yaxis_title="Flagged Records"
)

st.plotly_chart(fig_t5_category, width="stretch")

st.dataframe(
    task5_category_summary.round(2),
    width="stretch",
    hide_index=True
)

st.divider()
st.subheader("Task 5 — Isolation Forest Score Distribution")

fig_t5_if = px.histogram(
    task5_analysis_df,
    x="IF_Anomaly_Score",
    color="Anomaly_Type",
    nbins=35,
    barmode="overlay",
    title="Isolation Forest Scores of Flagged Records",
    color_discrete_map={
        "IQR Only": "#F59E0B",
        "Both Methods": "#EF4444"
    }
)

fig_t5_if.update_layout(
    template="plotly_dark",
    xaxis_title="Saved Isolation Forest Score",
    yaxis_title="Number of Records"
)

st.plotly_chart(fig_t5_if, width="stretch")

st.caption(
    "These visualizations cover the 350 records in the saved "
    "anomaly report, not the entire Amazon product dataset. "
    "The Isolation Forest scores are displayed as saved, "
    "without assuming a particular scoring convention."
)


st.divider()
st.subheader("Task 5 — Business Classification & Investigation")

task5_business_df = task5_analysis_df.copy()

task5_business_df["Business_Classification"] = (
    task5_business_df["Business_Classification"]
    .fillna("Not Classified")
    .astype(str)
    .str.strip()
)

task5_business_df["Investigation_Decision"] = (
    task5_business_df["Investigation_Decision"]
    .fillna("Not Specified")
    .astype(str)
    .str.strip()
)

business_summary = (
    task5_business_df["Business_Classification"]
    .value_counts()
    .rename_axis("Business Classification")
    .reset_index(name="Products")
)

decision_summary = (
    task5_business_df["Investigation_Decision"]
    .value_counts()
    .rename_axis("Investigation Decision")
    .reset_index(name="Products")
)

t5_business_col1, t5_business_col2 = st.columns(2)

with t5_business_col1:
    fig_t5_business = px.bar(
        business_summary,
        x="Products",
        y="Business Classification",
        orientation="h",
        color="Products",
        title="Business Classification of Anomalies",
        color_continuous_scale="Blues"
    )

    fig_t5_business.update_layout(
        template="plotly_dark",
        yaxis={"categoryorder": "total ascending"},
        height=450
    )

    st.plotly_chart(fig_t5_business, width="stretch")

with t5_business_col2:
    fig_t5_decision = px.pie(
        decision_summary,
        names="Investigation Decision",
        values="Products",
        hole=0.5,
        title="Investigation Decision Distribution"
    )

    fig_t5_decision.update_layout(
        template="plotly_dark",
        height=450
    )

    st.plotly_chart(fig_t5_decision, width="stretch")

st.divider()
st.subheader("Task 5 — Shared Review Flag Analysis")

task5_business_df["Shared_Review_Label"] = (
    task5_business_df["Shared_Review_Flag"]
    .astype(str)
    .str.strip()
    .str.lower()
    .map({
        "true": "Shared Review",
        "false": "No Shared Review"
    })
    .fillna("Unknown")
)

shared_review_summary = (
    task5_business_df
    .groupby(["Anomaly_Type", "Shared_Review_Label"])
    .size()
    .reset_index(name="Products")
)

fig_t5_shared = px.bar(
    shared_review_summary,
    x="Anomaly_Type",
    y="Products",
    color="Shared_Review_Label",
    barmode="group",
    title="Shared Review Flags by Anomaly Type"
)

fig_t5_shared.update_layout(
    template="plotly_dark",
    xaxis_title="Anomaly Classification",
    yaxis_title="Flagged Records"
)

st.plotly_chart(fig_t5_shared, width="stretch")

st.divider()
st.subheader("Task 5 — Top Anomalous Products")

st.caption(
    "Records are sorted by their saved Isolation Forest scores. "
    "Score direction is not assumed to indicate anomaly severity."
)

task5_display_df = task5_business_df[
    [
        "product_name",
        "Anomaly_Type",
        "IF_Anomaly_Score",
        "discounted_price",
        "discount_percentage",
        "rating",
        "rating_count",
        "Business_Classification",
        "Investigation_Decision",
        "Shared_Review_Label"
    ]
].sort_values("IF_Anomaly_Score", ascending=True)

task5_search = st.text_input(
    "Search anomalous products",
    key="task5_product_search"
)

if task5_search:
    task5_display_df = task5_display_df[
        task5_display_df["product_name"]
        .astype(str)
        .str.contains(task5_search, case=False, na=False, regex=False)
    ]

st.dataframe(
    task5_display_df,
    width="stretch",
    hide_index=True
)

st.markdown("#### Business Classification vs Investigation Decision")

task5_investigation_matrix = pd.crosstab(
    task5_business_df["Business_Classification"],
    task5_business_df["Investigation_Decision"]
)

st.dataframe(
    task5_investigation_matrix,
    width="stretch"
)


st.divider()
st.subheader("Task 5 — Key Findings & Conclusion")

t5_total = len(task5_business_df)

t5_iqr = int(
    (task5_business_df["Anomaly_Type"] == "IQR Only").sum()
)

t5_both = int(
    (task5_business_df["Anomaly_Type"] == "Both Methods").sum()
)

t5_shared = int(
    (task5_business_df["Shared_Review_Label"] == "Shared Review").sum()
)

t5_iqr_pct = t5_iqr / t5_total * 100
t5_both_pct = t5_both / t5_total * 100
t5_shared_pct = t5_shared / t5_total * 100

t5_top_classification = business_summary.iloc[0]
t5_top_decision = decision_summary.iloc[0]

t5_category_counts = (
    task5_business_df["main_category"]
    .value_counts()
)

t5_top_category = t5_category_counts.index[0]
t5_top_category_count = int(t5_category_counts.iloc[0])

st.markdown("#### Key Findings")

st.write(
    f"**1. Anomaly Detection:** The final report contains "
    f"{t5_total:,} flagged records. Of these, "
    f"{t5_iqr:,} ({t5_iqr_pct:.2f}%) were detected by IQR only, "
    f"while {t5_both:,} ({t5_both_pct:.2f}%) "
    f"were detected by both IQR and Isolation Forest."
)

st.write(
    f"**2. Shared Review Flags:** "
    f"{t5_shared:,} records ({t5_shared_pct:.2f}%) "
    f"have a shared-review flag."
)

st.write(
    f"**3. Most Represented Category:** "
    f"{t5_top_category} contains "
    f"{t5_top_category_count:,} flagged records, "
    f"the highest count in the saved anomaly report."
)

st.write(
    f"**4. Business Classification:** "
    f"The most frequent classification is "
    f"{t5_top_classification['Business Classification']}, "
    f"with {int(t5_top_classification['Products']):,} records."
)

st.write(
    f"**5. Investigation Decision:** "
    f"The most frequent investigation decision is "
    f"{t5_top_decision['Investigation Decision']}, "
    f"covering {int(t5_top_decision['Products']):,} records."
)

st.markdown("#### Business Interpretation")

st.info(
    "Combining statistical anomaly detection with business "
    "classification helps identify unusual products for further "
    "investigation. Records flagged by both methods provide "
    "cross-method agreement, while shared-review flags can "
    "highlight potential review-pattern concerns. However, "
    "statistical anomalies do not automatically indicate fraud, "
    "incorrect pricing, or poor product quality."
)

st.markdown("#### Final Conclusion")

st.success(
    "Task 5 Completed: Anomaly detection comparison, price and "
    "discount analysis, category-wise anomaly distribution, "
    "Isolation Forest score visualization, shared-review "
    "analysis, business classifications, investigation decisions, "
    "and automated findings have been integrated successfully."
)

st.caption(
    "All percentages in this section are calculated using "
    "the 350 records in the saved final anomaly report. "
    "They are not overall anomaly rates for the full dataset."
)


st.divider()
st.header("Task 6 — Predictive Rating Analysis")

st.markdown("### Model Performance & Cross-Validation")

task6_cv = pd.read_csv(
    Path(__file__).resolve().parent / "task6_cross_validation.csv"
)

task6_cv = task6_cv.sort_values(
    "CV_MAE",
    ascending=True
).reset_index(drop=True)

best_model = task6_cv.iloc[0]

t6_col1, t6_col2, t6_col3, t6_col4 = st.columns(4)

t6_col1.metric(
    "Models Evaluated",
    len(task6_cv)
)

t6_col2.metric(
    "Best Model",
    best_model["Model"]
)

t6_col3.metric(
    "Best CV MAE",
    f"{best_model['CV_MAE']:.4f}"
)

t6_col4.metric(
    "Best CV R²",
    f"{best_model['CV_R2']:.4f}"
)

st.markdown("#### Model Comparison")

t6_chart1, t6_chart2 = st.columns(2)

with t6_chart1:
    fig_t6_mae = px.bar(
        task6_cv,
        x="Model",
        y="CV_MAE",
        color="Model",
        text="CV_MAE",
        title="Cross-Validation Mean Absolute Error"
    )

    fig_t6_mae.update_traces(
        texttemplate="%{text:.4f}",
        textposition="outside"
    )

    fig_t6_mae.update_layout(
        template="plotly_dark",
        showlegend=False,
        yaxis_title="MAE (Lower is Better)"
    )

    st.plotly_chart(
        fig_t6_mae,
        width="stretch"
    )

with t6_chart2:
    fig_t6_rmse = px.bar(
        task6_cv,
        x="Model",
        y="CV_RMSE",
        color="Model",
        text="CV_RMSE",
        title="Cross-Validation Root Mean Squared Error"
    )

    fig_t6_rmse.update_traces(
        texttemplate="%{text:.4f}",
        textposition="outside"
    )

    fig_t6_rmse.update_layout(
        template="plotly_dark",
        showlegend=False,
        yaxis_title="RMSE (Lower is Better)"
    )

    st.plotly_chart(
        fig_t6_rmse,
        width="stretch"
    )

fig_t6_r2 = px.bar(
    task6_cv,
    x="Model",
    y="CV_R2",
    color="Model",
    text="CV_R2",
    title="Cross-Validation R² Comparison"
)

fig_t6_r2.update_traces(
    texttemplate="%{text:.4f}",
    textposition="outside"
)

fig_t6_r2.update_layout(
    template="plotly_dark",
    showlegend=False,
    yaxis_title="R² (Higher is Better)"
)

st.plotly_chart(
    fig_t6_r2,
    width="stretch"
)

st.markdown("#### Cross-Validation Results Table")

st.dataframe(
    task6_cv.style.format({
        "CV_MAE": "{:.4f}",
        "CV_RMSE": "{:.4f}",
        "CV_R2": "{:.4f}"
    }),
    width="stretch",
    hide_index=True
)

st.info(
    f"Random Forest achieved the best cross-validation "
    f"performance with MAE = {best_model['CV_MAE']:.4f}, "
    f"RMSE = {best_model['CV_RMSE']:.4f}, "
    f"and R² = {best_model['CV_R2']:.4f}. "
    f"However, the relatively low R² indicates that "
    f"the available features explain only a limited "
    f"proportion of rating variation."
)


st.divider()
st.subheader("Task 6 — Feature Importance Analysis")

task6_features = pd.read_csv(
    Path(__file__).resolve().parent / "task6_feature_importance.csv"
)

task6_features["Importance_Mean"] = pd.to_numeric(
    task6_features["Importance_Mean"],
    errors="coerce"
)

task6_features["Importance_Std"] = pd.to_numeric(
    task6_features["Importance_Std"],
    errors="coerce"
)

task6_features = (
    task6_features
    .dropna(subset=["Feature", "Importance_Mean"])
    .sort_values("Importance_Mean", ascending=False)
    .reset_index(drop=True)
)

task6_features["Rank"] = range(1, len(task6_features) + 1)

st.markdown("#### Feature Importance Ranking")

fig_t6_importance = px.bar(
    task6_features.sort_values("Importance_Mean"),
    x="Importance_Mean",
    y="Feature",
    orientation="h",
    color="Importance_Mean",
    color_continuous_scale="Blues",
    error_x="Importance_Std",
    title="Feature Importance with Standard Deviation",
    hover_data={
        "Importance_Mean": ":.6f",
        "Importance_Std": ":.6f"
    }
)

fig_t6_importance.update_layout(
    template="plotly_dark",
    xaxis_title="Mean Importance",
    yaxis_title="Feature",
    coloraxis_showscale=False,
    height=420
)

st.plotly_chart(
    fig_t6_importance,
    width="stretch"
)

st.markdown("#### Feature Importance Table")

st.dataframe(
    task6_features[
        ["Rank", "Feature", "Importance_Mean", "Importance_Std"]
    ].style.format({
        "Importance_Mean": "{:.6f}",
        "Importance_Std": "{:.6f}"
    }),
    width="stretch",
    hide_index=True
)

st.markdown("#### Key Findings")

top_feature = task6_features.iloc[0]
least_feature = task6_features.iloc[-1]

st.write(
    f"**1. Highest Importance:** "
    f"{top_feature['Feature']} has the highest mean importance "
    f"({top_feature['Importance_Mean']:.6f})."
)

st.write(
    f"**2. Lowest Importance:** "
    f"{least_feature['Feature']} has the lowest mean importance "
    f"({least_feature['Importance_Mean']:.6f}) "
    f"among the evaluated features."
)

st.write(
    "**3. Variability:** Error bars represent the saved standard "
    "deviation of importance estimates. Overlapping error bars "
    "mean small differences in ranking should be interpreted cautiously."
)

st.info(
    "Feature importance measures how useful each feature is "
    "for the evaluated prediction model. It does not establish "
    "causation or indicate whether a feature increases or "
    "decreases product ratings. The saved report should be "
    "interpreted alongside cross-validation performance."
)


st.divider()
st.subheader("Task 6 — Prediction Error Analysis")

task6_errors = pd.read_csv(
    Path(__file__).resolve().parent / "task6_prediction_errors.csv"
)

t6_numeric_cols = [
    "Actual_Rating",
    "Predicted_Rating",
    "Error",
    "Absolute_Error"
]

for col in t6_numeric_cols:
    task6_errors[col] = pd.to_numeric(
        task6_errors[col],
        errors="coerce"
    )

task6_errors = task6_errors.dropna(
    subset=t6_numeric_cols
).copy()

t6_n = len(task6_errors)
t6_mae = task6_errors["Absolute_Error"].mean()
t6_rmse = (
    (task6_errors["Error"] ** 2).mean()
) ** 0.5
t6_bias = task6_errors["Error"].mean()

t6_k1, t6_k2, t6_k3, t6_k4 = st.columns(4)

t6_k1.metric("Evaluated Predictions", f"{t6_n:,}")
t6_k2.metric("Sample MAE", f"{t6_mae:.4f}")
t6_k3.metric("Sample RMSE", f"{t6_rmse:.4f}")
t6_k4.metric("Mean Signed Error", f"{t6_bias:.4f}")

st.caption(
    "These metrics are calculated from the saved prediction-error "
    "file and may differ from cross-validation averages. "
    "Signed error is Actual Rating minus Predicted Rating."
)

st.markdown("#### Actual vs Predicted Ratings")

fig_t6_actual = px.scatter(
    task6_errors,
    x="Actual_Rating",
    y="Predicted_Rating",
    color="Rating_Group",
    hover_data=[
        "category_grouped",
        "discounted_price",
        "Absolute_Error"
    ],
    title="Actual vs Predicted Product Ratings",
    opacity=0.7
)

t6_line_min = min(
    task6_errors["Actual_Rating"].min(),
    task6_errors["Predicted_Rating"].min()
)

t6_line_max = max(
    task6_errors["Actual_Rating"].max(),
    task6_errors["Predicted_Rating"].max()
)

fig_t6_actual.add_shape(
    type="line",
    x0=t6_line_min,
    y0=t6_line_min,
    x1=t6_line_max,
    y1=t6_line_max,
    line=dict(color="white", dash="dash")
)

fig_t6_actual.update_layout(
    template="plotly_dark",
    xaxis_title="Actual Rating",
    yaxis_title="Predicted Rating"
)

st.plotly_chart(
    fig_t6_actual,
    width="stretch"
)

st.caption(
    "The dashed diagonal represents perfect predictions. "
    "Points farther from the line have larger prediction errors."
)

st.markdown("#### Prediction Error Distribution")

t6_chart1, t6_chart2 = st.columns(2)

with t6_chart1:
    fig_t6_error_dist = px.histogram(
        task6_errors,
        x="Error",
        nbins=35,
        title="Distribution of Signed Prediction Errors",
        labels={"Error": "Actual - Predicted"}
    )

    fig_t6_error_dist.add_vline(
        x=0,
        line_dash="dash",
        line_color="white"
    )

    fig_t6_error_dist.update_layout(
        template="plotly_dark",
        yaxis_title="Number of Predictions"
    )

    st.plotly_chart(
        fig_t6_error_dist,
        width="stretch"
    )

with t6_chart2:
    fig_t6_absolute = px.scatter(
        task6_errors,
        x="Actual_Rating",
        y="Absolute_Error",
        color="Rating_Group",
        title="Absolute Error by Actual Rating",
        opacity=0.7
    )

    fig_t6_absolute.update_layout(
        template="plotly_dark",
        xaxis_title="Actual Rating",
        yaxis_title="Absolute Prediction Error"
    )

    st.plotly_chart(
        fig_t6_absolute,
        width="stretch"
    )

st.markdown("#### Largest Prediction Errors")

task6_largest_errors = (
    task6_errors
    .sort_values("Absolute_Error", ascending=False)
    .head(15)
    .copy()
)

task6_largest_errors["Record"] = [
    f"Record {idx + 1}"
    for idx in task6_largest_errors.index
]

fig_t6_largest = px.bar(
    task6_largest_errors.sort_values("Absolute_Error"),
    x="Absolute_Error",
    y="Record",
    orientation="h",
    color="Rating_Group",
    hover_data=[
        "Actual_Rating",
        "Predicted_Rating",
        "category_grouped"
    ],
    title="Top 15 Largest Absolute Prediction Errors"
)

fig_t6_largest.update_layout(
    template="plotly_dark",
    xaxis_title="Absolute Error",
    yaxis_title="Prediction Record",
    height=550
)

st.plotly_chart(
    fig_t6_largest,
    width="stretch"
)

st.markdown("#### Prediction Error Records")

t6_search = st.text_input(
    "Search by product category",
    key="task6_error_category_search"
)

t6_filtered_errors = task6_errors.copy()

if t6_search:
    t6_filtered_errors = t6_filtered_errors[
        t6_filtered_errors["category_grouped"]
        .astype(str)
        .str.contains(
            t6_search,
            case=False,
            na=False,
            regex=False
        )
    ]

st.dataframe(
    t6_filtered_errors.sort_values(
        "Absolute_Error",
        ascending=False
    ),
    width="stretch",
    hide_index=True
)

st.markdown("#### Prediction Error Findings")

t6_overpredicted = int(
    (task6_errors["Error"] < 0).sum()
)

t6_underpredicted = int(
    (task6_errors["Error"] > 0).sum()
)

t6_large_errors = int(
    (task6_errors["Absolute_Error"] > 0.5).sum()
)

st.write(
    f"**1. Overall Accuracy:** The saved prediction file "
    f"contains {t6_n:,} observations with MAE = "
    f"{t6_mae:.4f} and RMSE = {t6_rmse:.4f}."
)

st.write(
    f"**2. Prediction Direction:** "
    f"{t6_overpredicted:,} predictions overestimated "
    f"the actual rating, while {t6_underpredicted:,} "
    f"underestimated it."
)

st.write(
    f"**3. Large Errors:** "
    f"{t6_large_errors:,} predictions have an absolute "
    f"error greater than 0.5 rating points."
)

st.info(
    "Prediction errors help identify where the rating model "
    "performs poorly. Large errors may indicate that product "
    "price, discount, rating count, and category alone do not "
    "capture all factors affecting customer ratings."
)


st.divider()
st.subheader("Task 6 — Category & Rating Group Error Analysis")

t6_group_df = task6_errors.copy()

t6_group_df["Squared_Error"] = t6_group_df["Error"] ** 2

t6_category_summary = (
    t6_group_df
    .groupby("category_grouped", dropna=False)
    .agg(
        Products=("Absolute_Error", "size"),
        MAE=("Absolute_Error", "mean"),
        MSE=("Squared_Error", "mean"),
        Mean_Error=("Error", "mean")
    )
    .reset_index()
)

t6_category_summary["RMSE"] = (
    t6_category_summary["MSE"] ** 0.5
)

t6_category_summary = t6_category_summary.sort_values(
    "MAE",
    ascending=True
)

st.markdown("#### Category-wise Prediction Performance")

t6_cat_col1, t6_cat_col2 = st.columns(2)

with t6_cat_col1:
    fig_t6_cat_mae = px.bar(
        t6_category_summary,
        x="category_grouped",
        y="MAE",
        color="category_grouped",
        text="MAE",
        title="Mean Absolute Error by Category"
    )

    fig_t6_cat_mae.update_traces(
        texttemplate="%{text:.3f}",
        textposition="outside"
    )

    fig_t6_cat_mae.update_layout(
        template="plotly_dark",
        showlegend=False,
        xaxis_title="Product Category",
        yaxis_title="MAE (Lower is Better)"
    )

    st.plotly_chart(
        fig_t6_cat_mae,
        width="stretch"
    )

with t6_cat_col2:
    fig_t6_cat_rmse = px.bar(
        t6_category_summary,
        x="category_grouped",
        y="RMSE",
        color="category_grouped",
        text="RMSE",
        title="Root Mean Squared Error by Category"
    )

    fig_t6_cat_rmse.update_traces(
        texttemplate="%{text:.3f}",
        textposition="outside"
    )

    fig_t6_cat_rmse.update_layout(
        template="plotly_dark",
        showlegend=False,
        xaxis_title="Product Category",
        yaxis_title="RMSE (Lower is Better)"
    )

    st.plotly_chart(
        fig_t6_cat_rmse,
        width="stretch"
    )

st.dataframe(
    t6_category_summary[
        ["category_grouped", "Products", "MAE", "RMSE", "Mean_Error"]
    ].style.format({
        "MAE": "{:.4f}",
        "RMSE": "{:.4f}",
        "Mean_Error": "{:.4f}"
    }),
    width="stretch",
    hide_index=True
)

st.divider()
st.markdown("#### Rating Group Prediction Performance")

t6_rating_summary = (
    t6_group_df
    .groupby("Rating_Group", dropna=False)
    .agg(
        Products=("Absolute_Error", "size"),
        MAE=("Absolute_Error", "mean"),
        MSE=("Squared_Error", "mean"),
        Mean_Error=("Error", "mean")
    )
    .reset_index()
)

t6_rating_summary["RMSE"] = (
    t6_rating_summary["MSE"] ** 0.5
)

t6_rating_summary = t6_rating_summary.sort_values(
    "MAE",
    ascending=True
)

t6_rating_col1, t6_rating_col2 = st.columns(2)

with t6_rating_col1:
    fig_t6_rating_mae = px.bar(
        t6_rating_summary,
        x="Rating_Group",
        y="MAE",
        color="Rating_Group",
        text="MAE",
        title="MAE by Rating Group"
    )

    fig_t6_rating_mae.update_traces(
        texttemplate="%{text:.3f}",
        textposition="outside"
    )

    fig_t6_rating_mae.update_layout(
        template="plotly_dark",
        showlegend=False,
        xaxis_title="Rating Group",
        yaxis_title="MAE"
    )

    st.plotly_chart(
        fig_t6_rating_mae,
        width="stretch"
    )

with t6_rating_col2:
    fig_t6_rating_box = px.box(
        t6_group_df,
        x="Rating_Group",
        y="Absolute_Error",
        color="Rating_Group",
        points="outliers",
        title="Absolute Error Distribution by Rating Group"
    )

    fig_t6_rating_box.update_layout(
        template="plotly_dark",
        showlegend=False,
        xaxis_title="Rating Group",
        yaxis_title="Absolute Error"
    )

    st.plotly_chart(
        fig_t6_rating_box,
        width="stretch"
    )

st.dataframe(
    t6_rating_summary[
        ["Rating_Group", "Products", "MAE", "RMSE", "Mean_Error"]
    ].style.format({
        "MAE": "{:.4f}",
        "RMSE": "{:.4f}",
        "Mean_Error": "{:.4f}"
    }),
    width="stretch",
    hide_index=True
)

st.divider()
st.markdown("#### Category & Rating Group Findings")

t6_best_category = t6_category_summary.iloc[0]
t6_worst_category = t6_category_summary.iloc[-1]

t6_best_group = t6_rating_summary.iloc[0]
t6_worst_group = t6_rating_summary.iloc[-1]

st.write(
    f"**1. Lowest Category MAE:** "
    f"{t6_best_category['category_grouped']} "
    f"has the lowest observed MAE "
    f"({t6_best_category['MAE']:.4f}) "
    f"across {int(t6_best_category['Products'])} predictions."
)

st.write(
    f"**2. Highest Category MAE:** "
    f"{t6_worst_category['category_grouped']} "
    f"has the highest observed MAE "
    f"({t6_worst_category['MAE']:.4f}) "
    f"across {int(t6_worst_category['Products'])} predictions."
)

st.write(
    f"**3. Lowest Rating Group MAE:** "
    f"{t6_best_group['Rating_Group']} "
    f"has the lowest MAE "
    f"({t6_best_group['MAE']:.4f})."
)

st.write(
    f"**4. Highest Rating Group MAE:** "
    f"{t6_worst_group['Rating_Group']} "
    f"has the highest MAE "
    f"({t6_worst_group['MAE']:.4f})."
)

st.info(
    "Category-wise and rating-group error comparisons reveal "
    "where the prediction model performs consistently and "
    "where additional features or training data may help. "
    "Groups with small sample sizes should be interpreted cautiously."
)


st.divider()
st.subheader("Task 6 — Model Comparison & Hyperparameter Tuning")

task6_models = pd.read_csv(
    Path(__file__).resolve().parent / "task6_model_comparison.csv"
)

for col in ["Test_MAE", "Test_RMSE", "Test_R2"]:
    task6_models[col] = pd.to_numeric(
        task6_models[col],
        errors="coerce"
    )

task6_models = task6_models.dropna(
    subset=["Model", "Test_MAE", "Test_RMSE", "Test_R2"]
).copy()

task6_models = task6_models.sort_values(
    "Test_MAE"
).reset_index(drop=True)

t6_best_test = task6_models.iloc[0]

st.markdown("#### Test Set Performance")

t6_m1, t6_m2, t6_m3, t6_m4 = st.columns(4)

t6_m1.metric("Models Compared", len(task6_models))
t6_m2.metric("Best Test Model", t6_best_test["Model"])
t6_m3.metric("Best Test MAE", f"{t6_best_test['Test_MAE']:.4f}")
t6_m4.metric("Best Test R²", f"{t6_best_test['Test_R2']:.4f}")

t6_compare1, t6_compare2 = st.columns(2)

with t6_compare1:
    fig_t6_test_mae = px.bar(
        task6_models,
        x="Model",
        y="Test_MAE",
        color="Model",
        text="Test_MAE",
        title="Test MAE by Model"
    )

    fig_t6_test_mae.update_traces(
        texttemplate="%{text:.4f}",
        textposition="outside"
    )

    fig_t6_test_mae.update_layout(
        template="plotly_dark",
        showlegend=False,
        xaxis_tickangle=-25,
        yaxis_title="Test MAE (Lower is Better)"
    )

    st.plotly_chart(fig_t6_test_mae, width="stretch")

with t6_compare2:
    fig_t6_test_rmse = px.bar(
        task6_models,
        x="Model",
        y="Test_RMSE",
        color="Model",
        text="Test_RMSE",
        title="Test RMSE by Model"
    )

    fig_t6_test_rmse.update_traces(
        texttemplate="%{text:.4f}",
        textposition="outside"
    )

    fig_t6_test_rmse.update_layout(
        template="plotly_dark",
        showlegend=False,
        xaxis_tickangle=-25,
        yaxis_title="Test RMSE (Lower is Better)"
    )

    st.plotly_chart(fig_t6_test_rmse, width="stretch")

fig_t6_test_r2 = px.bar(
    task6_models,
    x="Model",
    y="Test_R2",
    color="Model",
    text="Test_R2",
    title="Test R² Comparison"
)

fig_t6_test_r2.update_traces(
    texttemplate="%{text:.4f}",
    textposition="outside"
)

fig_t6_test_r2.update_layout(
    template="plotly_dark",
    showlegend=False,
    xaxis_tickangle=-25,
    yaxis_title="Test R² (Higher is Better)"
)

st.plotly_chart(fig_t6_test_r2, width="stretch")

st.markdown("#### Model Comparison Table")

st.dataframe(
    task6_models.style.format({
        "Test_MAE": "{:.4f}",
        "Test_RMSE": "{:.4f}",
        "Test_R2": "{:.4f}"
    }),
    width="stretch",
    hide_index=True
)

st.divider()
st.markdown("#### Random Forest Hyperparameter Tuning")

t6_rf = task6_models.loc[
    task6_models["Model"] == "Random Forest"
]

t6_tuned = task6_models.loc[
    task6_models["Model"] == "Tuned Random Forest"
]

if not t6_rf.empty and not t6_tuned.empty:
    t6_rf = t6_rf.iloc[0]
    t6_tuned = t6_tuned.iloc[0]

    t6_mae_improvement = (
        (t6_rf["Test_MAE"] - t6_tuned["Test_MAE"])
        / t6_rf["Test_MAE"] * 100
    )

    t6_rmse_improvement = (
        (t6_rf["Test_RMSE"] - t6_tuned["Test_RMSE"])
        / t6_rf["Test_RMSE"] * 100
    )

    t6_r2_gain = (
        t6_tuned["Test_R2"] - t6_rf["Test_R2"]
    )

    t6_tune1, t6_tune2, t6_tune3 = st.columns(3)

    t6_tune1.metric(
        "MAE Reduction",
        f"{t6_mae_improvement:.2f}%"
    )

    t6_tune2.metric(
        "RMSE Reduction",
        f"{t6_rmse_improvement:.2f}%"
    )

    t6_tune3.metric(
        "R² Improvement",
        f"{t6_r2_gain:+.4f}"
    )

    t6_tuning_chart = pd.DataFrame({
        "Model": [
            "Random Forest",
            "Tuned Random Forest"
        ],
        "Test MAE": [
            t6_rf["Test_MAE"],
            t6_tuned["Test_MAE"]
        ],
        "Test RMSE": [
            t6_rf["Test_RMSE"],
            t6_tuned["Test_RMSE"]
        ],
        "Test R²": [
            t6_rf["Test_R2"],
            t6_tuned["Test_R2"]
        ]
    })

    t6_tuning_long = t6_tuning_chart.melt(
        id_vars="Model",
        var_name="Metric",
        value_name="Value"
    )

    fig_t6_tuning = px.bar(
        t6_tuning_long,
        x="Metric",
        y="Value",
        color="Model",
        barmode="group",
        facet_col="Metric",
        facet_col_wrap=3,
        facet_col_spacing=0.08,
        title="Original vs Tuned Random Forest",
        text="Value"
    )

    fig_t6_tuning.update_yaxes(matches=None)

    fig_t6_tuning.update_traces(
        texttemplate="%{text:.4f}",
        textposition="outside"
    )

    fig_t6_tuning.update_layout(
        template="plotly_dark",
        height=500
    )

    st.plotly_chart(
        fig_t6_tuning,
        width="stretch"
    )

    st.write(
        f"**Tuning Result:** Test MAE changed from "
        f"{t6_rf['Test_MAE']:.4f} to "
        f"{t6_tuned['Test_MAE']:.4f}, "
        f"while Test R² changed from "
        f"{t6_rf['Test_R2']:.4f} to "
        f"{t6_tuned['Test_R2']:.4f}."
    )

st.divider()
st.markdown("#### Model Selection Findings")

st.success(
    f"{t6_best_test['Model']} achieved the lowest "
    f"test MAE ({t6_best_test['Test_MAE']:.4f}) "
    f"among the evaluated models."
)

st.info(
    "Hyperparameter tuning improved the saved Random Forest "
    "test metrics, but the best Test R² remains relatively low. "
    "The model should be treated as a baseline for rating "
    "estimation rather than a highly accurate predictor. "
    "The tuned model has test results but no separate "
    "cross-validation results in the supplied CV file."
)


st.divider()
st.subheader("Task 6 — Key Findings & Final Conclusion")

t6_final_models = task6_models.sort_values(
    "Test_MAE",
    ascending=True
).reset_index(drop=True)

t6_final_best = t6_final_models.iloc[0]

t6_final_cv = task6_cv.sort_values(
    "CV_MAE",
    ascending=True
).reset_index(drop=True)

t6_final_cv_best = t6_final_cv.iloc[0]

t6_final_features = task6_features.sort_values(
    "Importance_Mean",
    ascending=False
).reset_index(drop=True)

t6_final_top_feature = t6_final_features.iloc[0]

t6_final_mae = task6_errors["Absolute_Error"].mean()

t6_final_large_errors = int(
    (task6_errors["Absolute_Error"] > 0.5).sum()
)

t6_final_large_pct = (
    t6_final_large_errors / len(task6_errors) * 100
)

st.markdown("#### Key Findings")

st.write(
    f"**1. Best Test Model:** "
    f"{t6_final_best['Model']} achieved the lowest test MAE "
    f"({t6_final_best['Test_MAE']:.4f}), "
    f"with RMSE = {t6_final_best['Test_RMSE']:.4f} "
    f"and R² = {t6_final_best['Test_R2']:.4f}."
)

st.write(
    f"**2. Cross-Validation Performance:** "
    f"{t6_final_cv_best['Model']} achieved the lowest "
    f"cross-validation MAE ({t6_final_cv_best['CV_MAE']:.4f}) "
    f"among models included in the saved CV report, "
    f"with R² = {t6_final_cv_best['CV_R2']:.4f}."
)

t6_original_rf = task6_models.loc[
    task6_models["Model"] == "Random Forest"
]

t6_tuned_rf = task6_models.loc[
    task6_models["Model"] == "Tuned Random Forest"
]

if not t6_original_rf.empty and not t6_tuned_rf.empty:
    t6_original_rf = t6_original_rf.iloc[0]
    t6_tuned_rf = t6_tuned_rf.iloc[0]

    t6_final_mae_reduction = (
        (
            t6_original_rf["Test_MAE"]
            - t6_tuned_rf["Test_MAE"]
        )
        / t6_original_rf["Test_MAE"]
        * 100
    )

    t6_final_r2_improvement = (
        t6_tuned_rf["Test_R2"]
        - t6_original_rf["Test_R2"]
    )

    st.write(
        f"**3. Hyperparameter Tuning:** "
        f"Random Forest tuning reduced test MAE by "
        f"{t6_final_mae_reduction:.2f}% and improved "
        f"test R² by {t6_final_r2_improvement:.4f}."
    )

st.write(
    f"**4. Feature Importance:** "
    f"{t6_final_top_feature['Feature']} had the highest "
    f"mean importance "
    f"({t6_final_top_feature['Importance_Mean']:.6f}) "
    f"in the saved feature importance report."
)

st.write(
    f"**5. Prediction Errors:** "
    f"The saved prediction-error dataset contains "
    f"{len(task6_errors):,} observations with "
    f"MAE = {t6_final_mae:.4f}. "
    f"{t6_final_large_errors:,} observations "
    f"({t6_final_large_pct:.2f}%) had absolute "
    f"prediction errors greater than 0.5 rating points."
)

st.markdown("#### Business Interpretation")

st.info(
    "Product ratings are difficult to predict using only "
    "price, discount percentage, rating count, and product "
    "category. The model can provide approximate rating "
    "estimates, but individual predictions may have substantial "
    "errors, especially for products with unusual rating patterns."
)

st.markdown("#### Recommendations")

st.write(
    "**1. Additional Features:** Evaluate review sentiment, "
    "product attributes, brand information, and other relevant "
    "variables to improve predictive performance."
)

st.write(
    "**2. Model Validation:** Use consistent cross-validation "
    "and a separate test set to evaluate future model improvements."
)

st.write(
    "**3. Error Monitoring:** Monitor category-wise and "
    "rating-group errors before using predictions in business decisions."
)

st.markdown("#### Final Conclusion")

st.success(
    "Task 6 Completed: Multiple regression models were evaluated, "
    "cross-validation performance was compared, feature importance "
    "was examined, prediction errors were analysed, and Random Forest "
    "hyperparameter tuning was assessed. Tuned Random Forest "
    "achieved the best saved test performance, but its relatively "
    "low R² highlights the need for additional predictive features."
)

st.caption(
    "Test-set results, cross-validation results, and saved "
    "prediction-error statistics represent different evaluation "
    "summaries and should not be treated as interchangeable. "
    "The supplied cross-validation file does not include "
    "a separate result for the tuned Random Forest."
)
