import streamlit as st
import pickle
import pandas as pd
import numpy as np
import plotly.express as px

# 1. Config Canvas Setup
st.set_page_config(
    page_title="Shopper Spectrum Portal",
    page_icon="🔮",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 2. Complete Custom Premium Theme Injector
st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700;900&display=swap');
        
        /* Global Body Defaults override */
        html, body, [class*="css"] {
            font-family: 'Inter', sans-serif;
        }
        
        /* Master Gradient Page Titles */
        .ultimate-title {
            font-size: 46px !important;
            font-weight: 900 !important;
            background: linear-gradient(135deg, #00f2fe 0%, #4facfe 50%, #0000ff 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 2px;
            letter-spacing: -1.5px;
            text-transform: uppercase;
        }
        .ultimate-title-recommendation {
            font-size: 46px !important;
            font-weight: 900 !important;
            background: linear-gradient(135deg, #ff007f 0%, #ff5277 50%, #ff758c 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 2px;
            letter-spacing: -1.5px;
            text-transform: uppercase;
        }
        .ultimate-title-clustering {
            font-size: 46px !important;
            font-weight: 900 !important;
            background: linear-gradient(135deg, #7f00ff 0%, #e100ff 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 2px;
            letter-spacing: -1.5px;
            text-transform: uppercase;
        }
        
        .ultimate-subtitle {
            color: #a0a0cc;
            font-size: 18px;
            font-weight: 400;
            margin-bottom: 30px;
        }

        /* Container Premium Cards */
        .card-container {
            background: linear-gradient(145deg, #131326 0%, #1a1a36 100%);
            border: 1px solid #29294d;
            border-radius: 20px;
            padding: 30px;
            margin-bottom: 25px;
            box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.3);
        }
        
        .card-header-cyan {
            color: #00f2fe;
            font-size: 24px;
            font-weight: 700;
            margin-bottom: 15px;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }
        .card-header-pink {
            color: #ff007f;
            font-size: 24px;
            font-weight: 700;
            margin-bottom: 15px;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }
        
        /* Interactive Metric Grids */
        .metric-box {
            background: rgba(255, 255, 255, 0.03);
            border: 1px solid rgba(255, 255, 255, 0.08);
            border-radius: 12px;
            padding: 15px;
            text-align: center;
        }
        .metric-label { color: #8888aa; font-size: 12px; text-transform: uppercase; letter-spacing: 1px; }
        .metric-value { color: #ffffff; font-size: 26px; font-weight: 700; margin-top: 5px; }

        /* Modern Segment Badges */
        .badge-occasional { background: linear-gradient(90deg, #3b82f6, #1d4ed8); color: white; padding: 8px 18px; border-radius: 30px; font-weight: 700; font-size: 18px; display: inline-block; }
        .badge-atrisk { background: linear-gradient(90deg, #ef4444, #b91c1c); color: white; padding: 8px 18px; border-radius: 30px; font-weight: 700; font-size: 18px; display: inline-block; }
        .badge-high { background: linear-gradient(90deg, #eab308, #a16207); color: black; padding: 8px 18px; border-radius: 30px; font-weight: 700; font-size: 18px; display: inline-block; }
        .badge-regular { background: linear-gradient(90deg, #22c55e, #15803d); color: white; padding: 8px 18px; border-radius: 30px; font-weight: 700; font-size: 18px; display: inline-block; }

        /* Item Recommendation Blocks */
        .rec-item-card {
            background: #0d0d1a;
            border: 1px solid #22223b;
            border-radius: 16px;
            padding: 25px 15px;
            text-align: center;
            box-shadow: 0 4px 15px rgba(0,0,0,0.4);
            height: 100%;
            transition: all 0.3s ease;
        }
        .rec-item-card:hover {
            border-color: #ff007f;
            transform: translateY(-8px);
            box-shadow: 0 12px 24px rgba(255, 0, 127, 0.25);
        }
    </style>
""", unsafe_allow_html=True)

# ------------------------------------------------------------------
# 3. ML CACHE ASSETS ENGINE LOADING
# ------------------------------------------------------------------
@st.cache_resource
def load_assets():
    try:
        with open('kmeans_model.pkl', 'rb') as f:
            kmeans = pickle.load(f)
        with open('scaler.pkl', 'rb') as f:
            scaler = pickle.load(f)
        with open('item_similarity.pkl', 'rb') as f:
            similarity_df = pickle.load(f)
        return kmeans, scaler, similarity_df
    except FileNotFoundError:
        return None, None, None

kmeans, scaler, similarity_df = load_assets()

# Clean native static dataframe mappings to ensure perfect layout display
@st.cache_data
def get_clean_chart_assets():
    df_geo = pd.DataFrame({
        'Country': ['United Kingdom', 'Germany', 'France', 'EIRE', 'Spain', 'Netherlands', 'Belgium'],
        'Transactions Registered': [361878, 9495, 8557, 8196, 2533, 2371, 2069]
    })
    df_segments_dist = pd.DataFrame({
        'Segment Tier': ['Occasional Shopper', 'Regular Shopper', 'At-Risk Customer', 'High-Value VIP'],
        'Active Profiles Count': [2042, 1128, 1056, 112]
    })
    return df_geo, df_segments_dist

df_geo, df_segments_dist = get_clean_chart_assets()

cluster_map = {
    0: ('Occasional Shopper', 'badge-occasional', 'Infrequent buyers with conservative item acquisition habits. Target with welcome coupons.'),
    1: ('At-Risk Customer', 'badge-atrisk', 'Dormant buyers with zero interaction within months. Immediate push-reengagement required.'),
    2: ('High-Value VIP Spend', 'badge-high', 'Top-tier core brand advocates. Deliver white-glove support, early access, and premium tiers.'),
    3: ('Regular Shopper', 'badge-regular', 'Loyal recurring buyers. Highly susceptible to active cross-selling matching recommendation queries.')
}

# ------------------------------------------------------------------
# 4. SIDEBAR NAVIGATION CONTROLS
# ------------------------------------------------------------------
st.sidebar.markdown("<h1 style='text-align: center; color: #6366f1; font-weight:900; font-size:26px;'>🔮 MATRIX PORTAL</h1>", unsafe_allow_html=True)
st.sidebar.markdown("---")
page = st.sidebar.radio("CHOOSE INSTANCE:", ["🏠 Executive Home", "📊 Clustering Studio", "🎁 Recommendation Engine"])
st.sidebar.markdown("---")
st.sidebar.markdown("<p style='text-align: center; color: #555577; font-size:11px;'>Shopper Spectrum v4.0 Ultra<br>Enterprise Ready Deployment</p>", unsafe_allow_html=True)

# ------------------------------------------------------------------
# PAGE ONE: EXECUTIVE HOME PORTAL
# ------------------------------------------------------------------
if page == "🏠 Executive Home":
    st.markdown('<div class="ultimate-title">⚡ Shopper Spectrum Matrix</div>', unsafe_allow_html=True)
    st.markdown('<div class="ultimate-subtitle">Translating raw transactional streams into live, operational intelligence clusters.</div>', unsafe_allow_html=True)
    st.markdown("---")
    
    # Custom HTML Metric Layout Columns
    met1, met2, met3, met4 = st.columns(4)
    with met1:
        st.markdown('<div class="metric-box"><div class="metric-label">RAW INVOICES LOADED</div><div class="metric-value" style="color: #4facfe;">541,909</div></div>', unsafe_allow_html=True)
    with met2:
        st.markdown('<div class="metric-box"><div class="metric-label">CLEANED TRANSACTIONS</div><div class="metric-value" style="color: #00f2fe;">392,692</div></div>', unsafe_allow_html=True)
    with met3:
        st.markdown('<div class="metric-box"><div class="metric-label">CATALOG SIZE</div><div class="metric-value" style="color: #ff007f;">3,877</div></div>', unsafe_allow_html=True)
    with met4:
        st.markdown('<div class="metric-box"><div class="metric-label">ACTIVE BUYERS MAP</div><div class="metric-value" style="color: #22c55e;">4,338</div></div>', unsafe_allow_html=True)
        
    st.markdown("<br>", unsafe_allow_html=True)
    
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("""
            <div class="card-container">
                <div class="card-header-cyan">🎯 Unsupervised Clustering Studio</div>
                <p style="color: #b0b0cd; line-height:1.6; font-size:14px;">
                    Parses live customer interaction weights along real-time Recency, Frequency, and Monetary dimensions to instantly classify your audience into action-oriented lifecycle segments.
                </p>
            </div>
        """, unsafe_allow_html=True)
    with c2:
        st.markdown("""
            <div class="card-container" style="border-color: #ff007f;">
                <div class="card-header-pink">🛍️ Item-Based Affinity Recommendations</div>
                <p style="color: #b0b0cd; line-height:1.6; font-size:14px;">
                    Extracts product-to-product vector similarities using Cosine Distance calculations across customer interaction matrices, driving item cross-sells at checkout.
                </p>
            </div>
        """, unsafe_allow_html=True)

    st.markdown("<h2 style='color: #ffffff; font-weight:800; margin-top:15px;'>📊 Deep-Dive Preprocessing & Distribution Profiles</h2>", unsafe_allow_html=True)
    
    tab1, tab2 = st.tabs(["🌎 Core Geographic Spreads", "🍕 Segment Profile Distributions"])
    
    with tab1:
        fig1 = px.bar(df_geo, x='Transactions Registered', y='Country', orientation='h',
                     color='Transactions Registered', color_continuous_scale='Blues', template='plotly_dark')
        fig1.update_layout(yaxis={'categoryorder':'total ascending'}, plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', title_font_size=20)
        st.plotly_chart(fig1, use_container_width=True)
        
    with tab2:
        fig2 = px.pie(df_segments_dist, values='Active Profiles Count', names='Segment Tier',
                      color_discrete_sequence=px.colors.sequential.RdBu, hole=0.4, template='plotly_dark')
        fig2.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig2, use_container_width=True)

# ------------------------------------------------------------------
# PAGE TWO: CLUSTERING INFERENCE ENGINE
# ------------------------------------------------------------------
elif page == "📊 Clustering Studio":
    st.markdown('<div class="ultimate-title-clustering">📊 Customer Segmentation Studio</div>', unsafe_allow_html=True)
    st.markdown('<div class="ultimate-subtitle">Predictive inference mapping against pre-calculated feature scales.</div>', unsafe_allow_html=True)
    st.markdown("---")
    
    if kmeans is None or scaler is None:
        st.error("Pre-trained model matrices not found. Run your notebook cells first!")
    else:
        left_input, right_output = st.columns([1, 1.2])
        
        with left_input:
            st.markdown("<h3 style='color: #a100ff; font-weight:700;'>📥 Input Customer Metrics</h3>", unsafe_allow_html=True)
            
            recency = st.slider("Recency (Days since last interaction)", 1, 375, 45)
            frequency = st.slider("Frequency (Total unique invoice orders)", 1, 300, 12)
            monetary = st.number_input("Monetary Value ($ Overall Customer Spend)", min_value=1.0, max_value=250000.0, value=750.0, step=100.0)
            
            st.markdown("<br>", unsafe_allow_html=True)
            run_prediction = st.button("🔮 Calculate Customer Segment Tier", use_container_width=True)
            
        with right_output:
            st.markdown("<h3 style='color: #ffffff; font-weight:700;'>📤 AI Engine Classification</h3>", unsafe_allow_html=True)
            
            if run_prediction:
                features = np.array([[recency, frequency, monetary]])
                scaled_features = scaler.transform(features)
                cluster_id = kmeans.predict(scaled_features)[0]
                
                label, style_class, description = cluster_map[cluster_id]
                
                st.markdown(f"""
                    <div class="card-container" style="border-left: 6px solid #e100ff; margin-top: 15px;">
                        <span style="color:#888; font-size:11px; text-transform:uppercase; letter-spacing:1px; display:block; margin-bottom:8px;">Model Assignment Result</span>
                        <div style="margin-bottom: 20px;">
                            <span class="{style_class}">{label}</span>
                        </div>
                        <p style="color:#ffffff; font-size:16px;"><b>Assigned Segment ID:</b> Cluster Index {cluster_id}</p>
                        <p style="color:#b0b0cd; font-size:14px; margin-top:10px; line-height:1.5;">{description}</p>
                    </div>
                """, unsafe_allow_html=True)
            else:
                st.info("Adjust the parameter inputs on the left pane and select the prediction run engine button to evaluate results.")

# ------------------------------------------------------------------
# PAGE THREE: INTERACTIVE PRODUCT RECOMMENDATION
# ------------------------------------------------------------------
elif page == "🎁 Recommendation Engine":
    st.markdown('<div class="ultimate-title-recommendation">🎁 AI Product Recommendation Engine</div>', unsafe_allow_html=True)
    st.markdown('<div class="ultimate-subtitle">Real-time up-sell matching powered by item-to-item Cosine Similarity.</div>', unsafe_allow_html=True)
    st.markdown("---")
    
    if similarity_df is None:
        st.error("Pre-calculated similarity matrix weights missing. Run notebook workflows!")
    else:
        st.markdown("<h3 style='color: #ff5277; font-weight:700;'>🛒 Target Product Lookup Selection</h3>", unsafe_allow_html=True)
        
        target_item = st.selectbox(
            "Select or Type Active Catalog Item:",
            options=similarity_df.index.tolist()
        )
        
        st.markdown("<br>", unsafe_allow_html=True)
        get_recs = st.button("🎁 Generate Top 5 Best Recommendations", use_container_width=True)
        
        if get_recs:
            similar_scores = similarity_df[target_item].sort_values(ascending=False).head(6)
            recommended_items = similar_scores.iloc[1:].index.tolist()
            
            st.markdown("<h4 style='color: #ffffff; margin-top: 30px; margin-bottom: 20px; font-weight:800;'>📦 Co-Purchase Affinity Recommendations Matrix:</h4>", unsafe_allow_html=True)
            
            cols_grid = st.columns(5)
            for position, product_name in enumerate(recommended_items):
                with cols_grid[position]:
                    st.markdown(f"""
                        <div class="rec-item-card">
                            <div style="font-size: 32px; margin-bottom: 10px;">📦</div>
                            <div style="color: #ff007f; font-weight:800; font-size:12px; text-transform:uppercase; margin-bottom:10px;">Affinity Match #{position+1}</div>
                            <div style="color: #ffffff; font-size:13px; font-weight:600; line-height:1.4; min-height:60px;">{product_name}</div>
                        </div>
                    """, unsafe_allow_html=True)