import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go
import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ================================================
# CONFIG
# ================================================
st.set_page_config(
    page_title="Rapido Captain Churn Predictor",
    page_icon="🏍️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ================================================
# DESIGN TOKENS — Rapido Brand
# ================================================
COLORS = {
    'yellow':   '#F5C518',
    'yellow_dk':'#D4A800',
    'black':    '#1A1A1A',
    'charcoal': '#2D2D2D',
    'white':    '#FFFFFF',
    'bg':       '#F7F7F7',
    'border':   '#E8E8E8',
    'green':    '#1DB954',
    'red':      '#E53935',
    'blue':     '#1565C0',
    'text':     '#1A1A1A',
    'subtext':  '#6B6B6B',
    'grid':     '#F0F0F0',
}

FONT = 'Plus Jakarta Sans, sans-serif'

PLOTLY_LAYOUT = dict(
    paper_bgcolor=COLORS['white'],
    plot_bgcolor=COLORS['white'],
    font=dict(family=FONT, color=COLORS['text'], size=12),
    margin=dict(t=20, b=45, l=50, r=30),
    xaxis=dict(
        showgrid=False,
        linecolor=COLORS['border'],
        tickfont=dict(size=11, color=COLORS['subtext']),
        title_font=dict(size=12, color=COLORS['subtext']),
        zeroline=False,
    ),
    yaxis=dict(
        showgrid=True,
        gridcolor=COLORS['grid'],
        gridwidth=1,
        linecolor='rgba(0,0,0,0)',
        tickfont=dict(size=11, color=COLORS['subtext']),
        title_font=dict(size=12, color=COLORS['subtext']),
        zeroline=False,
    ),
    hoverlabel=dict(
        bgcolor=COLORS['white'],
        bordercolor=COLORS['border'],
        font=dict(family=FONT, size=12, color=COLORS['text'])
    ),
    title_font=dict(family=FONT, size=15, color=COLORS['black']),
    title_x=0.0,
)

# ================================================
# CUSTOM CSS — Rapido Theme
# ================================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&display=swap');


.material-symbols-rounded,
.material-icons,
span[class*="material-symbols"] {
    font-family: 'Material Symbols Rounded', 'Material Icons' !important;
}
/* Main background */
.stApp { background-color: #F7F7F7; }

/* Sidebar — Rapido black */
[data-testid="stSidebar"] {
    background: #1A1A1A;
    border-right: 1px solid #2D2D2D;
}
/* Apply custom font to everything EXCEPT icons and svgs */
*:not(i):not([class*="material-symbols"]):not([data-testid="stIconMaterial"]):not(svg):not(path) {
    font-family: 'Plus Jakarta Sans', sans-serif !important;
}
[data-testid="stSidebar"] * { color: #E0E0E0 !important; }
[data-testid="stSidebar"] .stRadio label {
    color: #C8C8C8 !important;
    font-size: 13.5px;
    padding: 7px 0;
    font-weight: 500;
}

/* Metric cards */
[data-testid="stMetric"] {
    background: white;
    border-radius: 10px;
    padding: 18px 20px;
    box-shadow: 0 1px 4px rgba(0,0,0,0.07);
    border-bottom: 3px solid #F5C518;
}
[data-testid="stMetricLabel"] {
    color: #6B6B6B !important;
    font-size: 11px !important;
    font-weight: 700 !important;
    text-transform: uppercase;
    letter-spacing: 0.06em;
}
[data-testid="stMetricValue"] {
    color: #1A1A1A !important;
    font-weight: 800 !important;
    font-size: 24px !important;
}

/* Headers */
h1, h2, h3 { color: #1A1A1A !important; font-weight: 800 !important; }

/* Dataframe */
[data-testid="stDataFrame"] {
    border-radius: 10px;
    overflow: hidden;
    box-shadow: 0 1px 4px rgba(0,0,0,0.07);
}

/* Buttons */
.stButton > button {
    background: #F5C518;
    color: #1A1A1A !important;
    border: none;
    border-radius: 8px;
    padding: 13px 30px;
    font-weight: 800;
    font-size: 14px;
    width: 100%;
    transition: all 0.18s ease;
    box-shadow: 0 2px 10px rgba(245,197,24,0.35);
    letter-spacing: 0.01em;
}
.stButton > button:hover {
    background: #D4A800;
    transform: translateY(-1px);
    box-shadow: 0 5px 18px rgba(245,197,24,0.45);
    color: #1A1A1A !important;
}

/* Divider */
hr { border-color: #2D2D2D; }

/* Info boxes */
.stAlert { border-radius: 10px; }

/* Page banner — Rapido black + yellow accent */
.page-banner {
    background: #1A1A1A;
    padding: 26px 30px;
    border-radius: 12px;
    margin-bottom: 24px;
    border-left: 5px solid #F5C518;
}
.page-banner h1 {
    color: white !important;
    margin: 0 0 4px 0;
    font-size: 24px;
    font-weight: 800;
}
.page-banner p {
    color: #888;
    margin: 0;
    font-size: 13px;
    font-weight: 400;
}

/* Chart context card */
.chart-card {
    background: white;
    border-radius: 12px;
    padding: 20px 22px 8px 22px;
    box-shadow: 0 1px 4px rgba(0,0,0,0.07);
    margin-bottom: 4px;
}
.chart-tag {
    font-size: 10px;
    font-weight: 700;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: #1A1A1A;
    background: #F5C518;
    padding: 2px 9px;
    border-radius: 4px;
    display: inline-block;
    margin-bottom: 8px;
}
.chart-title {
    font-size: 16px;
    font-weight: 800;
    color: #1A1A1A;
    margin: 0 0 4px 0;
}
.chart-subtitle {
    font-size: 12px;
    color: #6B6B6B;
    margin: 0 0 4px 0;
    line-height: 1.55;
}
.insight-pill {
    display: inline-flex;
    align-items: center;
    gap: 5px;
    background: #FFFBE6;
    color: #A67C00;
    font-size: 11px;
    font-weight: 700;
    padding: 3px 11px;
    border-radius: 4px;
    margin-top: 6px;
    margin-bottom: 6px;
    border: 1px solid #F5C518;
}
.insight-pill.red {
    background: #FFF0F0;
    color: #C62828;
    border-color: #FFCDD2;
}
.insight-pill.green {
    background: #F0FFF4;
    color: #1B5E20;
    border-color: #C8E6C9;
}
.insight-pill.blue {
    background: #EEF2FF;
    color: #1565C0;
    border-color: #BBDEFB;
}

/* Section label */
.section-label {
    font-size: 10.5px;
    font-weight: 700;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: #9CA3AF;
    margin: 26px 0 12px 0;
    display: flex;
    align-items: center;
    gap: 10px;
}
.section-label::after {
    content: '';
    flex: 1;
    height: 1px;
    background: #E5E5E5;
}

/* Slider labels */
[data-testid="stSlider"] label,
[data-testid="stSlider"] > label,
div[data-testid="stSlider"] p {
    color: #1A1A1A !important;
    font-size: 13px !important;
    font-weight: 600 !important;
    margin-bottom: 4px !important;
    display: block !important;
    visibility: visible !important;
    opacity: 1 !important;
}

/* Also fix selectbox labels */
[data-testid="stSelectbox"] label,
div[data-testid="stSelectbox"] p {
    color: #1A1A1A !important;
    font-size: 13px !important;
    font-weight: 600 !important;
}

/* Slider track — change from red to yellow */
[data-testid="stSlider"] .st-emotion-cache-1dx1gwv,
[data-testid="stSlider"] [class*="sliderTrack"] {
    background: #F5C518 !important;
}

/* Slider thumb color */
[data-testid="stSlider"] [role="slider"] {
    background: #F5C518 !important;
    border-color: #D4A800 !important;
}
</style>
""", unsafe_allow_html=True)

# ================================================
# LOAD DATA & MODEL
# ================================================
@st.cache_data
def load_data():
    return pd.read_csv(os.path.join(BASE_DIR, 'captains_scored.csv'))

@st.cache_resource
def load_model():
    return joblib.load(os.path.join(BASE_DIR, 'churn_model_final.pkl'))

df = load_data()
model = load_model()

avg_churn = df['is_churned'].mean() * 100

# ================================================
# SIDEBAR
# ================================================
with st.sidebar:
    st.markdown("""
    <div style='padding: 22px 0 12px 0;'>
        <div style='font-size:22px; font-weight:900; color:white; letter-spacing:-0.03em;'>
            <span style='background:#F5C518; color:#1A1A1A; padding:2px 8px; border-radius:4px;'>rapido</span>
        </div>
        <div style='font-size:11px; color:#666; margin-top:8px; font-weight:500; text-transform:uppercase; letter-spacing:0.08em;'>Churn Intelligence</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    page = st.radio("", [
        "Dashboard",
        "Predict Captain",
        "ROI Calculator",
        "City Action Plan"
    ])
    st.markdown("---")
    st.markdown("""
    <div style='font-size:11px; color:#555; padding: 8px 0; line-height:2.0;'>
        <span style='color:#888; font-weight:700; text-transform:uppercase; font-size:10px; letter-spacing:0.08em;'>Model Info</span><br>
        Algorithm: Logistic Regression<br>
        AUC-ROC: <b style='color:#F5C518;'>0.739</b> &nbsp;|&nbsp; Recall: <b style='color:#F5C518;'>70%</b><br>
        Dataset: 10,000 captains<br>
        Cities: BLR · HYD · CHN · PNQ · DEL
    </div>
    """, unsafe_allow_html=True)

# ================================================
# HELPER
# ================================================
def chart_card(tag, title, subtitle, insight=None, insight_type='yellow'):
    insight_html = ''
    if insight:
        cls = '' if insight_type == 'yellow' else insight_type
        insight_html = f"<div class='insight-pill {cls}'>{insight}</div>"
    st.markdown(f"""
    <div class='chart-card'>
        <span class='chart-tag'>{tag}</span>
        <div class='chart-title'>{title}</div>
        <div class='chart-subtitle'>{subtitle}</div>
        {insight_html}
    </div>
    """, unsafe_allow_html=True)

# ================================================
# PAGE 1 — DASHBOARD
# ================================================
if page == "Dashboard":

    st.markdown("""
    <div class='page-banner'>
        <h1>Captain Churn Dashboard</h1>
        <p>Early warning system to identify and retain at-risk Rapido captains</p>
    </div>
    """, unsafe_allow_html=True)

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Captains", f"{len(df):,}")
    with col2:
        st.metric("Overall Churn Rate", f"{df['is_churned'].mean():.1%}")
    with col3:
        high_risk = len(df[df['risk_segment'] == 'High Risk'])
        st.metric("High Risk Captains", f"{high_risk:,}")
    with col4:
        st.metric("Model AUC Score", "0.739")

    st.markdown("<div class='section-label'>Risk & Vehicle Analysis</div>", unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        chart_card(
            tag="Risk Segmentation",
            title="Churn Rate by Risk Segment",
            subtitle="High-risk captains churn significantly more than low-risk ones — early flagging is the primary retention lever.",
            insight="High-risk captains churn 3x more than low-risk",
            insight_type="red"
        )
        seg_data = df.groupby('risk_segment')['is_churned'].mean().reset_index()
        seg_data.columns = ['Segment', 'Churn Rate']
        seg_data['Churn Rate Pct'] = (seg_data['Churn Rate'] * 100).round(1)
        seg_data['Segment'] = pd.Categorical(
            seg_data['Segment'],
            categories=['Low Risk', 'Medium Risk', 'High Risk'], ordered=True
        )
        seg_data = seg_data.sort_values('Segment')

        fig = px.bar(
            seg_data, x='Segment', y='Churn Rate Pct',
            color='Segment',
            color_discrete_map={
                'Low Risk':    COLORS['green'],
                'Medium Risk': COLORS['yellow'],
                'High Risk':   COLORS['red']
            },
            text='Churn Rate Pct'
        )
        fig.update_traces(
            texttemplate='<b>%{text}%</b>',
            textposition='outside',
            marker_line_width=0,
            width=0.45,
        )
        fig.add_hline(
            y=avg_churn,
            line_dash='dot',
            line_color='#BBBBBB',
            annotation_text=f'  Fleet avg: {avg_churn:.1f}%',
            annotation_font=dict(size=10, color='#BBBBBB', family=FONT),
            annotation_position='right'
        )
        fig.update_layout(
            **PLOTLY_LAYOUT,
            height=300,
            showlegend=False,
            title=None,
            yaxis_title='Churn Rate (%)',
            xaxis_title=None,
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        chart_card(
            tag="Vehicle Type",
            title="Churn Rate by Vehicle Type",
            subtitle="Bike captains churn at higher rates — lower per-km earnings vs autos is the likely contributing factor.",
            insight="Bikes earn ~42% less per km vs autos",
            insight_type="red"
        )
        veh_data = df.groupby('vehicle_type')['is_churned'].mean().reset_index()
        veh_data['churn_pct'] = (veh_data['is_churned'] * 100).round(1)

        fig2 = px.bar(
            veh_data, x='vehicle_type', y='churn_pct',
            color='vehicle_type',
            color_discrete_map={'bike': COLORS['red'], 'auto': COLORS['blue']},
            text='churn_pct',
            labels={'vehicle_type': '', 'churn_pct': 'Churn Rate (%)'}
        )
        fig2.update_traces(
            texttemplate='<b>%{text}%</b>',
            textposition='outside',
            marker_line_width=0,
            width=0.35,
        )
        fig2.add_hline(
            y=avg_churn,
            line_dash='dot',
            line_color='#BBBBBB',
            annotation_text=f'  Fleet avg: {avg_churn:.1f}%',
            annotation_font=dict(size=10, color='#BBBBBB', family=FONT),
            annotation_position='right'
        )
        fig2.update_layout(
            **PLOTLY_LAYOUT,
            height=300,
            showlegend=False,
            title=None,
            yaxis_title='Churn Rate (%)',
        )
        st.plotly_chart(fig2, use_container_width=True)

    st.markdown("<div class='section-label'>Behavioral Signal</div>", unsafe_allow_html=True)
    chart_card(
        tag="Weekly Ride Pattern",
        title="Ride Decay Curve — Churned vs Active Captains",
        subtitle="Churned captains start dropping rides as early as Week 2, giving a roughly 2-week intervention window before full dropout.",
        insight="Week 2 drop is the critical early-warning signal",
        insight_type="red"
    )

    decay = df.groupby('is_churned')[
        ['rides_week1', 'rides_week2', 'rides_week3', 'rides_week4']
    ].mean().T
    decay.columns = ['Active', 'Churned']
    decay.index = ['Week 1', 'Week 2', 'Week 3', 'Week 4']

    fig3 = go.Figure()
    fig3.add_trace(go.Scatter(
        x=decay.index, y=decay['Active'],
        mode='lines+markers',
        name='Active',
        line=dict(color=COLORS['green'], width=3),
        marker=dict(size=9, symbol='circle', line=dict(color='white', width=2.5)),
        fill='tozeroy',
        fillcolor='rgba(29,185,84,0.07)'
    ))
    fig3.add_trace(go.Scatter(
        x=decay.index, y=decay['Churned'],
        mode='lines+markers',
        name='Churned',
        line=dict(color=COLORS['red'], width=3),
        marker=dict(size=9, symbol='circle', line=dict(color='white', width=2.5)),
        fill='tozeroy',
        fillcolor='rgba(229,57,53,0.07)'
    ))
    fig3.add_annotation(
        x='Week 4', y=decay['Active']['Week 4'],
        text=f"  {decay['Active']['Week 4']:.1f} rides",
        showarrow=False,
        font=dict(color=COLORS['green'], size=11, family=FONT),
        xanchor='left'
    )
    fig3.add_annotation(
        x='Week 4', y=decay['Churned']['Week 4'],
        text=f"  {decay['Churned']['Week 4']:.1f} rides",
        showarrow=False,
        font=dict(color=COLORS['red'], size=11, family=FONT),
        xanchor='left'
    )
    fig3.update_layout(
        **PLOTLY_LAYOUT,
        height=300,
        showlegend=True,
        title=None,
        yaxis_title='Avg Rides per Week',
        xaxis_title=None,
        legend=dict(
            orientation='h',
            yanchor='bottom', y=1.02,
            xanchor='right', x=1,
            font=dict(size=12, family=FONT),
            bgcolor='rgba(0,0,0,0)',
            borderwidth=0,
        )
    )
    st.plotly_chart(fig3, use_container_width=True)

    st.markdown("<div class='section-label'>Immediate Priorities</div>", unsafe_allow_html=True)
    st.markdown("""
    <div class='chart-card'>
        <span class='chart-tag'>Action Required</span>
        <div class='chart-title'>Top 20 Highest Risk Captains</div>
        <div class='chart-subtitle'>Sorted by churn probability. These captains need outreach within 48 hours.</div>
    </div>
    """, unsafe_allow_html=True)

    top_risk = df[['captain_id', 'city', 'vehicle_type', 'captain_type',
                   'rides_week4', 'estimated_daily_earnings',
                   'churn_probability', 'risk_segment']] \
        .sort_values('churn_probability', ascending=False).head(20).copy()
    top_risk['churn_probability'] = (top_risk['churn_probability'] * 100).round(1).astype(str) + '%'
    top_risk['estimated_daily_earnings'] = 'Rs.' + top_risk['estimated_daily_earnings'].astype(int).astype(str)
    top_risk.columns = ['Captain ID', 'City', 'Vehicle', 'Type',
                        'Week 4 Rides', 'Daily Earnings', 'Churn %', 'Risk']
    st.dataframe(top_risk, use_container_width=True, hide_index=True)

# ================================================
# PAGE 2 — PREDICT CAPTAIN
# ================================================
elif page == "Predict Captain":

    st.markdown("""
    <div class='page-banner'>
        <h1>Single Captain Churn Predictor</h1>
        <p>Enter captain behavior data to get churn probability and intervention recommendation</p>
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### Captain Profile")
        vehicle_type = st.selectbox("Vehicle Type", ["bike", "auto"])
        captain_type = st.selectbox("Captain Type", ["parttime", "fulltime"])
        city = st.selectbox("City", ["Bengaluru", "Hyderabad", "Chennai", "Pune", "Delhi"])
        rides_week1 = st.slider("Rides in Week 1", 0, 120, 30)
        rides_week2 = st.slider("Rides in Week 2", 0, 120, 22)
        rides_week3 = st.slider("Rides in Week 3", 0, 100, 15)
        rides_week4 = st.slider("Rides in Week 4", 0, 100, 8)

    with col2:
        st.markdown("#### Behavioral Features")
        cancellation_rate = st.slider("Cancellation Rate", 0.0, 0.5, 0.15, 0.01)
        incentive_claimed = st.selectbox("Incentive Claimed?", [1, 0],
                                          format_func=lambda x: "Yes" if x == 1 else "No")
        streak_completed = st.selectbox("Streak Completed?", [0, 1],
                                         format_func=lambda x: "Yes" if x == 1 else "No")
        zone_switches = st.slider("Zone Switches per Week", 0, 10, 3)
        support_tickets = st.slider("Support Tickets Raised", 0, 5, 1)
        night_rides_ratio = st.slider("Night Rides Ratio (10PM-6AM)", 0.0, 1.0, 0.4)

    is_bike = 1 if vehicle_type == 'bike' else 0
    is_parttime = 1 if captain_type == 'parttime' else 0
    is_bengaluru = 1 if city == 'Bengaluru' else 0
    avg_fare_per_km = 11 if is_bike else 19
    avg_ride_duration = 17 if is_bike else 24
    ride_decay_rate = round((rides_week1 - rides_week4) / (rides_week1 + 1), 3)
    estimated_daily_earnings = round(
        (rides_week4 / 6) * avg_fare_per_km * 8 * (1 - cancellation_rate), 0
    )
    rides_decay_x_parttime = ride_decay_rate * is_parttime
    low_earnings_x_bike = estimated_daily_earnings * is_bike
    cancellation_x_zone = cancellation_rate * zone_switches
    week4_x_incentive = rides_week4 * incentive_claimed

    features = pd.DataFrame([{
        'rides_week1': rides_week1, 'rides_week2': rides_week2,
        'rides_week3': rides_week3, 'rides_week4': rides_week4,
        'ride_decay_rate': ride_decay_rate, 'night_rides_ratio': night_rides_ratio,
        'peak_hour_ratio': 0.35, 'cancellation_rate': cancellation_rate,
        'avg_ride_duration': avg_ride_duration, 'avg_fare_per_km': avg_fare_per_km,
        'petrol_cost_sensitivity': 0.45, 'estimated_daily_earnings': estimated_daily_earnings,
        'zone_switches': zone_switches, 'incentive_claimed': incentive_claimed,
        'streak_completed': streak_completed, 'app_opens_per_day': 5,
        'support_tickets': support_tickets, 'is_bike': is_bike,
        'is_parttime': is_parttime, 'is_bengaluru': is_bengaluru,
        'rides_decay_x_parttime': rides_decay_x_parttime,
        'low_earnings_x_bike': low_earnings_x_bike,
        'cancellation_x_zone': cancellation_x_zone,
        'week4_x_incentive': week4_x_incentive
    }])

    st.markdown("<br>", unsafe_allow_html=True)

    if st.button("Run Churn Prediction", type="primary"):
        prob = model.predict_proba(features)[0][1]

        st.markdown("<br>", unsafe_allow_html=True)
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Churn Probability", f"{prob:.1%}")
        with col2:
            if prob >= 0.65:
                st.metric("Risk Segment", "High Risk")
            elif prob >= 0.35:
                st.metric("Risk Segment", "Medium Risk")
            else:
                st.metric("Risk Segment", "Low Risk")
        with col3:
            st.metric("Est. Daily Earnings", f"Rs.{estimated_daily_earnings:,.0f}")

        bar_color = COLORS['red'] if prob >= 0.65 else COLORS['yellow'] if prob >= 0.35 else COLORS['green']
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=prob * 100,
            number={'suffix': '%', 'font': {'size': 34, 'family': FONT, 'color': bar_color}},
            title={'text': "Churn Probability", 'font': {'size': 14, 'family': FONT, 'color': COLORS['subtext']}},
            delta={'reference': 40, 'increasing': {'color': COLORS['red']}, 'decreasing': {'color': COLORS['green']}},
            gauge={
                'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': '#E0E0E0',
                         'tickfont': {'size': 10, 'color': COLORS['subtext']}},
                'bar': {'color': bar_color, 'thickness': 0.22},
                'bgcolor': 'white',
                'borderwidth': 0,
                'steps': [
                    {'range': [0, 35],  'color': '#F0FFF4'},
                    {'range': [35, 65], 'color': '#FFFDE7'},
                    {'range': [65, 100], 'color': '#FFF0F0'}
                ],
                'threshold': {
                    'line': {'color': COLORS['black'], 'width': 3},
                    'thickness': 0.8,
                    'value': prob * 100
                }
            }
        ))
        fig_gauge.update_layout(
            paper_bgcolor='white', height=270,
            margin=dict(t=40, b=10, l=40, r=40),
            font=dict(family=FONT)
        )
        st.plotly_chart(fig_gauge, use_container_width=True)

        if prob >= 0.65:
            st.error(f"""
            **IMMEDIATE ACTION REQUIRED — {prob:.1%} Churn Risk**

            Recommended Interventions:
            - Send Rs.200 retention bonus within 48 hours
            - Reassign to high-demand zone in {city}
            - Personal outreach call from captain support team
            - Offer guaranteed minimum earnings for next 7 days
            """)
        elif prob >= 0.35:
            st.warning(f"""
            **MONITOR CLOSELY — {prob:.1%} Churn Risk**

            Recommended Interventions:
            - Push notification with streak bonus offer
            - Highlight peak hour earning opportunities
            - Monitor ride count again next week
            """)
        else:
            st.success(f"""
            **LOW RISK — {prob:.1%} Churn Probability**

            No intervention needed. Standard engagement only.
            Continue monitoring weekly ride patterns.
            """)

# ================================================
# PAGE 3 — ROI CALCULATOR
# ================================================
elif page == "ROI Calculator":

    st.markdown("""
    <div class='page-banner'>
        <h1>Retention ROI Calculator</h1>
        <p>Adjust assumptions to model the business impact of early captain intervention</p>
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns([1, 1])

    with col1:
        st.markdown("#### Adjust Assumptions")
        avg_rides_day = st.slider("Avg rides per day (active captain)", 5, 20, 10)
        avg_fare = st.slider("Avg fare per ride (Rs.)", 50, 150, 90)
        commission = st.slider("Platform commission %", 10, 30, 18)
        days_retained = st.slider("Days retained after intervention", 15, 90, 30)
        success_rate = st.slider("Intervention success rate %", 10, 60, 30)
        bonus_amount = st.slider("Retention bonus per captain (Rs.)", 100, 500, 200)

    revenue_per_day = avg_rides_day * avg_fare * (commission / 100)
    revenue_per_saved = revenue_per_day * days_retained
    high_risk_count = len(df[df['risk_segment'] == 'High Risk'])
    actual_churners = int(high_risk_count * 0.68)
    captains_saved = int(actual_churners * (success_rate / 100))
    intervention_cost = bonus_amount + 50
    total_spend = high_risk_count * intervention_cost
    revenue_recovered = captains_saved * revenue_per_saved
    net_roi = revenue_recovered - total_spend
    roi_pct = (net_roi / total_spend) * 100

    with col2:
        st.markdown("#### Calculated Results")
        m1, m2 = st.columns(2)
        with m1:
            st.metric("High Risk Captains", f"{high_risk_count:,}")
            st.metric("Captains Saved", f"{captains_saved:,}")
            st.metric("Total Spend", f"Rs.{total_spend:,.0f}")
        with m2:
            st.metric("Revenue / Captain", f"Rs.{revenue_per_saved:,.0f}")
            st.metric("Revenue Recovered", f"Rs.{revenue_recovered:,.0f}")
            st.metric("Net ROI", f"{roi_pct:.0f}%")

    st.markdown("<div class='section-label'>Financial Breakdown</div>", unsafe_allow_html=True)
    chart_card(
        tag="Business Impact",
        title="ROI Waterfall — Revenue vs Spend",
        subtitle="Every rupee spent on intervention is expected to return multiples in recovered platform revenue.",
        insight=f"Net return of Rs.{net_roi:,.0f} on Rs.{total_spend:,.0f} spend" if net_roi > 0 else "Adjust assumptions to improve ROI",
        insight_type="green" if net_roi > 0 else "red"
    )

    fig_wf = go.Figure(go.Waterfall(
        name="ROI",
        orientation="v",
        measure=["absolute", "relative", "total"],
        x=["Intervention Spend", "Revenue Recovered", "Net Profit"],
        y=[-total_spend, revenue_recovered, 0],
        text=[f"-Rs.{total_spend:,.0f}", f"+Rs.{revenue_recovered:,.0f}", f"Rs.{net_roi:,.0f}"],
        textposition="outside",
        textfont=dict(family=FONT, size=12, color=COLORS['text']),
        connector={"line": {"color": "#DDDDDD", "width": 1.5, "dash": "dot"}},
        decreasing={"marker": {"color": COLORS['red'],   "line": {"width": 0}}},
        increasing={"marker": {"color": COLORS['green'], "line": {"width": 0}}},
        totals={"marker":    {"color": COLORS['yellow'], "line": {"width": 0}}}
    ))
    fig_wf.update_layout(
        **PLOTLY_LAYOUT,
        height=350,
        title=None,
        showlegend=False,
        yaxis_title='Amount (Rs.)',
    )
    st.plotly_chart(fig_wf, use_container_width=True)

# ================================================
# PAGE 4 — CITY ACTION PLAN
# ================================================
elif page == "City Action Plan":

    st.markdown("""
    <div class='page-banner'>
        <h1>City-Level Retention Action Plan</h1>
        <p>Prioritized intervention targets across all Rapido cities</p>
    </div>
    """, unsafe_allow_html=True)

    city_data = df.groupby('city').agg(
        total_captains=('captain_id', 'count'),
        high_risk=('risk_segment', lambda x: (x == 'High Risk').sum()),
        avg_churn_prob=('churn_probability', 'mean'),
        avg_daily_earnings=('estimated_daily_earnings', 'mean')
    ).reset_index()

    city_data['churn_prob_pct'] = (city_data['avg_churn_prob'] * 100).round(1)
    city_data['avg_daily_earnings'] = city_data['avg_daily_earnings'].round(0)
    city_data['retention_budget'] = city_data['high_risk'] * 250
    city_data = city_data.sort_values('high_risk', ascending=False)

    st.markdown("<div class='section-label'>City Intelligence</div>", unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        chart_card(
            tag="Volume Priority",
            title="High Risk Captains by City",
            subtitle="Bar height shows scale of problem. Color intensity reflects churn rate urgency.",
            insight="Bengaluru leads in volume — highest absolute intervention need",
            insight_type="red"
        )
        fig_city = px.bar(
            city_data, x='city', y='high_risk',
            color='churn_prob_pct',
            color_continuous_scale=[
                [0.0, '#FFF9E6'],
                [0.5, '#F5C518'],
                [1.0, '#D4A800']
            ],
            text='high_risk',
            labels={'high_risk': 'High Risk Captains', 'churn_prob_pct': 'Avg Churn %', 'city': ''}
        )
        fig_city.update_traces(
            textposition='outside',
            textfont=dict(family=FONT, size=12),
            marker_line_width=0,
            width=0.55,
        )
        fig_city.update_layout(
            **PLOTLY_LAYOUT,
            height=330,
            title=None,
            showlegend=False,
            yaxis_title='High Risk Captains',
            coloraxis_colorbar=dict(
                title=dict(
                    text='Churn %',
                    font=dict(size=10, color=COLORS['subtext'], family=FONT)
                ),
                tickfont=dict(size=10, color=COLORS['subtext'], family=FONT),
                thickness=10,
                len=0.65,
            )
        )
        st.plotly_chart(fig_city, use_container_width=True)

    with col2:
        chart_card(
            tag="Budget Allocation",
            title="Retention Budget Split by City",
            subtitle="Proportional spend based on high-risk captain count at Rs.250 per captain.",
            insight="Bengaluru + Hyderabad account for 60%+ of total budget",
            insight_type="blue"
        )
        fig_budget = px.pie(
            city_data, values='retention_budget', names='city',
            color_discrete_sequence=[
                COLORS['black'], COLORS['yellow'], '#444444',
                COLORS['yellow_dk'], '#888888'
            ],
            hole=0.50
        )
        fig_budget.update_traces(
            textfont=dict(family=FONT, size=12),
            marker=dict(line=dict(color='white', width=3)),
            pull=[0.02] * len(city_data),
            hovertemplate='<b>%{label}</b><br>Budget: Rs.%{value:,.0f}<br>Share: %{percent}<extra></extra>'
        )
        fig_budget.update_layout(
            paper_bgcolor='white',
            height=330,
            margin=dict(t=20, b=20, l=20, r=20),
            font=dict(family=FONT),
            legend=dict(font=dict(size=12, family=FONT), bgcolor='rgba(0,0,0,0)'),
            showlegend=True,
        )
        st.plotly_chart(fig_budget, use_container_width=True)

    st.markdown("<div class='section-label'>City Breakdown</div>", unsafe_allow_html=True)
    st.markdown("""
    <div class='chart-card'>
        <span class='chart-tag'>Data Table</span>
        <div class='chart-title'>Full City Breakdown</div>
        <div class='chart-subtitle'>Sorted by number of high-risk captains. Use this to prioritise city-level outreach campaigns.</div>
    </div>
    """, unsafe_allow_html=True)

    display_df = city_data[['city', 'total_captains', 'high_risk',
                             'churn_prob_pct', 'avg_daily_earnings', 'retention_budget']].copy()
    display_df.columns = ['City', 'Total Captains', 'High Risk',
                          'Avg Churn %', 'Avg Daily Earnings (Rs.)', 'Budget Needed (Rs.)']
    st.dataframe(display_df, use_container_width=True, hide_index=True)

    st.info("""
    Key Insight: Bengaluru has the most high-risk captains but Hyderabad and Chennai
    have higher churn rates (48.5–48.8%).

    Strategy: Prioritize Bengaluru for volume impact, Hyderabad/Chennai for rate reduction.
    """)
