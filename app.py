import streamlit as st
from streamlit_option_menu import option_menu
from modules import (
    data_cleaning, feature_engineering,
    ml_classification, ml_regression, ml_time_series,
    dl_value_prediction, dl_time_series,
    model_deploy
)

st.set_page_config(page_title="AI智能平台", layout="wide")

# --------- option-menu 侧边栏 ---------
with st.sidebar:
    selected = option_menu(
        "AI智能平台",  # 顶部标题
        [
            "🧹 数据清洗", "🛠️ 特征构造",
            "🤖 机器学习", "🧠 深度学习",
            "🚀 模型部署", "📄 关于"
        ],
        icons=["", "", "robot", "cpu",  "rocket", "file-earmark-text"],
        menu_icon="cast",
        default_index=0,
        styles={
            "container": {"padding": "0!important", "background-color": "#f6f8fa"},
            "icon": {"color": "#1976d2", "font-size": "18px"},
            "nav-link": {
                "font-size": "16px",
                "text-align": "left",
                "margin": "2px 0",
                "border-radius": "8px",
                "color": "#34495e",
                "font-weight": "400",
            },
            "nav-link-selected": {
                "background-color": "#e6f0fa",
                "color": "#1976d2",
                "font-weight": "bold",
            },
        }
    )

# --------- 页面内容区 ---------
# st.title("AI 智能建模平台")

if selected == "🧹 数据清洗":
    data_cleaning.run()
elif selected == "🛠️ 特征构造":
    feature_engineering.run()
elif selected == "🤖 机器学习":
    ml_menu = option_menu(
        None,
        ["分类", "回归", "时序预测"],
        icons=["bar-chart", "graph-up", "clock-history"],
        orientation="horizontal",
        styles={
            "container": {"padding": "0!important", "background-color": "#f6f8fa"},
            "icon": {"color": "#1976d2", "font-size": "16px"},
            "nav-link": {"font-size": "15px", "color": "#34495e"},
            "nav-link-selected": {"background-color": "#e6f0fa", "color": "#1976d2", "font-weight": "bold"},
        }
    )
    if ml_menu == "分类":
        ml_classification.run()
    elif ml_menu == "回归":
        ml_regression.run()
    elif ml_menu == "时序预测":
        ml_time_series.run()
elif selected == "🧠 深度学习":
    dl_menu = option_menu(
        None,
        ["时序预测", "数值预测"],
        icons=["clock-history", "123"],
        orientation="horizontal",
        styles={
            "container": {"padding": "0!important", "background-color": "#f6f8fa"},
            "icon": {"color": "#1976d2", "font-size": "16px"},
            "nav-link": {"font-size": "15px", "color": "#34495e"},
            "nav-link-selected": {"background-color": "#e6f0fa", "color": "#1976d2", "font-weight": "bold"},
        }
    )
    if dl_menu == "数值预测":
        dl_value_prediction.run()
    elif dl_menu == "时序预测":
        dl_time_series.run()
elif selected == "🚀 模型部署":
    model_deploy.run()
elif selected == "📄 关于":
    st.subheader("关于本平台")
    st.markdown("本平台由 Streamlit 驱动，旨在为用户提供一站式AI建模体验。")

st.markdown("---")
st.caption("© 2025 AI智能平台 - Powered by Streamlit")
