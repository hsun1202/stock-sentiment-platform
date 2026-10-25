"""
結合市場情緒與財報資訊之股票虛擬交易平台：試作版（本機）。版面架構參考組員 jameschen108/tw-stock-quant 的 frontend/app.py。
cd D:/專題claude && streamlit run platform_app/app.py
"""
import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402

st.set_page_config(page_title="情緒＋財報 虛擬交易平台（試作）", page_icon=":material/monitoring:", layout="wide",
                   initial_sidebar_state="collapsed")
st.markdown(common.CSS, unsafe_allow_html=True)
page = st.navigation([
    st.Page("p1_overview.py", title="今日名單", icon=":material/dashboard:", default=True),
    st.Page("p2_stock.py", title="個股", icon=":material/query_stats:", url_path="stock"),
    st.Page("p4_trade.py", title="虛擬交易", icon=":material/account_balance_wallet:", url_path="trade"),
    st.Page("p3_research.py", title="研究結果", icon=":material/science:", url_path="research"),
], position="top")
common.header()
page.run()
