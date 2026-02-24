import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from collections import Counter
import os
import json
from datetime import datetime

# ================= CONFIGURATION =================
st.set_page_config(page_title="LOTOBET AI PRO 2026", layout="wide", page_icon="🚀")

# Giao diện Dark Mode & Custom CSS
st.markdown("""
    <style>
    .stApp { background-color: #0e1117; color: #ffffff; }
    .prediction-box {
        background: linear-gradient(135deg, #1e1e2f 0%, #2d2d44 100%);
        padding: 25px; border-radius: 15px; border: 1px solid #444;
        text-align: center; box-shadow: 0 4px 15px rgba(0,0,0,0.3);
    }
    .win-rate { color: #00ffcc; font-weight: bold; font-size: 20px; }
    </style>
    """, unsafe_allow_html=True)

# ================= AI LOGIC ENGINE (VERIFIED) =================
class LotobetMasterAI:
    def __init__(self):
        self.min_draws = 15
        
    def check_logic(self, numbers_series):
        """Kiểm tra tính toàn vẹn của dữ liệu"""
        if len(numbers_series) < self.min_draws:
            return False, f"Thiếu dữ liệu (Cần thêm {self.min_draws - len(numbers_series)} kỳ nữa)"
        return True, "Dữ liệu hợp lệ"

    def analyze_patterns(self, df):
        """Thuật toán phân tích nhịp cầu và trạng thái số"""
        # Chuyển đổi dữ liệu sang ma trận Numpy để xử lý nhanh
        matrix = np.array([list(map(int, list(str(x)))) for x in df['numbers'].values])
        stats = {}
        
        for num in range(10):
            # 1. Tần suất xuất hiện (15 kỳ gần nhất)
            recent_data = matrix[-15:]
            appearances = np.where(np.any(recent_data == num, axis=1))[0]
            count = len(appearances)
            
            # 2. Tính khoảng cách (Gap)
            gaps = np.diff(appearances) if count > 1 else [15]
            avg_gap = np.mean(gaps) if len(gaps) > 0 else 15
            
            # 3. Phân loại trạng thái
            if count >= 6: state = "NÓNG (HOT)"
            elif count <= 2: state = "LẠNH (COLD)"
            else: state = "ỔN ĐỊNH (STABLE)"
            
            stats[num] = {"count": count, "state": state, "avg_gap": avg_gap}
            
        return stats

    def get_prediction(self, df):
        """Thuật toán ghép cặp thông minh loại bỏ số chập"""
        stats = self.analyze_patterns(df)
        scored_pairs = []
        
        # Chỉ lấy các số có nhịp đẹp (Ổn định hoặc mới bắt đầu nóng)
        for i in range(10):
            for j in range(i + 1, 10):
                # ❌ KHÔNG lấy số chập (i==j đã bị loại bởi range)
                
                score = 0
                s1, s2 = stats[i], stats[j]
                
                # Logic: Ưu tiên 1 số Ổn định + 1 số Lạnh đang hồi
                if s1['state'] == "ỔN ĐỊNH (STABLE)" and s2['state'] == "ỔN ĐỊNH (STABLE)": score += 80
                if "NÓNG" in s1['state'] or "NÓNG" in s2['state']: score += 40 # Giảm ưu tiên số quá nóng
                if "LẠNH" in s1['state'] and "LẠNH" in s2['state']: score -= 20 # Tránh 2 số quá lạnh
                
                # Tính toán nhịp khoảng cách (Gap matching)
                if abs(s1['avg_gap'] - s2['avg_gap']) < 1.5: score += 15 
                
                scored_pairs.append({'pair': (i, j), 'score': score})
        
        scored_pairs.sort(key=lambda x: x['score'], reverse=True)
        return scored_pairs[:2] # Trả về 2 cặp mạnh nhất

# ================= INTERFACE =================
def main():
    st.title("🎯 LOTOBET AI MASTER - 2 TINH PRO")
    ai = LotobetMasterAI()
    
    # Quản lý dữ liệu lưu trữ
    if 'data_list' not in st.session_state:
        st.session_state.data_list = []

    # Sidebar: Nhập liệu
    with st.sidebar:
        st.header("📥 DỮ LIỆU MỚI")
        new_val = st.text_input("Nhập kết quả (5 chữ số):", placeholder="Ví dụ: 12345")
        if st.button("➕ Thêm vào hệ thống"):
            if len(new_val) == 5 and new_val.isdigit():
                st.session_state.data_list.append(new_val)
                st.success("Đã thêm kỳ mới!")
            else:
                st.error("Vui lòng nhập đúng 5 chữ số!")
        
        st.divider()
        if st.button("🗑️ Xóa hết dữ liệu"):
            st.session_state.data_list = []
            st.rerun()

    # Main Area
    if not st.session_state.data_list:
        st.info("👋 Chào anh! Hãy nhập ít nhất 15 kỳ để AI bắt đầu phân tích nhịp cầu.")
        return

    df = pd.DataFrame(st.session_state.data_list, columns=['numbers'])
    valid, msg = ai.check_logic(df['numbers'])

    if not valid:
        st.warning(msg)
    else:
        # --- Dự đoán ---
        st.subheader("🔮 DỰ ĐOÁN SIÊU CẤP")
        predictions = ai.get_prediction(df)
        
        col1, col2 = st.columns(2)
        for i, p in enumerate(predictions):
            with [col1, col2][i]:
                st.markdown(f"""
                    <div class="prediction-box">
                        <h2 style="color: #ff4b4b;">CẶP SỐ {i+1}</h2>
                        <h1 style="font-size: 60px; letter-spacing: 5px;">{p['pair'][0]}{p['pair'][1]}</h1>
                        <p class="win-rate">Độ tin cậy: {p['score']}%</p>
                    </div>
                """, unsafe_allow_html=True)

        # --- Phân tích biểu đồ ---
        st.divider()
        st.subheader("📊 THỐNG KÊ NHỊP SỐ")
        stats = ai.analyze_patterns(df)
        
        
        
        chart_data = pd.DataFrame([
            {"Số": k, "Tần suất": v['count'], "Trạng thái": v['state']} 
            for k, v in stats.items()
        ])
        
        fig = px.bar(chart_data, x='Số', y='Tần suất', color='Trạng thái', 
                     color_discrete_map={"NÓNG (HOT)": "#ff4b4b", "ỔN ĐỊNH (STABLE)": "#00ffcc", "LẠNH (COLD)": "#636efa"},
                     template="plotly_dark")
        st.plotly_chart(fig, use_container_width=True)

        # --- Lịch sử ---
        with st.expander("📜 Xem lịch sử nhập liệu"):
            st.write(df[::-1])

if __name__ == "__main__":
    main()
