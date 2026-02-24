import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from datetime import datetime
import os

# ================= CONFIG & INTERFACE =================
st.set_page_config(page_title="AI 2-TINH LOTOBET v2", layout="wide", page_icon="🎯")

st.markdown("""
    <style>
    .main { background-color: #f0f2f6; }
    .stButton>button { width: 100%; border-radius: 5px; height: 3em; background-color: #007bff; color: white; }
    .prediction-box { padding: 20px; border-radius: 15px; border: 2px solid #ff4b4b; background-color: white; text-align: center; }
    .skip-box { padding: 20px; border-radius: 15px; border: 2px solid #6c757d; background-color: #e9ecef; text-align: center; }
    </style>
    """, unsafe_allow_html=True)

DATA_FILE = "lotobet_history.csv"

# ================= LOGIC AI CHUẨN v2 =================
class LotobetAI_V2:
    def __init__(self):
        self.min_confidence = 75  # Theo đặc tả: >=75% mới là điều kiện tốt
        
    def clean_data(self, df):
        """Lọc dữ liệu rác, chỉ lấy đúng 5 chữ số"""
        valid_matrix = []
        for val in df['numbers'].astype(str):
            nums = [int(d) for d in val.strip() if d.isdigit()]
            if len(nums) == 5:
                valid_matrix.append(nums)
        return np.array(valid_matrix)

    def analyze_numbers(self, matrix):
        """Bước 3: Phân tích số đơn (0-9)"""
        if len(matrix) < 5: return None
        
        analysis = {}
        total_draws = len(matrix)
        
        for num in range(10):
            # Tìm các kỳ xuất hiện (index)
            appearances = np.where(np.any(matrix == num, axis=1))[0]
            
            # 5. Trọng số thời gian
            recent_3 = sum(1 for row in matrix[-3:] if num in row)
            recent_5 = sum(1 for row in matrix[-5:] if num in row)
            recent_10 = sum(1 for row in matrix[-10:] if num in row)
            
            # Tính khoảng cách (Gap)
            gap_since_last = (total_draws - 1 - appearances[-1]) if len(appearances) > 0 else 99
            
            # 6. Phân loại trạng thái
            if recent_3 >= 2: state = "NGUY HIỂM" # Ra dồn
            elif recent_5 >= 3: state = "NÓNG"
            elif 1 <= recent_10 <= 2 and gap_since_last >= 3: state = "ỔN ĐỊNH"
            elif gap_since_last >= 5: state = "HỒI"
            else: state = "YẾU"
            
            analysis[num] = {
                "state": state,
                "gap": gap_since_last,
                "freq_10": recent_10,
                "last_val": matrix[-1] # Dùng để kiểm tra cầu lặp
            }
        return analysis

    def get_prediction(self, matrix):
        """Bước 7 & 8: Logic ghép cặp và Không đánh"""
        analysis = self.analyze_numbers(matrix)
        if not analysis: return None, "DỮ LIỆU ÍT", ["Cần ít nhất 10 kỳ để phân tích chính xác."]
        
        reasons_to_skip = []
        
        # Kiểm tra điều kiện KHÔNG ĐÁNH (Mục 8)
        hot_count = sum(1 for v in analysis.values() if v['state'] in ["NÓNG", "NGUY HIỂM"])
        if hot_count >= 6: reasons_to_skip.append("Thị trường quá NÓNG (nhiều số ra dồn)")
        
        last_draw = matrix[-1]
        repeat_count = sum(1 for n in last_draw if analysis[n]['state'] == "NGUY HIỂM")
        if repeat_count >= 3: reasons_to_skip.append("Nhiều số vừa ra lại (Cầu lặp nhiễu)")

        if reasons_to_skip:
            return None, "KHÔNG ĐÁNH KỲ NÀY", reasons_to_skip

        # Ghép cặp 2 tinh (Loại số chập - Mục 1)
        potential_pairs = []
        for i in range(10):
            for j in range(i + 1, 10): # Tự động loại i=j (số chập)
                s1, s2 = analysis[i], analysis[j]
                
                # Logic loại trừ (Mục 6)
                if s1['state'] == s2['state'] and s1['state'] in ["NÓNG", "NGUY HIỂM", "YẾU"]:
                    continue
                
                # Tính điểm tự tin (%)
                score = 50
                if s1['state'] == "ỔN ĐỊNH" and s2['state'] == "HỒI": score = 85
                if s1['state'] == "ỔN ĐỊNH" and s2['state'] == "ỔN ĐỊNH": score = 80
                if s1['state'] == "HỒI" and s2['state'] == "HỒI": score = 75
                
                if score >= self.min_confidence:
                    potential_pairs.append({"pair": f"{i}{j}", "score": score})

        potential_pairs.sort(key=lambda x: x['score'], reverse=True)
        
        if not potential_pairs:
            return None, "KHÔNG ĐÁNH KỲ NÀY", ["Không có cặp số nào đạt ngưỡng an toàn (>=75%)"]
            
        return potential_pairs[:1], "ĐÁNH", [] # Tối đa 1 cặp tốt nhất như yêu cầu

# ================= HÀM XỬ LÝ DỮ LIỆU =================
def load_data():
    if os.path.exists(DATA_FILE):
        return pd.read_csv(DATA_FILE)
    return pd.DataFrame(columns=["time", "numbers"])

def save_data(raw_text):
    lines = [line.strip() for line in raw_text.split('\n') if len(line.strip()) == 5 and line.strip().isdigit()]
    if not lines: return 0
    
    df = load_data()
    new_data = pd.DataFrame({"time": [datetime.now().strftime("%H:%M:%S")] * len(lines), "numbers": lines})
    df = pd.concat([df, new_data], ignore_index=True).tail(100) # Giữ 100 kỳ gần nhất
    df.to_csv(DATA_FILE, index=False)
    return len(lines)

# ================= GIAO DIỆN STREAMLIT =================
def main():
    st.title("🎯 AI LOTOBET 2-TINH - CHUẨN V2.0")
    st.caption("Hệ thống phân tích dựa trên nhịp cầu và trạng thái số đơn - Không đánh số chập")
    
    ai = LotobetAI_V2()
    
    col_input, col_display = st.columns([1, 2])
    
    with col_input:
        st.subheader("📥 Nhập dữ liệu")
        raw_input = st.text_area("Nhập 5 số viết liền (mỗi kỳ 1 dòng):", height=200, placeholder="Ví dụ:\n12345\n67890")
        if st.button("💾 Cập nhật & Phân tích"):
            added = save_data(raw_input)
            if added > 0:
                st.success(f"Đã thêm {added} kỳ!")
                st.rerun()
            else:
                st.error("Dữ liệu không hợp lệ (Phải đúng 5 chữ số)")

        if st.button("🗑️ Xóa dữ liệu cũ"):
            if os.path.exists(DATA_FILE):
                os.remove(DATA_FILE)
                st.warning("Đã xóa lịch sử")
                st.rerun()

    with col_display:
        df = load_data()
        if df.empty:
            st.info("Hãy nhập ít nhất 10 kỳ để AI bắt đầu làm việc.")
            return

        matrix = ai.clean_data(df)
        
        # --- PHẦN DỰ ĐOÁN CHÍNH ---
        st.subheader("📊 Kết quả phân tích")
        preds, status, reasons = ai.get_prediction(matrix)
        
        if status == "ĐÁNH":
            for p in preds:
                st.markdown(f"""
                <div class="prediction-box">
                    <h3 style="color:#666;">GỢI Ý DUY NHẤT</h3>
                    <h1 style="font-size: 80px; color: #ff4b4b; margin: 10px 0;">{p['pair']}</h1>
                    <h4 style="color: #28a745;">ĐỘ TỰ TIN: {p['score']}%</h4>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="skip-box">
                <h2 style="color:#6c757d;">🚫 {status}</h2>
                <ul style="text-align: left; display: inline-block;">
                    {"".join([f"<li>{r}</li>" for r in reasons])}
                </ul>
            </div>
            """, unsafe_allow_html=True)

        # --- BIỂU ĐỒ TRỰC QUAN ---
        st.divider()
        analysis = ai.analyze_numbers(matrix)
        if analysis:
            st.subheader("📈 Trạng thái 10 số đơn")
            chart_df = pd.DataFrame([
                {"Số": str(i), "Khoảng cách (Gap)": v['gap'], "Trạng thái": v['state']} 
                for i, v in analysis.items()
            ])
            fig = px.bar(chart_df, x="Số", y="Khoảng cách (Gap)", color="Trạng thái",
                         title="Khoảng cách kỳ chưa ra của các số đơn",
                         color_discrete_map={"NGUY HIỂM": "red", "NÓNG": "orange", "ỔN ĐỊNH": "green", "HỒI": "blue", "YẾU": "gray"})
            st.plotly_chart(fig, use_container_width=True)

if __name__ == "__main__":
    main()
