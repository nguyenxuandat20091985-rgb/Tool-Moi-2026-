import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from datetime import datetime
import os

# ================= CONFIGURATION =================
st.set_page_config(page_title="AI LOTOBET 2-TINH v2", layout="wide", page_icon="🎯")

# Giao diện tối giản, tập trung vào kết quả
st.markdown("""
    <style>
    .main { background-color: #f0f2f6; }
    .prediction-box { background-color: #ffffff; padding: 25px; border-radius: 15px; border-left: 10px solid #e74c3c; box-shadow: 0 4px 6px rgba(0,0,0,0.1); }
    .status-safe { color: #27ae60; font-weight: bold; }
    .status-risk { color: #e74c3c; font-weight: bold; }
    </style>
    """, unsafe_allow_html=True)

DATA_FILE = "lotobet_history.csv"

# ================= CORE AI LOGIC v2 =================
class LotobetV2:
    def __init__(self):
        self.MIN_DRAWS = 10
        self.STATES = ["NÓNG", "ỔN ĐỊNH", "YẾU", "NGUY HIỂM"]

    def clean_input(self, raw_data):
        """Lọc dữ liệu: Chỉ lấy dòng đúng 5 số"""
        cleaned = []
        lines = raw_data.strip().split('\n')
        for line in lines:
            nums = [int(d) for d in line.strip() if d.isdigit()]
            if len(nums) == 5:
                cleaned.append(nums)
        return cleaned

    def analyze_numbers(self, matrix):
        """Phân tích số đơn 0-9 theo đặc tả"""
        analysis = {}
        total_draws = len(matrix)
        matrix_np = np.array(matrix)

        for num in range(10):
            # 1. Tìm các kỳ xuất hiện
            appears = []
            for idx, row in enumerate(matrix):
                if num in row:
                    appears.append(idx)
            
            # 2. Tính khoảng cách (Gap)
            gaps = np.diff(appears) if len(appears) > 1 else []
            last_appearance = (total_draws - 1) - appears[-1] if appears else 99
            
            # 3. Tần suất 5 kỳ gần nhất
            recent_5 = matrix[-5:]
            freq_5 = sum(1 for row in recent_5 if num in row)
            
            # 4. Phân loại trạng thái (Điều 6)
            if freq_5 >= 3: 
                state = "NÓNG"
            elif last_appearance == 0 or (len(gaps) > 0 and gaps[-1] == 1):
                state = "NGUY HIỂM"
            elif 3 <= last_appearance <= 7:
                state = "ỔN ĐỊNH"
            else:
                state = "YẾU"

            # 5. Nhận diện cầu (Điều 4)
            bridge = "BÌNH THƯỜNG"
            if len(gaps) >= 2 and gaps[-1] == gaps[-2] and gaps[-1] in [2, 3]:
                bridge = "CẦU NHẢY"
            elif last_appearance >= 5 and last_appearance <= 8:
                bridge = "CẦU HỒI"
            elif last_appearance == 0 and freq_5 >= 3:
                bridge = "CẦU BỆT"

            analysis[num] = {
                "state": state,
                "bridge": bridge,
                "last_appearance": last_appearance,
                "freq_5": freq_5
            }
        return analysis

    def get_predictions(self, matrix):
        """Logic ghép cặp & lọc (Điều 7 & 8)"""
        if len(matrix) < self.MIN_DRAWS:
            return None, "DỮ LIỆU ÍT", [f"Cần thêm ít nhất {self.MIN_DRAWS - len(matrix)} kỳ."]

        analysis = self.analyze_numbers(matrix)
        
        # Kiểm tra điều kiện KHÔNG ĐÁNH (Điều 8)
        hot_nums = [n for n, v in analysis.items() if v['state'] == "NÓNG"]
        recent_repeats = [n for n, v in analysis.items() if v['last_appearance'] == 0]
        
        if len(hot_nums) >= 6:
            return None, "KHÔNG ĐÁNH", ["Thị trường quá NÓNG (nhiều số ra dồn dập)."]
        if len(recent_repeats) >= 4:
            return None, "KHÔNG ĐÁNH", ["Quá nhiều số vừa ra kỳ trước, cầu đang nhiễu."]

        # Ghép cặp (Điều 7)
        candidates = []
        for i in range(10):
            for j in range(i + 1, 10):
                # ❌ Loại số chập (Điều 1) đã tự động loại vì i != j trong vòng lặp
                s1, s2 = analysis[i], analysis[j]
                
                # ❌ Không ghép các tổ hợp cấm (Điều 6)
                if s1['state'] == s2['state'] and s1['state'] in ["NÓNG", "NGUY HIỂM", "YẾU"]:
                    continue
                
                score = 50
                # ✅ Ưu tiên 1 số ổn định + 1 số hồi (Điều 7)
                if (s1['state'] == "ỔN ĐỊNH" and s2['bridge'] == "CẦU HỒI") or \
                   (s2['state'] == "ỔN ĐỊNH" and s1['bridge'] == "CẦU HỒI"):
                    score += 35
                
                # ✅ Ưu tiên nhảy nhịp + ổn định
                if (s1['bridge'] == "CẦU NHẢY" and s2['state'] == "ỔN ĐỊNH") or \
                   (s2['bridge'] == "CẦU NHẢY" and s1['state'] == "ỔN ĐỊNH"):
                    score += 30

                # Giảm trọng số nếu vừa ra kỳ trước (Điều 5)
                if s1['last_appearance'] == 0: score -= 20
                if s2['last_appearance'] == 0: score -= 20

                if score >= 75:
                    candidates.append({"pair": f"{i}{j}", "score": score})

        candidates.sort(key=lambda x: x['score'], reverse=True)
        
        if not candidates:
            return None, "KHÔNG ĐÁNH", ["Không có cặp số nào đạt ngưỡng an toàn (75%)."]
            
        return candidates[:1], "ĐÁNH", [] # Chỉ trả về 1 cặp tốt nhất (Điều 7)

# ================= INTERFACE =================
def main():
    st.title("🎯 AI LOTOBET 2-TINH (CHUẨN v2)")
    model = LotobetV2()

    # Sidebar nhập liệu
    with st.sidebar:
        st.header("📥 Nhập dữ liệu")
        st.info("Nhập 5 số viết liền, mỗi dòng 1 kỳ.")
        raw_data = st.text_area("Dữ liệu kết quả:", height=300, placeholder="12345\n67890\n...")
        
        if st.button("🔄 Xóa dữ liệu cũ"):
            if os.path.exists(DATA_FILE): os.remove(DATA_FILE)
            st.rerun()

    if not raw_data:
        st.warning("Vui lòng nhập dữ liệu vào ô bên trái để bắt đầu.")
        return

    # Xử lý dữ liệu
    clean_matrix = model.clean_input(raw_data)
    
    if not clean_matrix:
        st.error("Dữ liệu không hợp lệ. Vui lòng nhập đúng định dạng 5 số mỗi dòng.")
        return

    # Phân tích & Dự đoán
    col1, col2 = st.tabs(["📊 Dự đoán & Thống kê", "📈 Biểu đồ xu hướng"])

    with col1:
        preds, status, reasons = model.get_predictions(clean_matrix)
        
        if status == "KHÔNG ĐÁNH":
            st.error("🚫 **KHÔNG ĐÁNH KỲ NÀY**")
            for r in reasons:
                st.write(f"- {r}")
        elif status == "ĐÁNH":
            st.success("✅ **CƠ HỘI ĐẦU TƯ TỐT**")
            for p in preds:
                st.markdown(f"""
                <div class="prediction-box">
                    <span style="font-size: 1.2em; color: #7f8c8d;">Cặp số duy nhất:</span><br>
                    <span style="font-size: 4em; font-weight: bold; color: #2c3e50;">{p['pair']}</span><br>
                    <span style="font-size: 1.5em; color: #27ae60;">Độ tự tin: {p['score']}%</span>
                </div>
                """, unsafe_allow_html=True)
            st.caption("Lưu ý: Đánh cả hai số này trong cùng 1 đơn cược.")

        # Bảng thống kê số đơn
        st.divider()
        st.subheader("📋 Bảng trạng thái số đơn (0-9)")
        analysis = model.analyze_numbers(clean_matrix)
        stat_df = pd.DataFrame([
            {"Số": k, "Trạng thái": v['state'], "Loại cầu": v['bridge'], "Kỳ chưa ra": v['last_appearance']}
            for k, v in analysis.items()
        ])
        st.table(stat_df)

    with col2:
        st.subheader("Tần suất xuất hiện (5 kỳ gần nhất)")
        chart_data = pd.DataFrame([
            {"Số": str(k), "Tần suất": v['freq_5']} for k, v in analysis.items()
        ])
        fig = px.bar(chart_data, x='Số', y='Tần suất', color='Tần suất', color_continuous_scale='Reds')
        st.plotly_chart(fig, use_container_width=True)

if __name__ == "__main__":
    main()
