import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from datetime import datetime
import os

# ================= CONFIG =================
st.set_page_config(page_title="AI LOTOBET 2-TINH v2", layout="wide")

DATA_FILE = "lotobet_data_v2.csv"

class LotobetLogicV2:
    def __init__(self):
        self.min_confidence = 75 # Ngưỡng an toàn tuyệt đối
        self.states = {
            "HOT": "NÓNG",
            "STABLE": "ỔN ĐỊNH",
            "WEAK": "YẾU",
            "RISKY": "NGUY HIỂM"
        }

    def process_data(self, df):
        """Chuyển đổi dữ liệu thô thành ma trận chuẩn"""
        matrix = []
        for val in df['numbers'].values:
            clean_val = str(val).strip()
            if len(clean_val) == 5 and clean_val.isdigit():
                matrix.append([int(d) for d in clean_val])
        return np.array(matrix)

    def analyze_numbers(self, matrix):
        """Phân tích 10 số đơn (0-9)"""
        if len(matrix) < 5: return None
        
        analysis = {}
        for num in range(10):
            # 1. Tìm vị trí xuất hiện
            appears = np.where(np.any(matrix == num, axis=1))[0]
            gaps = np.diff(appears) if len(appears) > 1 else []
            
            # 2. Tần suất 5 kỳ gần nhất
            recent_5 = matrix[-5:]
            freq_5 = sum(1 for row in recent_5 if num in row)
            
            # 3. Kỳ cuối cùng xuất hiện (cách đây bao lâu)
            last_seen = (len(matrix) - 1) - appears[-1] if len(appears) > 0 else 99
            
            # 4. Gán trạng thái theo Đặc tả
            state = self.states["STABLE"]
            if freq_5 >= 3: state = self.states["HOT"]
            if last_seen == 0: state = self.states["RISKY"] # Vừa ra kỳ trước
            if freq_5 <= 1 and last_seen > 5: state = self.states["WEAK"]

            analysis[num] = {
                "freq_5": freq_5,
                "last_seen": last_seen,
                "state": state,
                "score": self.calculate_individual_score(freq_5, last_seen, state)
            }
        return analysis

    def calculate_individual_score(self, freq, last_seen, state):
        """Tính điểm cho từng số đơn (Trọng số thời gian)"""
        score = 50
        if state == self.states["STABLE"]: score += 20
        if 3 <= last_seen <= 7: score += 15 # Ưu tiên số đã nghỉ vài kỳ (Cầu nhảy/Hồi)
        if last_seen == 0: score -= 30 # Giảm xác suất lặp (Cầu lặp)
        if freq >= 4: score -= 20 # Nguy cơ gãy cầu bệt
        return score

    def get_predictions(self, df):
        matrix = self.process_data(df)
        if len(matrix) < 10: 
            return [], "DATA_INSUFFICIENT", ["Cần tối nhất 10 kỳ để phân tích nhịp."]
        
        analysis = self.analyze_numbers(matrix)
        if not analysis: return [], "ERROR", ["Lỗi xử lý dữ liệu."]

        # --- BƯỚC: LOẠI BỎ 3 SỐ YẾU NHẤT, GIỮ LẠI 7 SỐ ---
        sorted_nums = sorted(analysis.items(), key=lambda x: x[1]['score'], reverse=True)
        top_7_nums = [item[0] for item in sorted_nums[:7]]
        
        # --- BƯỚC: GHÉP CẶP 2 TINH (LOẠI SỐ CHẬP) ---
        candidates = []
        for i in range(len(top_7_nums)):
            for j in range(i + 1, len(top_7_nums)):
                n1, n2 = top_7_nums[i], top_7_nums[j]
                
                # Không ghép số chập (đã đảm bảo vì n1 != n2)
                s1, s2 = analysis[n1], analysis[n2]
                
                # Logic loại trừ: Không ghép 2 Nóng, 2 Nguy hiểm, 2 Yếu
                bad_states = [self.states["HOT"], self.states["RISKY"], self.states["WEAK"]]
                if s1['state'] == s2['state'] and s1['state'] in bad_states:
                    continue
                
                # Tính điểm cặp
                pair_score = (s1['score'] + s2['score']) / 2
                
                # Ưu tiên: 1 Ổn định + 1 Hồi
                if (s1['state'] == self.states["STABLE"] and 3 <= s2['last_seen'] <= 6) or \
                   (s2['state'] == self.states["STABLE"] and 3 <= s1['last_seen'] <= 6):
                    pair_score += 10

                if pair_score >= self.min_confidence:
                    candidates.append({
                        "pair": f"{min(n1,n2)}{max(n1,n2)}",
                        "confidence": pair_score,
                        "details": f"{s1['state']} + {s2['state']}"
                    })

        # --- LOGIC KHÔNG ĐÁNH ---
        if not candidates or len(matrix) < 15:
            reasons = []
            if len(matrix) < 15: reasons.append("Dữ liệu quá ít để đảm bảo an toàn.")
            if not candidates: reasons.append("Không có cặp số nào đạt ngưỡng an toàn (75%).")
            return [], "SKIP", reasons

        # Sắp xếp lấy 1-2 cặp mạnh nhất
        candidates.sort(key=lambda x: x['confidence'], reverse=True)
        return candidates[:2], "PREDICT", []

# ================= INTERFACE =================
def main():
    st.markdown("<h1 style='text-align: center; color: #E74C3C;'>🎯 AI LOTOBET 2-TINH PRO v2</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center;'>Hệ thống phân tích chuẩn v2 - Ưu tiên chính xác tuyệt đối</p>", unsafe_allow_html=True)

    if 'data_df' not in st.session_state:
        if os.path.exists(DATA_FILE):
            st.session_state.data_df = pd.read_csv(DATA_FILE)
        else:
            st.session_state.data_df = pd.DataFrame(columns=["time", "numbers"])

    tab1, tab2 = st.tabs(["📊 Phân tích & Dự đoán", "📥 Nhập dữ liệu"])

    with tab2:
        st.subheader("📥 Nhập kết quả Lotobet")
        txt = st.text_area("Nhập kết quả (5 số viết liền, mỗi dòng 1 kỳ):", height=200, placeholder="12345\n67890\n...")
        if st.button("💾 Cập nhật hệ thống"):
            lines = [l.strip() for l in txt.split("\n") if len(l.strip()) == 5]
            if lines:
                new_data = pd.DataFrame({"time": [datetime.now().strftime("%H:%M:%S")] * len(lines), "numbers": lines})
                st.session_state.data_df = pd.concat([st.session_state.data_df, new_data], ignore_index=True)
                st.session_state.data_df.to_csv(DATA_FILE, index=False)
                st.success(f"Đã cập nhật thêm {len(lines)} kỳ!")
                st.rerun()

    with tab1:
        df = st.session_state.data_df
        if len(df) < 5:
            st.warning("⚠️ Cần nhập thêm dữ liệu (tối thiểu 10-15 kỳ) để AI bắt đầu làm việc.")
            return

        ai = LotobetLogicV2()
        preds, status, reasons = ai.get_predictions(df)

        # Hiển thị Dashboard
        c1, c2, c3 = st.columns(3)
        c1.metric("Tổng số kỳ", len(df))
        
        # Phân tích số đơn để vẽ biểu đồ
        matrix = ai.process_data(df)
        analysis = ai.analyze_numbers(matrix)
        
        if status == "SKIP":
            st.error("🚫 KHÔNG ĐÁNH KỲ NÀY")
            for r in reasons: st.write(f"- {r}")
        elif status == "PREDICT":
            st.success(f"✅ TÌM THẤY {len(preds)} CẶP TIỀM NĂNG")
            cols = st.columns(len(preds))
            for i, p in enumerate(preds):
                with cols[i]:
                    st.markdown(f"""
                    <div style="background: #2ECC71; padding: 20px; border-radius: 15px; text-align: center; color: white;">
                        <span style="font-size: 1.2em;">CẶP SỐ {i+1}</span>
                        <h1 style="font-size: 4em; margin: 10px 0;">{p['pair']}</h1>
                        <p>Độ tin cậy: {p['confidence']:.1f}%</p>
                    </div>
                    """, unsafe_allow_html=True)
        
        # Biểu đồ trạng thái số đơn
        st.divider()
        st.subheader("📊 Trạng thái nhịp số (0-9)")
        if analysis:
            chart_data = pd.DataFrame([{"Số": k, "Điểm": v['score'], "Trạng thái": v['state']} for k, v in analysis.items()])
            fig = px.bar(chart_data, x="Số", y="Điểm", color="Trạng thái", 
                         title="Biểu đồ sức mạnh số đơn (Ưu tiên > 70 điểm)",
                         color_discrete_map={"ỔN ĐỊNH": "#27AE60", "NÓNG": "#E67E22", "NGUY HIỂM": "#E74C3C", "YẾU": "#95A5A6"})
            st.plotly_chart(fig, use_container_width=True)

if __name__ == "__main__":
    main()
