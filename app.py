import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from datetime import datetime
import os

# ================= CONFIG & CONSTANTS =================
st.set_page_config(page_title="AI 2 TINH LOTOBET v2", layout="wide", page_icon="🎯")

# Hằng số trạng thái theo đặc tả
STATE_HOT = "NÓNG"
STATE_STABLE = "ỔN ĐỊNH"
STATE_WEAK = "YẾU"
STATE_DANGER = "NGUY HIỂM"
STATE_NORMAL = "BÌNH THƯỜNG"

DATA_FILE = "lotobet_history_v2.csv"

# ================= CORE AI LOGIC =================
class LotobetAI_V2:
    def __init__(self, history_df):
        self.df = history_df
        self.matrix = self._prepare_matrix()
        
    def _prepare_matrix(self):
        """Chuyển dữ liệu text thành ma trận số đơn"""
        matrix = []
        for val in self.df['numbers'].values:
            nums = [int(d) for d in str(val).strip() if d.isdigit()]
            if len(nums) == 5:
                matrix.append(nums)
        return np.array(matrix)

    def analyze_single_numbers(self):
        """3️⃣ PHÂN TÍCH SỐ ĐƠN (0-9) - TRỤ CỘT CỦA ĐẶC TẢ"""
        if len(self.matrix) < 5: return None
        
        analysis = {}
        total_draws = len(self.matrix)
        
        for num in range(10):
            # Tìm các kỳ xuất hiện
            appears = np.where(np.any(self.matrix == num, axis=1))[0]
            last_idx = appears[-1] if len(appears) > 0 else -1
            
            # 5️⃣ TRỌNG SỐ THỜI GIAN
            gap_from_last = total_draws - 1 - last_idx if last_idx != -1 else 99
            freq_3 = sum(1 for row in self.matrix[-3:] if num in row)
            freq_5 = sum(1 for row in self.matrix[-5:] if num in row)
            freq_10 = sum(1 for row in self.matrix[-10:] if num in row)
            
            # 6️⃣ PHÂN LOẠI TRẠNG THÁI SỐ
            state = STATE_NORMAL
            if freq_3 >= 2: state = STATE_DANGER  # Vừa ra hoặc ra dồn
            elif freq_5 >= 3: state = STATE_HOT    # Ra dày, sát nhau
            elif 2 <= freq_10 <= 4 and gap_from_last > 1: state = STATE_STABLE # Ra đều, có nhịp
            elif freq_10 <= 1: state = STATE_WEAK # Ít xuất hiện
            
            # 4️⃣ NHẬN DIỆN CẦU
            bridge = "BÌNH THƯỜNG"
            gaps = np.diff(appears) if len(appears) > 1 else []
            if len(gaps) >= 2 and gaps[-1] == gaps[-2] and gaps[-1] > 1:
                bridge = "CẦU NHẢY" # Nhịp đều
            elif gap_from_last == 0:
                bridge = "CẦU LẶP"
            elif gap_from_last >= 5 and gap_from_last <= 8:
                bridge = "CẦU HỒI"

            analysis[num] = {
                "state": state,
                "bridge": bridge,
                "gap": gap_from_last,
                "freq_10": freq_10,
                "last_in_prev": (gap_from_last == 0)
            }
        return analysis

    def get_prediction(self):
        """7️⃣ & 8️⃣ LOGIC GHÉP CẶP & KHÔNG ĐÁNH"""
        analysis = self.analyze_single_numbers()
        if not analysis: return None, "DỮ LIỆU CHƯA ĐỦ", []

        # 8️⃣ LOGIC "KHÔNG ĐÁNH"
        reasons_to_skip = []
        hot_count = sum(1 for v in analysis.values() if v['state'] in [STATE_HOT, STATE_DANGER])
        if hot_count >= 7: reasons_to_skip.append("Toàn số quá nóng")
        
        repeat_count = sum(1 for v in analysis.values() if v['last_in_prev'])
        if repeat_count >= 3: reasons_to_skip.append("Nhiều số vừa ra kỳ trước (Cầu lặp nhiễu)")

        if reasons_to_skip:
            return None, "KHÔNG ĐÁNH KỲ NÀY", reasons_to_skip

        # 7️⃣ LOGIC GHÉP 2 TINH
        scored_pairs = []
        # 1️⃣ ĐỊNH NGHĨA 2 TINH: Ghép i và j (i luôn khác j -> Không chập)
        for i in range(10):
            for j in range(i + 1, 10):
                s1, s2 = analysis[i], analysis[j]
                
                # BẮT BUỘC LOẠI TRỪ THEO MỤC 6
                # Không ghép 2 số đều nóng, 2 số nguy hiểm, 2 số yếu
                forbidden_states = [STATE_HOT, STATE_DANGER, STATE_WEAK]
                if s1['state'] in forbidden_states and s2['state'] in forbidden_states:
                    continue
                
                # Tính điểm tự tin (%)
                score = 50
                # Ưu tiên 1: Ổn định + Hồi
                if (s1['state'] == STATE_STABLE and s2['bridge'] == "CẦU HỒI") or \
                   (s2['state'] == STATE_STABLE and s1['bridge'] == "CẦU HỒI"):
                    score += 35
                # Ưu tiên 2: Nhảy nhịp + Ổn định
                if (s1['bridge'] == "CẦU NHẢY" and s2['state'] == STATE_STABLE) or \
                   (s2['bridge'] == "CẦU NHẢY" and s1['state'] == STATE_STABLE):
                    score += 25
                
                # Trừ điểm nếu có số vừa ra (Mục 5)
                if s1['last_in_prev'] or s2['last_in_prev']:
                    score -= 20

                if score >= 60:
                    scored_pairs.append({
                        "pair": f"{i}{j}",
                        "score": min(95, score),
                        "desc": f"{s1['state']} + {s2['state']}"
                    })

        scored_pairs.sort(key=lambda x: x['score'], reverse=True)
        
        # 7️⃣ KẾT QUẢ CUỐI: Tối đa 1-2 cặp
        if not scored_pairs:
            return None, "KHÔNG ĐÁNH KỲ NÀY", ["Không có cặp đạt ngưỡng an toàn"]
        
        return scored_pairs[:2], "PREDICT", []

# ================= STREAMLIT UI =================
def main():
    st.markdown("<h1 style='text-align: center; color: #FF4B4B;'>🎯 AI 2 TINH LOTOBET v2</h1>", unsafe_allow_html=True)
    
    # Load data
    if not os.path.exists(DATA_FILE):
        pd.DataFrame(columns=["time", "numbers"]).to_csv(DATA_FILE, index=False)
    df = pd.read_csv(DATA_FILE)

    menu = ["📊 Dự đoán & Thống kê", "📥 Nhập liệu", "⚙️ Quản lý"]
    choice = st.sidebar.selectbox("MENU", menu)

    if choice == "📥 Nhập liệu":
        st.subheader("📥 Nhập kết quả kỳ mới")
        raw = st.text_area("Nhập 5 số (ví dụ: 12345), mỗi kỳ một dòng", height=200)
        if st.button("Lưu dữ liệu"):
            lines = [l.strip() for l in raw.split("\n") if len(l.strip()) == 5]
            if lines:
                new_data = pd.DataFrame([{"time": datetime.now().strftime("%H:%M:%S"), "numbers": l} for l in lines])
                df = pd.concat([df, new_data], ignore_index=True)
                df.to_csv(DATA_FILE, index=False)
                st.success(f"Đã thêm {len(lines)} kỳ!")
                st.rerun()

    elif choice == "📊 Dự đoán & Thống kê":
        if len(df) < 10:
            st.warning("⚠️ Cần tối thiểu 10 kỳ để phân tích chính xác.")
            return

        ai = LotobetAI_V2(df)
        analysis = ai.analyze_single_numbers()
        preds, status, reasons = ai.get_prediction()

        # Hiển thị khu vực Dự đoán
        st.markdown("### 🔮 Dự đoán kỳ tiếp theo")
        if status == "KHÔNG ĐÁNH KỲ NÀY":
            st.error("🚫 KHÔNG ĐÁNH KỲ NÀY")
            for r in reasons: st.write(f"• {r}")
        else:
            cols = st.columns(len(preds))
            for i, p in enumerate(preds):
                with cols[i]:
                    color = "#2ECC71" if p['score'] >= 75 else "#F1C40F"
                    st.markdown(f"""
                        <div style="background: white; padding: 20px; border-radius: 15px; border: 2px solid {color}; text-align: center;">
                            <h1 style="margin:0; font-size: 50px; color: #2C3E50;">{p['pair']}</h1>
                            <b style="color: {color}; font-size: 20px;">Độ tự tin: {p['score']}%</b>
                            <p style="color: gray; font-size: 14px;">Trạng thái: {p['desc']}</p>
                        </div>
                    """, unsafe_allow_html=True)

        # Hiển thị Biểu đồ Thống kê
        st.divider()
        st.subheader("📊 Trạng thái 10 số đơn (0-9)")
        if analysis:
            chart_df = pd.DataFrame([
                {"Số": k, "Khoảng cách": v['gap'], "Trạng thái": v['state'], "Cầu": v['bridge']}
                for k, v in analysis.items()
            ])
            fig = px.bar(chart_df, x="Số", y="Khoảng cách", color="Trạng thái",
                         hover_data=["Cầu"], text_auto=True,
                         title="Khoảng cách kỳ chưa ra (Càng cao càng lâu chưa về)")
            st.plotly_chart(fig, use_container_width=True)

            # Bảng chi tiết
            st.table(chart_df)

    elif choice == "⚙️ Quản lý":
        st.subheader("Dữ liệu hiện tại")
        st.write(df.tail(20))
        if st.button("Xóa toàn bộ dữ liệu"):
            os.remove(DATA_FILE)
            st.rerun()

if __name__ == "__main__":
    main()
