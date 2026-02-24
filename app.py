import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import google.generativeai as genai
from datetime import datetime
import os

# ================= CONFIG & API =================
# API KEY của anh: AIzaSyBgd0Au6FGhsiqTkADgz1SBECjs2e1MwGE
genai.configure(api_key="AIzaSyBgd0Au6FGhsiqTkADgz1SBECjs2e1MwGE")
model_gemini = genai.GenerativeModel('gemini-1.5-flash')

st.set_page_config(page_title="AI LOTOBET V2 - CHUẨN ĐẶC TẢ", layout="wide")

DATA_FILE = "lotobet_v2_data.csv"

# ================= CORE AI ENGINE =================
class LotobetStandardV2:
    def __init__(self):
        self.states = ["NÓNG", "ỔN ĐỊNH", "YẾU", "NGUY HIỂM"]

    def clean_matrix(self, df):
        """Xử lý triệt để lỗi ValueError khi tạo matrix"""
        clean_data = []
        for val in df['numbers'].values:
            val_str = str(val).strip()
            if len(val_str) == 5 and val_str.isdigit():
                clean_data.append([int(d) for d in val_str])
        return np.array(clean_data)

    def analyze_numbers(self, df):
        matrix = self.clean_matrix(df)
        if len(matrix) < 5: return None
        
        results = {}
        for num in range(10):
            # 1. Tìm vị trí xuất hiện
            appears = np.where(np.any(matrix == num, axis=1))[0]
            gaps = np.diff(appears) if len(appears) > 1 else []
            
            # 2. Thống kê theo trọng số thời gian
            recent_3 = 1 if any(num in row for row in matrix[-3:]) else 0
            recent_5 = sum(1 for row in matrix[-5:] if num in row)
            
            # 3. Gán trạng thái theo đặc tả v2
            last_idx = appears[-1] if len(appears) > 0 else -1
            dist_from_now = len(matrix) - 1 - last_idx
            
            if recent_5 >= 3: state = "NÓNG"
            elif dist_from_now == 0 or (len(gaps) > 0 and gaps[-1] == 1): state = "NGUY HIỂM"
            elif 3 <= dist_from_now <= 7: state = "ỔN ĐỊNH" # Cầu hồi/nhảy tốt
            else: state = "YẾU"

            results[num] = {
                "state": state,
                "freq_5": recent_5,
                "gap": dist_from_now,
                "avg_gap": np.mean(gaps) if len(gaps) > 0 else 10
            }
        return results

    def get_gemini_advice(self, history_str, suggestion):
        """Kết nối Gemini để thẩm định cầu lừa"""
        try:
            prompt = f"""
            Dữ liệu Lotobet 5 số: {history_str}
            AI đang định đánh cặp: {suggestion}
            Dựa trên đặc tả: Không đánh số chập, tránh số vừa ra kỳ trước, tránh cầu bệt quá dài.
            Hãy trả lời ngắn gọn: 'CHỐT' hoặc 'KHÔNG ĐÁNH' và lý do trong 10 từ.
            """
            response = model_gemini.generate_content(prompt)
            return response.text
        except:
            return "Gemini đang bận, dùng logic mặc định."

    def final_decision(self, analysis, df):
        """Logic ghép cặp & Quyết định KHÔNG ĐÁNH"""
        reasons_to_skip = []
        
        # Kiểm tra điều kiện "KHÔNG ĐÁNH"
        hot_count = sum(1 for v in analysis.values() if v['state'] == "NÓNG")
        recent_matches = set([int(d) for d in str(df['numbers'].iloc[-1])])
        
        if hot_count >= 6: reasons_to_skip.append("Thị trường quá NÓNG (nhiều số ra dồn)")
        if len(df) < 10: reasons_to_skip.append("Dữ liệu quá ít (cần tối thiểu 10 kỳ)")
        
        # Lọc số đơn để ghép
        # Ưu tiên: Ổn định + Ổn định hoặc Ổn định + Yếu (đang hồi)
        candidates = [n for n, v in analysis.items() if v['state'] in ["ỔN ĐỊNH", "YẾU"]]
        
        best_pair = None
        confidence = 0
        
        if len(candidates) >= 2:
            # Chọn 2 số có nhịp đẹp nhất (không phải số vừa ra kỳ trước)
            potential = [c for c in candidates if c not in recent_matches]
            if len(potential) >= 2:
                best_pair = tuple(sorted(potential[:2]))
                confidence = 78 if hot_count < 4 else 62
        
        if confidence < 60 or not best_pair:
            return None, "KHÔNG ĐÁNH KỲ NÀY", reasons_to_skip or ["Không có cặp đạt ngưỡng an toàn"]
        
        return best_pair, "PREDICT", [f"Cầu đang nhịp {analysis[best_pair[0]]['state']}"]

# ================= UI & APP =================
def load_data():
    if os.path.exists(DATA_FILE):
        return pd.read_csv(DATA_FILE).drop_duplicates().tail(50)
    return pd.DataFrame(columns=["time", "numbers"])

def main():
    st.markdown(f"<h1 style='text-align: center; color: #E74C3C;'>🎯 AI LOTOBET 2-TINH (BẢN CHUẨN v2)</h1>", unsafe_allow_html=True)
    
    engine = LotobetStandardV2()
    df = load_data()

    tab1, tab2 = st.tabs(["📊 Phân tích & Dự đoán", "📥 Nhập dữ liệu"])

    with tab2:
        st.subheader("📥 Cập nhật dữ liệu sạch")
        raw = st.text_area("Nhập 5 số viết liền (mỗi dòng 1 kỳ):", height=200, help="Ví dụ: 12345")
        if st.button("💾 Lưu dữ liệu"):
            if raw:
                lines = [l.strip() for l in raw.split("\n") if len(l.strip()) == 5 and l.strip().isdigit()]
                if lines:
                    new_df = pd.DataFrame({"time": [datetime.now().strftime("%H:%M:%S")]*len(lines), "numbers": lines})
                    df_final = pd.concat([df, new_df], ignore_index=True)
                    df_final.to_csv(DATA_FILE, index=False)
                    st.success(f"Đã lưu {len(lines)} kỳ hợp lệ!")
                    st.rerun()
                else: st.error("Dữ liệu không đúng định dạng 5 chữ số!")

    with tab1:
        if len(df) < 5:
            st.warning("Vui lòng nhập thêm dữ liệu (Cần ít nhất 5-10 kỳ).")
            return

        # 1. Phân tích số đơn
        analysis = engine.analyze_numbers(df)
        if not analysis:
            st.error("Lỗi xử lý ma trận dữ liệu.")
            return

        # 2. Hiển thị Dashboard trạng thái
        st.subheader("📋 Trạng thái 10 số đơn")
        cols = st.columns(5)
        for i in range(10):
            with cols[i % 5]:
                color = "red" if analysis[i]['state'] == "NÓNG" else "green" if analysis[i]['state'] == "ỔN ĐỊNH" else "gray"
                st.markdown(f"**Số {i}**: <span style='color:{color}'>{analysis[i]['state']}</span>", unsafe_allow_html=True)

        st.divider()

        # 3. Dự đoán theo Đặc tả v2
        pair, status, reasons = engine.final_decision(analysis, df)
        
        if status == "KHÔNG ĐÁNH KỲ NÀY":
            st.error("🚫 **KHÔNG ĐÁNH KỲ NÀY**")
            for r in reasons: st.write(f"- {r}")
        else:
            # Gọi Gemini thẩm định
            history_str = ", ".join(df['numbers'].tail(5).tolist())
            gemini_check = engine.get_gemini_advice(history_str, f"{pair[0]}{pair[1]}")
            
            st.success(f"✅ **CẶP SỐ ĐỀ XUẤT: {pair[0]}{pair[1]}**")
            st.info(f"🤖 **Gemini thẩm định:** {gemini_check}")
            
            st.divider()
            st.subheader("📈 Biểu đồ tần suất")
            fig_data = pd.DataFrame([{"Số": k, "Lần ra (5 kỳ)": v['freq_5'], "Trạng thái": v['state']} for k, v in analysis.items()])
            fig = px.bar(fig_data, x="Số", y="Lần ra (5 kỳ)", color="Trạng thái", barmode="group")
            st.plotly_chart(fig, use_container_width=True)



if __name__ == "__main__":
    main()
