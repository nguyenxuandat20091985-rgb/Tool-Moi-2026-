import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import google.generativeai as genai
from datetime import datetime
import os

# ================= CONFIG & API =================
st.set_page_config(page_title="AI LOTOBET 2-TINH v2", layout="wide")

# Cấu hình Gemini từ API của anh
GEMINI_API_KEY = "AIzaSyBgd0Au6FGhsiqTkADgz1SBECjs2e1MwGE"
genai.configure(api_key=GEMINI_API_KEY)

DATA_FILE = "lotobet_data_v2.csv"

# ================= AI LOGIC ENGINE =================
class LotoEngineV2:
    def __init__(self):
        self.model = genai.GenerativeModel('gemini-1.5-flash')

    def clean_data(self, raw_text):
        """Lọc dữ liệu sạch: Chỉ lấy dòng có đúng 5 chữ số"""
        lines = raw_text.split('\n')
        clean_lines = []
        for line in lines:
            nums = "".join(filter(str.isdigit, line.strip()))
            if len(nums) == 5:
                clean_lines.append(nums)
        return clean_lines

    def analyze_numbers(self, df):
        """Phân tích số đơn 0-9 theo ma trận"""
        if df.empty: return None
        
        # Chuyển series thành ma trận số nguyên an toàn
        matrix = []
        for s in df['numbers'].values:
            matrix.append([int(d) for d in str(s)])
        matrix = np.array(matrix)
        
        stats = {}
        for n in range(10):
            # Tìm các kỳ xuất hiện số n
            appears = np.where(np.any(matrix == n, axis=1))[0]
            gaps = np.diff(appears) if len(appears) > 1 else []
            
            # Tần suất gần đây
            recent_5 = sum(1 for row in matrix[-5:] if n in row)
            recent_10 = sum(1 for row in matrix[-10:] if n in row)
            
            # Gán nhãn trạng thái theo đặc tả
            last_idx = appears[-1] if len(appears) > 0 else -1
            dist = len(matrix) - 1 - last_idx
            
            state = "ỔN ĐỊNH"
            if recent_5 >= 4: state = "NÓNG"
            elif dist == 0: state = "NGUY HIỂM" # Vừa ra kỳ trước
            elif recent_10 <= 1: state = "YẾU"
            
            stats[n] = {
                "state": state,
                "dist": dist,
                "freq_10": recent_10,
                "last_gap": gaps[-1] if len(gaps) > 0 else 99
            }
        return stats

    def get_gemini_insight(self, history_str):
        """Hỏi ý kiến Gemini về bóng số và quy luật nâng cao"""
        prompt = f"""
        Dữ liệu Lotobet 5 số: {history_str}
        Dựa trên quy luật bóng số (0-5, 1-6, 2-7, 3-8, 4-9) và nhịp cầu lặp, hãy phân tích 2 số đơn tiềm năng nhất.
        Yêu cầu: Chỉ trả về 2 số đơn tiềm năng, cách nhau dấu phẩy. Ví dụ: 3,8. Không giải thích dài dòng.
        """
        try:
            response = self.model.generate_content(prompt)
            return response.text.strip()
        except:
            return None

    def get_predictions(self, df, stats):
        """Logic ghép cặp & lọc số chập"""
        if not stats: return [], "Dữ liệu yếu", []
        
        # 1. Lấy danh sách số đạt tiêu chuẩn (Loại NÓNG quá mức và YẾU quá mức)
        candidates = []
        for n, s in stats.items():
            if s['state'] in ["ỔN ĐỊNH", "YẾU"] and s['dist'] > 0:
                candidates.append(n)
        
        # 2. Ghép cặp (Không chập)
        potential_pairs = []
        for i in range(len(candidates)):
            for j in range(i + 1, len(candidates)):
                n1, n2 = candidates[i], candidates[j]
                
                # Logic trọng số: 1 ổn định + 1 hồi (dist từ 5-7)
                score = 50
                if (stats[n1]['dist'] >= 5 or stats[n2]['dist'] >= 5): score += 25
                if (stats[n1]['state'] == "ỔN ĐỊNH" and stats[n2]['state'] == "ỔN ĐỊNH"): score += 15
                
                potential_pairs.append({
                    "pair": (n1, n2),
                    "score": score
                })
        
        # 3. Kiểm tra điều kiện "KHÔNG ĐÁNH"
        if len(df) < 15:
            return [], "KHÔNG ĐÁNH (Thiếu dữ liệu)", ["Cần ít nhất 15 kỳ để soi cầu chuẩn."]
        
        hot_count = sum(1 for s in stats.values() if s['state'] == "NÓNG")
        if hot_count >= 7:
            return [], "KHÔNG ĐÁNH (Cầu nhiễu)", ["Thị trường đang quá NÓNG, nhà cái đang đảo cầu."]

        potential_pairs.sort(key=lambda x: x['score'], reverse=True)
        return potential_pairs[:1], "PREDICT", []

# ================= INTERFACE =================
def main():
    st.header("🎯 AI LOTOBET 2-TINH (BẢN CHUẨN v2)")
    engine = LotoEngineV2()

    # Quản lý file dữ liệu
    if not os.path.exists(DATA_FILE):
        pd.DataFrame(columns=["time", "numbers"]).to_csv(DATA_FILE, index=False)

    tab1, tab2 = st.tabs(["📊 Phân tích & Dự đoán", "📥 Nhập dữ liệu"])

    with tab2:
        st.subheader("📥 Cập nhật dữ liệu sạch")
        raw_data = st.text_area("Nhập 5 số viết liền (mỗi dòng 1 kỳ):", height=200)
        col_btn1, col_btn2 = st.columns(2)
        
        if col_btn1.button("💾 Lưu dữ liệu"):
            clean_lines = engine.clean_data(raw_data)
            if clean_lines:
                df_old = pd.read_csv(DATA_FILE)
                new_records = [{"time": datetime.now().strftime("%H:%M"), "numbers": n} for n in clean_lines]
                df_new = pd.concat([df_old, pd.DataFrame(new_records)], ignore_index=True)
                df_new.to_csv(DATA_FILE, index=False)
                st.success(f"Đã cập nhật {len(clean_lines)} kỳ thành công!")
                st.rerun()
            else:
                st.error("Dữ liệu không đúng định dạng 5 chữ số.")

        if col_btn2.button("🗑 Xóa hết dữ liệu"):
            pd.DataFrame(columns=["time", "numbers"]).to_csv(DATA_FILE, index=False)
            st.warning("Đã xóa sạch dữ liệu lịch sử.")

    with tab1:
        df = pd.read_csv(DATA_FILE)
        if df.empty:
            st.info("Vui lòng nhập dữ liệu lịch sử tại tab Nhập liệu.")
            return

        # Thống kê nhanh
        st.write(f"📈 Đang phân tích trên **{len(df)}** kỳ gần nhất.")
        
        # Thực hiện phân tích
        stats = engine.analyze_numbers(df)
        preds, status, reasons = engine.get_predictions(df, stats)

        # Khu vực Dự đoán chính
        st.subheader("🔮 Kết quả dự đoán")
        if status == "PREDICT" and preds:
            p = preds[0]
            confidence = p['score']
            pair_str = f"{p['pair'][0]}{p['pair'][1]}"
            
            # Gọi Gemini bổ trợ
            history_str = ",".join(df['numbers'].tail(10).astype(str).tolist())
            insight = engine.get_gemini_insight(history_str)
            
            col_res1, col_res2 = st.columns(2)
            with col_res1:
                st.markdown(f"""
                <div style="background:#1E1E1E; padding:30px; border-radius:15px; text-align:center; border: 2px solid #FF4B4B;">
                    <h1 style="color:#FF4B4B; font-size:60px; margin:0;">{pair_str}</h1>
                    <p style="color:white; margin:0;">Độ tự tin AI: {confidence}%</p>
                </div>
                """, unsafe_allow_html=True)
            
            with col_res2:
                st.info(f"**Gợi ý từ Gemini (Bóng số):** {insight if insight else 'Đang tính toán...'}")
                st.write("✅ Ưu tiên đánh cặp này không cố định vị trí.")
                st.write("⚠️ Quản lý vốn: Chỉ nên vào 1-2% tài khoản.")

        else:
            st.error(f"🚫 {status}")
            for r in reasons: st.write(f"• {r}")

        # Biểu đồ trạng thái số đơn
        st.divider()
        st.subheader("📈 Trạng thái dòng số hiện tại")
        chart_data = pd.DataFrame([
            {"Số": str(k), "Khoảng cách kỳ chưa ra": v['dist'], "Trạng thái": v['state']}
            for k, v in stats.items()
        ])
        fig = px.bar(chart_data, x="Số", y="Khoảng cách kỳ chưa ra", color="Trạng thái",
                     title="Biểu đồ độ trễ (Số càng cao kỳ chưa ra càng dài - Tiềm năng hồi cầu)")
        st.plotly_chart(fig, use_container_width=True)

        with st.expander("Xem chi tiết thông số"):
            st.table(pd.DataFrame(stats).T)

if __name__ == "__main__":
    main()
