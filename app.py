import streamlit as st
import pandas as pd
import numpy as np
import google.generativeai as genai
from datetime import datetime
import os

# ================= CONFIG & API =================
st.set_page_config(page_title="AI LOTOBET 2-TINH v2", layout="wide", page_icon="🎯")

# Kết nối Gemini (Dùng API anh cung cấp)
try:
    genai.configure(api_key="AIzaSyBgd0Au6FGhsiqTkADgz1SBECjs2e1MwGE")
    model = genai.GenerativeModel('gemini-pro')
except:
    st.error("Lỗi kết nối Gemini. Vui lòng kiểm tra lại API Key.")

DATA_FILE = "lotobet_history.csv"

# ================= LOGIC PHÂN TÍCH AI =================
class LotobetEngineV2:
    def __init__(self):
        self.min_draws = 10
        self.labels = {
            "HOT": "NÓNG (Ra dày)",
            "STABLE": "ỔN ĐỊNH (Nhịp đều)",
            "WEAK": "YẾU (Ít ra)",
            "RISKY": "NGUY HIỂM (Vừa ra/Bệt)"
        }

    def clean_input(self, text):
        """Lọc dữ liệu rác, chỉ lấy đúng 5 chữ số"""
        lines = text.split('\n')
        valid_data = []
        for line in lines:
            clean_line = "".join(filter(str.isdigit, line.strip()))
            if len(clean_line) == 5:
                valid_data.append(clean_line)
        return valid_data

    def analyze_numbers(self, df):
        """Phân tích số đơn (0-9)"""
        if len(df) < 5: return None
        
        # Chuyển dữ liệu sang ma trận số đơn
        try:
            raw_matrix = []
            for s in df['numbers'].values:
                raw_matrix.append([int(d) for d in str(s)])
            matrix = np.array(raw_matrix)
        except Exception:
            return None

        analysis = {}
        for num in range(10):
            # Tìm các kỳ có mặt số này
            appears = np.where(np.any(matrix == num, axis=1))[0]
            count_10 = len(appears)
            
            # Tính khoảng cách (Gap)
            gaps = np.diff(appears) if len(appears) > 1 else []
            last_idx = appears[-1] if len(appears) > 0 else -1
            dist_from_last = len(matrix) - 1 - last_idx

            # Gán trạng thái theo Đặc tả v2
            if dist_from_last == 0: state = "RISKY" # Vừa ra kỳ trước
            elif count_10 >= 6: state = "HOT"
            elif 1 < dist_from_last <= 4: state = "STABLE"
            else: state = "WEAK"

            analysis[num] = {
                "state": state,
                "count": count_10,
                "gap": dist_from_last,
                "avg_gap": np.mean(gaps) if len(gaps) > 0 else 10
            }
        return analysis

    def get_final_prediction(self, analysis, df):
        """Logic ghép cặp & Không đánh"""
        reasons = []
        
        # 1. Kiểm tra điều kiện "Không đánh"
        hot_nums = [n for n, v in analysis.items() if v['state'] == "HOT"]
        risky_nums = [n for n, v in analysis.items() if v['state'] == "RISKY"]
        
        if len(risky_nums) >= 4:
            return None, "KHÔNG ĐÁNH KỲ NÀY", "Nhiều số vừa ra (Cầu lặp nhiễu)"
        if len(hot_nums) >= 6:
            return None, "KHÔNG ĐÁNH KỲ NÀY", "Thị trường quá NÓNG (Dễ gãy cầu)"
        if len(df) < self.min_draws:
            return None, "DỮ LIỆU THIẾU", f"Cần tối thiểu {self.min_draws} kỳ"

        # 2. Logic ưu tiên ghép (Ổn định + Hồi)
        stable = [n for n, v in analysis.items() if v['state'] == "STABLE"]
        weak = [n for n, v in analysis.items() if v['state'] == "WEAK" and 5 <= v['gap'] <= 8]
        
        candidates = []
        if stable and weak:
            # Ưu tiên 1 ổn định + 1 hồi
            pair = tuple(sorted([stable[0], weak[0]]))
            score = 85
            reasons = ["Ghép Ổn định + Cầu hồi (Đúng nhịp)"]
        elif len(stable) >= 2:
            pair = tuple(sorted([stable[0], stable[1]]))
            score = 75
            reasons = ["Ghép 2 số Ổn định"]
        else:
            return None, "KHÔNG ĐÁNH KỲ NÀY", "Không tìm thấy nhịp cầu an toàn"

        # 3. Loại bỏ số chập (Đã đảm bảo do chọn 2 số khác nhau từ list)
        if pair[0] == pair[1]:
            return None, "LỖI HỆ THỐNG", "Số chập bị loại"

        return {"pair": pair, "score": score, "reasons": reasons}, "PREDICT", ""

# ================= INTERFACE =================
def main():
    engine = LotobetEngineV2()
    
    st.title("🎯 AI LOTOBET 2-TINH (BẢN CHUẨN v2)")
    st.caption("Hệ thống phân tích số đơn - Tuyệt đối không đánh số chập")

    # Sidebar: Nhập liệu
    with st.sidebar:
        st.header("📥 Nhập dữ liệu")
        raw_data = st.text_area("Dán kết quả (5 số viết liền, mỗi dòng 1 kỳ):", height=250)
        if st.button("💾 Cập nhật & Làm sạch"):
            valid_list = engine.clean_input(raw_data)
            if valid_list:
                new_df = pd.DataFrame(valid_list, columns=["numbers"])
                new_df.to_csv(DATA_FILE, index=False)
                st.success(f"Đã lưu {len(valid_list)} kỳ sạch.")
                st.rerun()
            else:
                st.error("Dữ liệu không hợp lệ!")

    # Load dữ liệu
    if not os.path.exists(DATA_FILE):
        st.info("Vui lòng nhập dữ liệu ở thanh bên trái để bắt đầu.")
        return

    df = pd.read_csv(DATA_FILE)
    
    # Dashboard chính
    tab1, tab2 = st.tabs(["📊 Phân tích & Dự đoán", "📚 Lịch sử"])

    with tab1:
        if len(df) > 0:
            analysis = engine.analyze_numbers(df)
            if analysis:
                st.subheader("📡 Trạng thái số đơn (0-9)")
                cols = st.columns(5)
                for n in range(10):
                    v = analysis[n]
                    with cols[n % 5]:
                        color = "red" if v['state'] == "HOT" else "green" if v['state'] == "STABLE" else "gray"
                        st.markdown(f"**Số {n}**: :{color}[{v['state']}]")
                        st.caption(f"Lần cuối: {v['gap']} kỳ")

                st.divider()
                
                # Thực hiện dự đoán
                res, status, msg = engine.get_final_prediction(analysis, df)
                
                if status == "PREDICT":
                    c1, c2 = st.columns([1, 2])
                    with c1:
                        st.markdown(f"""
                        <div style="background:#1E1E1E; padding:20px; border-radius:15px; border:2px solid #00FF00; text-align:center;">
                            <h3 style="color:white; margin:0;">CẶP DUY NHẤT</h3>
                            <h1 style="color:#00FF00; font-size:60px; margin:10px 0;">{res['pair'][0]}{res['pair'][1]}</h1>
                            <b style="color:#FFD700;">Độ tự tin: {res['score']}%</b>
                        </div>
                        """, unsafe_allow_html=True)
                    with c2:
                        st.success(f"**Chiến thuật**: {res['reasons'][0]}")
                        st.info("**Bạch thủ**: Số " + str(res['pair'][0]))
                        
                        # Kết nối Gemini Phân tích cầu sâu
                        if st.button("🤖 Hỏi Gemini về cầu này"):
                            with st.spinner("Đang hỏi ý kiến chuyên gia AI..."):
                                prompt = f"Dữ liệu lotobet 5 kỳ gần: {df['numbers'].tail(5).tolist()}. AI đề xuất cặp {res['pair']}. Hãy phân tích ngắn gọn nhịp cầu này có an toàn không?"
                                try:
                                    response = model.generate_content(prompt)
                                    st.write(f"**Gemini:** {response.text}")
                                except:
                                    st.write("Gemini đang bận, anh hãy thử lại sau.")
                else:
                    st.error(f"🚫 {status}")
                    st.warning(f"Lý do: {msg}")
        else:
            st.warning("Dữ liệu trống.")

    with tab2:
        st.dataframe(df.tail(20), use_container_width=True)
        if st.button("🗑 Xóa hết dữ liệu"):
            os.remove(DATA_FILE)
            st.rerun()

if __name__ == "__main__":
    main()
