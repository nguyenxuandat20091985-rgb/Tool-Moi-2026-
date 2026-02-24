import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
import os

# ================= CONFIG & UI =================
st.set_page_config(page_title="AI 2-TINH LOTOBET v2", layout="wide", page_icon="🎯")

st.markdown("""
    <style>
    .reportview-container { background: #f0f2f6; }
    .stMetric { background: white; padding: 15px; border-radius: 10px; border: 1px solid #ddd; }
    .status-box { padding: 20px; border-radius: 10px; text-align: center; margin: 10px 0; }
    .prediction-card { background: #ffffff; padding: 25px; border-radius: 15px; border: 2px solid #ff4b4b; text-align: center; }
    </style>
""", unsafe_allow_html=True)

DATA_FILE = "loto_data_v2.csv"

# ================= AI LOGIC ENGINE =================
class LotobetStandardAI:
    def __init__(self):
        self.min_draws = 10
        self.states = ["NÓNG", "ỔN ĐỊNH", "YẾU", "NGUY HIỂM"]

    def analyze_single_numbers(self, df):
        """Bước 3: Phân tích từng số đơn từ 0-9"""
        if len(df) < 5: return None
        
        # Chuyển dữ liệu thành mảng số đơn
        matrix = []
        for val in df['numbers'].values:
            matrix.append([int(d) for d in str(val) if d.isdigit()])
        matrix = np.array(matrix)
        
        analysis = {}
        for num in range(10):
            # Tìm vị trí xuất hiện (0 là kỳ cũ nhất, len-1 là kỳ mới nhất)
            pos = np.where(np.any(matrix == num, axis=1))[0]
            gaps = np.diff(pos) if len(pos) > 1 else []
            
            # 1. Kiểm tra Lặp (Kỳ gần nhất có ra không)
            is_last_present = (num in matrix[-1])
            
            # 2. Đếm tần suất
            freq_3 = sum(1 for row in matrix[-3:] if num in row)
            freq_5 = sum(1 for row in matrix[-5:] if num in row)
            freq_10 = sum(1 for row in matrix[-10:] if num in row)
            
            # 3. Gán nhãn trạng thái (Bước 6)
            state = "ỔN ĐỊNH"
            if freq_3 >= 2: state = "NGUY HIỂM" # Ra dồn
            elif freq_5 >= 3: state = "NÓNG"
            elif freq_10 <= 1: state = "YẾU"
            
            # 4. Nhận diện nhịp cầu (Bước 4)
            bridge = "BÌNH THƯỜNG"
            if len(gaps) >= 2:
                if gaps[-1] == 1 and gaps[-2] == 1: bridge = "BỆT"
                elif 2 <= gaps[-1] <= 3 and gaps[-1] == gaps[-2]: bridge = "NHẢY"
            
            analysis[num] = {
                "state": state,
                "bridge": bridge,
                "is_last": is_last_present,
                "score": self.calculate_score(state, bridge, is_last_present, freq_10)
            }
        return analysis

    def calculate_score(self, state, bridge, is_last, freq_10):
        """Tính điểm sức mạnh cho từng số đơn"""
        score = 50
        # Ưu tiên theo đặc tả
        if bridge == "NHẢY": score += 20
        if state == "ỔN ĐỊNH": score += 15
        if is_last: score -= 25 # Số vừa ra kỳ trước -> giảm trọng số (Bước 5)
        if state == "YẾU": score -= 10
        if state == "NÓNG": score -= 5
        return score

    def get_predictions(self, df):
        """Logic ghép cặp và lọc (Bước 7 & 8)"""
        analysis = self.analyze_single_numbers(df)
        if not analysis: return None, "DỮ LIỆU THIẾU", []

        # Kiểm tra điều kiện KHÔNG ĐÁNH (Bước 8)
        hot_count = sum(1 for v in analysis.values() if v['state'] == "NÓNG")
        recent_count = sum(1 for v in analysis.values() if v['is_last'])
        
        if hot_count >= 7: return None, "KHÔNG ĐÁNH", ["Toàn số quá NÓNG"]
        if recent_count >= 4: return None, "KHÔNG ĐÁNH", ["Quá nhiều số vừa ra kỳ trước"]
        if len(df) < self.min_draws: return None, "KHÔNG ĐÁNH", ["Dữ liệu quá ít"]

        # Lọc danh sách số đơn tiềm năng (Loại Nóng, Nguy hiểm, Yếu nếu cần)
        candidates = []
        for num, data in analysis.items():
            # Bước 6: Không ghép 2 số cùng trạng thái xấu
            candidates.append({"num": num, **data})
        
        # Sắp xếp theo điểm số
        candidates.sort(key=lambda x: x['score'], reverse=True)
        
        pairs = []
        # Logic ghép (Bước 1 & 7)
        for i in range(len(candidates)):
            for j in range(i + 1, len(candidates)):
                n1, n2 = candidates[i], candidates[j]
                
                # ❌ CẤM số chập (Bước 1) - n1['num'] và n2['num'] luôn khác nhau do vòng lặp
                # ❌ Không ghép 2 số đều nóng/nguy hiểm (Bước 6)
                bad_states = ["NÓNG", "NGUY HIỂM"]
                if n1['state'] in bad_states and n2['state'] in bad_states: continue
                if n1['state'] == "YẾU" and n2['state'] == "YẾU": continue

                conf = (n1['score'] + n2['score']) / 1.5 # Thang đo độ tự tin
                
                if conf >= 60: # Bước 9: Chỉ lấy trên 60%
                    pairs.append({
                        "pair": f"{n1['num']}{n2['num']}",
                        "conf": int(conf),
                        "detail": f"{n1['state']} + {n2['state']}"
                    })

        pairs.sort(key=lambda x: x['conf'], reverse=True)
        
        if not pairs or pairs[0]['conf'] < 60:
            return None, "KHÔNG ĐÁNH", ["Không có cặp nào đạt ngưỡng an toàn"]
            
        return pairs[:1], "ĐÁNH", [] # Chỉ trả về 1 cặp tốt nhất (Bước 7)

# ================= UTILS =================
def handle_data():
    if not os.path.exists(DATA_FILE):
        pd.DataFrame(columns=["numbers"]).to_csv(DATA_FILE, index=False)
    return pd.read_csv(DATA_FILE)

# ================= APP UI =================
def main():
    st.title("🎯 AI LOTOBET 2-TINH (CHUẨN v2)")
    ai = LotobetStandardAI()
    
    # Sidebar nhập liệu
    with st.sidebar:
        st.header("📥 Nhập dữ liệu")
        new_data = st.text_area("Nhập kết quả (5 số viết liền, mỗi dòng 1 kỳ):", height=200)
        if st.button("💾 Lưu & Phân tích"):
            lines = [n.strip() for n in new_data.split("\n") if len(n.strip()) == 5]
            if lines:
                old_df = handle_data()
                new_df = pd.DataFrame(lines, columns=["numbers"])
                pd.concat([old_df, new_df]).tail(50).to_csv(DATA_FILE, index=False)
                st.success(f"Đã thêm {len(lines)} kỳ!")
                st.rerun()
        
        if st.button("🗑 Xóa dữ liệu cũ"):
            os.remove(DATA_FILE)
            st.rerun()

    # Main Area
    df = handle_data()
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("📊 Phân tích & Dự đoán")
        if len(df) < 5:
            st.warning("Vui lòng nhập ít nhất 5 kỳ để bắt đầu phân tích.")
        else:
            preds, status, reasons = ai.get_predictions(df)
            
            if status == "KHÔNG ĐÁNH":
                st.markdown(f"""<div class="status-box" style="background:#ffebee; color:#c62828;">
                    <h2>🚫 KHÔNG ĐÁNH KỲ NÀY</h2>
                    <p>{', '.join(reasons)}</p>
                </div>""", unsafe_allow_html=True)
            else:
                for p in preds:
                    color = "#4caf50" if p['conf'] >= 75 else "#ff9800"
                    st.markdown(f"""<div class="prediction-card">
                        <p style="color:gray; margin:0;">CẶP SỐ ĐỀ XUẤT</p>
                        <h1 style="font-size:80px; margin:10px 0;">{p['pair']}</h1>
                        <div style="font-size:24px; font-weight:bold; color:{color};">ĐỘ TỰ TIN: {p['conf']}%</div>
                        <p style="color:gray;">Trạng thái: {p['detail']}</p>
                    </div>""", unsafe_allow_html=True)

    with col2:
        st.subheader("📋 Lịch sử gần đây")
        st.dataframe(df.tail(10), use_container_width=True)
        
        if len(df) >= 5:
            st.subheader("💡 Trạng thái số đơn")
            analysis = ai.analyze_single_numbers(df)
            stat_df = pd.DataFrame([
                {"Số": k, "Trạng thái": v['state'], "Cầu": v['bridge']} 
                for k, v in analysis.items()
            ])
            st.table(stat_df)

if __name__ == "__main__":
    main()
