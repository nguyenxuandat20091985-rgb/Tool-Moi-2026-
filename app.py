import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from datetime import datetime
import os

# ================= CONFIG & STYLE =================
st.set_page_config(page_title="AI LOTOBET 2-TINH v2", layout="wide", page_icon="🎯")

st.markdown("""
    <style>
    .reportview-container { background: #f0f2f6; }
    .stMetric { background-color: #ffffff; padding: 15px; border-radius: 10px; border: 1px solid #d1d5db; }
    .prediction-box { padding: 25px; border-radius: 15px; background: #ffffff; border: 2px solid #3b82f6; text-align: center; }
    .skip-box { padding: 25px; border-radius: 15px; background: #fff5f5; border: 2px solid #e53e3e; text-align: center; color: #c53030; }
    </style>
    """, unsafe_allow_html=True)

DATA_FILE = "lotobet_v2_data.csv"

# ================= CORE AI ENGINE (ĐẶC TẢ v2) =================
class LotobetLogicV2:
    def __init__(self):
        self.states = {
            'HOT': "🔥 NÓNG",
            'STABLE': "✅ ỔN ĐỊNH",
            'WEAK': "❄️ YẾU",
            'DANGER': "⚠️ NGUY HIỂM"
        }

    def analyze_single_numbers(self, df):
        """Bước 3 & 6: Phân tích số đơn (0-9) và gán nhãn trạng thái"""
        if len(df) < 5: return None
        
        # Chuyển dữ liệu thành Matrix
        matrix = []
        for val in df['numbers'].values:
            matrix.append([int(d) for d in str(val)])
        matrix = np.array(matrix)
        
        analysis = {}
        for num in range(10):
            # Tìm các kỳ xuất hiện
            appears = np.where(np.any(matrix == num, axis=1))[0]
            last_3 = sum(1 for row in matrix[-3:] if num in row)
            last_5 = sum(1 for row in matrix[-5:] if num in row)
            last_10 = sum(1 for row in matrix[-10:] if num in row)
            
            # Tính khoảng cách kỳ gần nhất (Kỳ hiện tại - kỳ cuối xuất hiện)
            last_seen_ago = (len(df) - 1 - appears[-1]) if len(appears) > 0 else 99
            
            # Nhận diện nhịp (Gaps)
            gaps = np.diff(appears) if len(appears) > 1 else []
            
            # GÁN NHÃN TRẠNG THÁI (Bước 6)
            state = 'STABLE'
            if last_3 >= 2 or last_seen_ago == 0: state = 'DANGER' # Vừa ra hoặc ra dồn
            elif last_5 >= 3: state = 'HOT' # Ra dày
            elif last_10 <= 1: state = 'WEAK' # Ít xuất hiện
            else: state = 'STABLE' # Ra đều có nhịp
            
            analysis[num] = {
                'state': state,
                'last_3': last_3,
                'last_seen_ago': last_seen_ago,
                'gaps': gaps
            }
        return analysis

    def predict_pair(self, df, analysis):
        """Bước 7 & 8: Logic ghép cặp và Lọc 'Không Đánh'"""
        if not analysis: return None, "DỮ LIỆU THẤP", []

        # Kiểm tra điều kiện KHÔNG ĐÁNH (Bước 8)
        danger_count = sum(1 for v in analysis.values() if v['state'] == 'DANGER')
        hot_count = sum(1 for v in analysis.values() if v['state'] == 'HOT')
        
        reasons_to_skip = []
        if danger_count >= 5: reasons_to_skip.append("Thị trường quá biến động (Nhiều số NGUY HIỂM)")
        if hot_count >= 6: reasons_to_skip.append("Toàn số quá NÓNG (Dễ gãy cầu)")
        if len(df) < 10: reasons_to_skip.append("Dữ liệu quá ít (Cần >10 kỳ)")
        
        if reasons_to_skip:
            return None, "SKIP", reasons_to_skip

        # LOGIC GHÉP CẶP (Bước 7)
        # 1. Lấy danh sách số Ổn định và Yếu (sau khi đã hồi)
        stable_nums = [n for n, v in analysis.items() if v['state'] == 'STABLE']
        weak_nums = [n for n, v in analysis.items() if v['state'] == 'WEAK' and 5 <= v['last_seen_ago'] <= 8]
        
        candidates = []
        
        # Ưu tiên: 1 Ổn định + 1 Yếu (đang hồi)
        for s in stable_nums:
            for w in weak_nums:
                candidates.append({
                    'pair': tuple(sorted((s, w))),
                    'score': 85,
                    'type': 'Ổn định + Hồi cầu'
                })
        
        # Ưu tiên: 2 số Ổn định khác nhau
        if len(stable_nums) >= 2:
            from itertools import combinations
            for p in combinations(stable_nums, 2):
                candidates.append({
                    'pair': p,
                    'score': 72,
                    'type': 'Cặp song hành Ổn định'
                })

        if not candidates:
            return None, "SKIP", ["Không tìm thấy cặp số đạt ngưỡng an toàn"]

        # Lọc kết quả cuối: Tối đa 1-2 cặp (Bước 7)
        candidates.sort(key=lambda x: x['score'], reverse=True)
        
        # Loại bỏ trùng lặp cặp số
        unique_candidates = []
        seen = set()
        for c in candidates:
            if c['pair'] not in seen:
                unique_candidates.append(c)
                seen.add(c['pair'])
        
        return unique_candidates[:2], "PREDICT", []

# ================= DATA INTERFACE =================
def load_data():
    if os.path.exists(DATA_FILE):
        return pd.read_csv(DATA_FILE)
    return pd.DataFrame(columns=["time", "numbers"])

def save_data(val):
    df = load_data()
    now = datetime.now().strftime("%H:%M:%S")
    # Chỉ lưu nếu đúng 5 chữ số
    new_rows = []
    for line in val.split("\n"):
        clean = line.strip()
        if len(clean) == 5 and clean.isdigit():
            new_rows.append({"time": now, "numbers": clean})
    
    if new_rows:
        df = pd.concat([df, pd.DataFrame(new_rows)], ignore_index=True)
        df.to_csv(DATA_FILE, index=False)
        return len(new_rows)
    return 0

# ================= APP UI =================
def main():
    st.header("📘 AI LOTOBET 2-TINH - BẢN CHUẨN v2")
    ai = LotobetLogicV2()
    df = load_data()

    col_main, col_side = st.columns([7, 3])

    with col_side:
        st.subheader("📥 Nhập dữ liệu")
        raw_input = st.text_area("Dán kết quả (5 số/dòng):", height=200)
        if st.button("💾 Cập nhật hệ thống"):
            added = save_data(raw_input)
            if added > 0:
                st.success(f"Đã thêm {added} kỳ!")
                st.rerun()
            else:
                st.error("Dữ liệu không hợp lệ!")
        
        if st.button("🗑 Xóa dữ liệu cũ"):
            if os.path.exists(DATA_FILE):
                os.remove(DATA_FILE)
                st.rerun()

    with col_main:
        if len(df) < 5:
            st.warning("⚠️ Hệ thống cần tối thiểu 5 kỳ để bắt đầu phân tích số đơn.")
            return

        # Phân tích
        analysis = ai.analyze_single_numbers(df)
        
        # Dashboard Thống kê
        st.subheader("📊 Trạng thái số đơn (0-9)")
        cols = st.columns(5)
        for i in range(10):
            with cols[i % 5]:
                st.metric(f"Số {i}", f"{analysis[i]['last_3']} lần", analysis[i]['state'])

        st.divider()

        # Dự đoán
        st.subheader("🔮 Kết quả dự đoán AI")
        preds, status, reasons = ai.predict_pair(df, analysis)

        if status == "SKIP":
            st.markdown(f"""
            <div class="skip-box">
                <h2>🚫 KHÔNG ĐÁNH KỲ NÀY</h2>
                <p>Lý do: {', '.join(reasons)}</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            c1, c2 = st.columns(2)
            for idx, p in enumerate(preds):
                with (c1 if idx == 0 else c2):
                    st.markdown(f"""
                    <div class="prediction-box">
                        <p style="color:gray; margin:0;">Cặp đề xuất {idx+1}</p>
                        <h1 style="font-size: 50px; color: #1e40af; margin: 10px 0;">{p['pair'][0]}{p['pair'][1]}</h1>
                        <p style="background: #e0f2fe; display: inline-block; padding: 5px 15px; border-radius: 20px;">
                            Độ tự tin: <b>{p['score']}%</b>
                        </p>
                        <p><small>{p['type']}</small></p>
                    </div>
                    """, unsafe_allow_html=True)

        st.divider()
        # Trực quan hóa Gaps
        st.subheader("📉 Biểu đồ tần suất (10 kỳ)")
        chart_data = pd.DataFrame([{'Số': i, 'Lần xuất hiện': analysis[i]['last_10']} for i in range(10)])
        fig = px.bar(chart_data, x='Số', y='Lần xuất hiện', color='Lần xuất hiện', color_continuous_scale='Blues')
        st.plotly_chart(fig, use_container_width=True)

if __name__ == "__main__":
    main()
