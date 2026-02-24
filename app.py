import streamlit as st
import google.generativeai as genai
import re
import json
import os
from collections import Counter
import itertools
import numpy as np
from typing import List, Tuple, Dict

# ================= CẤU HÌNH HỆ THỐNG =================
API_KEY = "AIzaSyChq-KF-DXqPQUpxDsVIvx5D4_jRH1ERqM"
DB_FILE = "titan_memory_v21.json" 

def setup_neural():
    try:
        genai.configure(api_key=API_KEY)
        return genai.GenerativeModel('gemini-1.5-flash')
    except: 
        return None 

neural_engine = setup_neural() 

# ================= HỆ THỐNG GHI NHỚ VĨNH VIỄN =================
def load_memory():
    if os.path.exists(DB_FILE):
        with open(DB_FILE, "r") as f: 
            try: 
                data = json.load(f)
                # Chuẩn hóa dữ liệu: đảm bảo mỗi kỳ là string 5 số
                normalized = []
                for item in data[-1000:]:
                    if isinstance(item, str) and len(item) == 5 and item.isdigit():
                        normalized.append(item)
                    elif isinstance(item, list) and len(item) == 5:
                        normalized.append(''.join(map(str, item)))
                return normalized
            except: 
                return []
    return [] 

def save_memory(data):
    # Giữ lại 1000 kỳ gần nhất để AI có dữ liệu sâu
    with open(DB_FILE, "w") as f: 
        json.dump(data[-1000:], f) 

if "history" not in st.session_state:
    st.session_state.history = load_memory()
if "prediction_history" not in st.session_state:
    st.session_state.prediction_history = []

# ================= THUẬT TOÁN NÂNG CAO =================

class TitanPredictor:
    """Thuật toán dự đoán chuyên sâu cho game 5D - 3 số 5 tính"""
    
    def __init__(self, history: List[str]):
        self.history = [h if isinstance(h, str) else ''.join(map(str, h)) for h in history if h]
        self.positions = ['chuc_ngan', 'ngan', 'tram', 'chuc', 'don_vi']
        
    def analyze_position_frequency(self, window: int = 30) -> Dict:
        """Phân tích tần suất từng vị trí"""
        recent = self.history[-window:] if len(self.history) >= window else self.history
        pos_freq = {pos: {str(i): 0 for i in range(10)} for pos in self.positions}
        
        for draw in recent:
            for idx, pos in enumerate(self.positions):
                if idx < len(draw):
                    pos_freq[pos][draw[idx]] += 1
        return pos_freq
    
    def detect_bong_numbers(self) -> List[str]:
        """
        Phát hiện số "bóng" theo quy tắc:
        - Bóng dương: 0-5, 1-6, 2-7, 3-8, 4-9
        - Bóng âm: 0-7, 1-4, 2-9, 3-6, 5-8
        """
        if len(self.history) < 10:
            return []
            
        last_draw = self.history[-1]
        bong_candidates = set()
        
        # Bóng dương
        duong_map = {'0':'5', '1':'6', '2':'7', '3':'8', '4':'9', 
                     '5':'0', '6':'1', '7':'2', '8':'3', '9':'4'}
        
        # Bóng âm
        am_map = {'0':'7', '1':'4', '2':'9', '3':'6', '4':'1',
                  '5':'8', '6':'3', '7':'0', '8':'5', '9':'2'}
        
        for num in last_draw:
            bong_candidates.add(duong_map[num])
            bong_candidates.add(am_map[num])
        
        return list(bong_candidates)[:5]  # Lấy tối đa 5 số
    
    def detect_cau_bac_thang(self) -> List[str]:
        """Phát hiện cầu bậc thang (tăng/giảm dần)"""
        if len(self.history) < 5:
            return []
        
        candidates = []
        for pos in range(5):  # Duyệt từng vị trí
            values = []
            for draw in self.history[-5:]:
                if pos < len(draw):
                    values.append(int(draw[pos]))
            
            if len(values) >= 3:
                # Kiểm tra xu hướng
                diffs = [values[i+1] - values[i] for i in range(len(values)-1)]
                
                # Nếu các số tăng/giảm đều đặn
                if all(d == diffs[0] for d in diffs) and abs(diffs[0]) == 1:
                    next_val = values[-1] + diffs[0]
                    if 0 <= next_val <= 9:
                        candidates.append(str(next_val))
        
        return candidates
    
    def detect_cau_ke_1_2(self) -> List[str]:
        """Phát hiện cầu kè 1-2 (số cách nhau 1-2 đơn vị thường về cùng nhau)"""
        if len(self.history) < 20:
            return []
        
        pair_counts = Counter()
        for draw in self.history[-50:]:
            nums = [int(n) for n in draw]
            for i, j in itertools.combinations(nums, 2):
                if abs(i - j) in [1, 2]:  # Kè 1 hoặc 2
                    pair = tuple(sorted([i, j]))
                    pair_counts[pair] += 1
        
        # Lấy các cặp hay về nhất
        common_pairs = [p for p, c in pair_counts.most_common(5) if c >= 3]
        
        # Kết hợp với số cuối cùng
        if self.history:
            last_nums = [int(n) for n in self.history[-1]]
            candidates = []
            for pair in common_pairs:
                for num in last_nums:
                    if abs(num - pair[0]) in [1, 2]:
                        candidates.append(str(pair[1]))
                    if abs(num - pair[1]) in [1, 2]:
                        candidates.append(str(pair[0]))
            return list(set(candidates))[:5]
        return []
    
    def detect_dao_cau(self) -> Tuple[bool, List[str]]:
        """
        Phát hiện nhà cái đảo cầu
        Trả về: (có đảo cầu không, các số an toàn)
        """
        if len(self.history) < 20:
            return False, []
        
        # Tính tần suất các số ở 20 kỳ gần
        recent_20 = ''.join(self.history[-20:])
        recent_counts = Counter(recent_20)
        
        # Tính tần suất ở 5 kỳ gần nhất
        recent_5 = ''.join(self.history[-5:])
        recent_5_counts = Counter(recent_5)
        
        # Nếu số hay về đột ngột ít về
        hot_numbers = [n for n, c in recent_counts.most_common(5) if c >= 3]
        cold_in_recent = [n for n in hot_numbers if recent_5_counts.get(n, 0) <= 1]
        
        is_dao = len(cold_in_recent) >= 2
        
        # Số an toàn khi đảo cầu: số lạnh (ít về) và số bóng
        cold_numbers = [n for n in range(10) if recent_counts.get(str(n), 0) <= 2]
        bong_numbers = self.detect_bong_numbers()
        
        safe_numbers = list(set(map(str, cold_numbers + bong_numbers)))
        
        return is_dao, safe_numbers[:7]
    
    def predict_3so5tinh(self) -> Dict:
        """
        Thuật toán chính dự đoán 3 số 5 tính
        Trả về: {"dan4": [...], "dan3": [...], "logic": "..."}
        """
        if len(self.history) < 10:
            return {
                "dan4": ["0", "1", "2", "3"],
                "dan3": ["4", "5", "6"],
                "logic": "⚠️ Cần thêm dữ liệu (tối thiểu 10 kỳ)"
            }
        
        logic_parts = []
        candidates = Counter()
        
        # 1. Phân tích tần suất từng vị trí
        pos_freq = self.analyze_position_frequency(30)
        hot_by_position = []
        for pos in self.positions:
            sorted_nums = sorted(pos_freq[pos].items(), key=lambda x: x[1], reverse=True)
            hot_by_position.extend([n for n, _ in sorted_nums[:2]])
        
        for num in hot_by_position[:8]:
            candidates[num] += 3
        logic_parts.append("📊 Phân tích vị trí")
        
        # 2. Phát hiện bóng số
        bong_nums = self.detect_bong_numbers()
        for num in bong_nums:
            candidates[num] += 2
        if bong_nums:
            logic_parts.append(f"🔄 Bóng số: {', '.join(bong_nums)}")
        
        # 3. Cầu bậc thang
        stair_nums = self.detect_cau_bac_thang()
        for num in stair_nums:
            candidates[num] += 4  # Ưu tiên cao
        if stair_nums:
            logic_parts.append(f"📈 Cầu bậc thang: {', '.join(stair_nums)}")
        
        # 4. Cầu kè 1-2
        ke_nums = self.detect_cau_ke_1_2()
        for num in ke_nums:
            candidates[num] += 2
        if ke_nums:
            logic_parts.append(f"🔗 Cầu kè 1-2: {', '.join(ke_nums)}")
        
        # 5. Phát hiện đảo cầu
        is_dao, safe_nums = self.detect_dao_cau()
        if is_dao:
            # Reset candidates, ưu tiên số an toàn
            candidates = Counter()
            for num in safe_nums[:7]:
                candidates[num] += 5
            logic_parts.append(f"⚠️ PHÁT HIỆN ĐẢO CẦU - Ưu tiên số lạnh/bóng")
        else:
            logic_parts.append("✅ Cầu ổn định")
        
        # 6. Phân tích xu hướng tổng (sum) và chẵn lẻ
        if len(self.history) >= 10:
            sums = [sum(int(d) for d in draw) for draw in self.history[-10:]]
            avg_sum = np.mean(sums)
            
            if avg_sum > 22.5:  # Tổng cao
                candidates.update([str(i) for i in range(5, 10)] * 2)
                logic_parts.append("📈 Xu hướng tổng CAO")
            else:
                candidates.update([str(i) for i in range(0, 5)] * 2)
                logic_parts.append("📉 Xu hướng tổng THẤP")
        
        # 7. Lấy top 7 số
        top_numbers = [num for num, _ in candidates.most_common(7)]
        
        # Đảm bảo đủ 7 số
        if len(top_numbers) < 7:
            all_nums = list(map(str, range(10)))
            for num in all_nums:
                if num not in top_numbers:
                    top_numbers.append(num)
                if len(top_numbers) >= 7:
                    break
        
        # Chia thành 4 số chủ lực và 3 số lót
        dan4 = top_numbers[:4]
        dan3 = top_numbers[4:7]
        
        # Logic tổng hợp
        logic_summary = " | ".join(logic_parts[-3:])  # Lấy 3 logic gần nhất
        
        # Lưu lịch sử dự đoán
        prediction_record = {
            "dan4": dan4,
            "dan3": dan3,
            "logic": logic_summary,
            "timestamp": len(self.history)
        }
        st.session_state.prediction_history.append(prediction_record)
        
        return {
            "dan4": dan4,
            "dan3": dan3,
            "logic": f"📌 {logic_summary} | Dựa trên {min(50, len(self.history))} kỳ gần nhất"
        }

# ================= UI DESIGN (Tối giản - Chống nhầm số) =================
st.set_page_config(page_title="TITAN v21.0 PRO", layout="centered")
st.markdown("""
    <style>
    .stApp { background: #010409; color: #c9d1d9; }
    .status-active { color: #238636; font-weight: bold; border-left: 3px solid #238636; padding-left: 10px; }
    .status-warning { color: #f0883e; font-weight: bold; border-left: 3px solid #f0883e; padding-left: 10px; }
    .prediction-card {
        background: #0d1117; border: 2px solid #30363d;
        border-radius: 12px; padding: 25px; margin-top: 15px;
        box-shadow: 0 4px 20px rgba(0,0,0,0.5);
    }
    .num-display { 
        font-size: 60px; font-weight: 900; color: #58a6ff; 
        text-align: center; letter-spacing: 10px; text-shadow: 0 0 25px #58a6ff;
    }
    .logic-box { 
        font-size: 14px; color: #8b949e; background: #161b22; 
        padding: 15px; border-radius: 8px; margin-bottom: 20px;
        border-left: 4px solid #58a6ff;
    }
    .stats-box {
        background: #0d1117;
        border: 1px solid #30363d;
        border-radius: 6px;
        padding: 10px;
        font-size: 13px;
        color: #8b949e;
    }
    </style>
""", unsafe_allow_html=True) 

st.markdown("<h2 style='text-align: center; color: #58a6ff;'>🧬 TITAN v21.0 OMNI - 3 SỐ 5 TÍNH</h2>", unsafe_allow_html=True)

# Hiển thị trạng thái kết nối
col_status1, col_status2 = st.columns(2)
with col_status1:
    if neural_engine:
        st.markdown(f"<p class='status-active'>● NEURAL: ONLINE</p>", unsafe_allow_html=True)
    else:
        st.markdown(f"<p class='status-warning'>● NEURAL: OFFLINE (Dùng thuật toán cục bộ)</p>", unsafe_allow_html=True)

with col_status2:
    st.markdown(f"<p class='status-active'>📊 DỮ LIỆU: {len(st.session_state.history)} KỲ</p>", unsafe_allow_html=True)

# ================= XỬ LÝ DỮ LIỆU =================
raw_input = st.text_area("📡 NẠP DỮ LIỆU (Dán các dãy 5 số):", height=100, 
                        placeholder="32880\n21808\n12664\n... Mỗi dòng 1 kỳ") 

col1, col2, col3 = st.columns([1, 1, 1])
with col1:
    if st.button("🚀 DỰ ĐOÁN NGAY"):
        new_data = re.findall(r"\d{5}", raw_input)
        if new_data:
            st.session_state.history.extend(new_data)
            save_memory(st.session_state.history)
        
        # Sử dụng thuật toán nâng cao
        if len(st.session_state.history) >= 5:
            predictor = TitanPredictor(st.session_state.history[-200:])  # Dùng 200 kỳ gần nhất
            result = predictor.predict_3so5tinh()
            st.session_state.last_result = result
        else:
            st.warning("⚠️ Cần ít nhất 5 kỳ dữ liệu để dự đoán")
        st.rerun()

with col2:
    if st.button("🤖 GỌI AI (NÂNG CAO)"):
        new_data = re.findall(r"\d{5}", raw_input)
        if new_data:
            st.session_state.history.extend(new_data)
            save_memory(st.session_state.history)
        
        if neural_engine and len(st.session_state.history) >= 20:
            # Gửi Prompt "Bẫy nhà cái" cho AI
            recent_data = st.session_state.history[-100:]
            prompt = f"""
            Bạn là AI chuyên gia xác suất 5D cho trò chơi "3 số 5 tính".
            
            QUY TẮC: Người chơi chọn 3 số bất kỳ từ 0-9. Thắng nếu 3 số này xuất hiện trong 5 số kết quả (không cần đúng thứ tự).
            
            DỮ LIỆU 100 KỲ GẦN NHẤT:
            {recent_data}
            
            PHÂN TÍCH CHUYÊN SÂU:
            1. Xác định các số đang "bệt" (xuất hiện liên tục)
            2. Xác định các số "bóng" sắp nổ (theo bóng dương: 0-5,1-6,2-7,3-8,4-9 và bóng âm: 0-7,1-4,2-9,3-6,5-8)
            3. Phát hiện "cầu kè" (các cặp số thường về cùng nhau: 1-2, 3-4, 5-6, 7-8, 8-9...)
            4. Phát hiện nếu nhà cái đang "đảo cầu" (các số nóng bỗng dưng vắng mặt)
            5. Dự đoán xu hướng 5-10 kỳ tới
            
            YÊU CẦU: Trả về JSON chính xác với format:
            {{
                "dan4": [4 số chủ lực nhất, sắp xếp theo thứ tự ưu tiên],
                "dan3": [3 số lót, an toàn],
                "logic": "Giải thích ngắn gọn (dưới 100 từ) về thuật toán và lý do chọn số"
            }}
            
            CHỈ TRẢ VỀ JSON, không thêm text khác.
            """
            
            try:
                with st.spinner("AI đang phân tích dữ liệu..."):
                    response = neural_engine.generate_content(prompt)
                    res_text = response.text
                    # Tìm JSON trong response
                    json_match = re.search(r'(\{.*\})', res_text, re.DOTALL)
                    if json_match:
                        data = json.loads(json_match.group())
                        # Đảm bảo có đủ các trường
                        if "dan4" in data and "dan3" in data and "logic" in data:
                            st.session_state.last_result = data
                        else:
                            st.error("AI trả về thiếu dữ liệu, dùng thuật toán dự phòng")
                            predictor = TitanPredictor(st.session_state.history[-200:])
                            st.session_state.last_result = predictor.predict_3so5tinh()
                    else:
                        st.error("Không parse được JSON từ AI")
                        predictor = TitanPredictor(st.session_state.history[-200:])
                        st.session_state.last_result = predictor.predict_3so5tinh()
            except Exception as e:
                st.error(f"Lỗi AI: {str(e)[:50]}... Dùng thuật toán cục bộ")
                predictor = TitanPredictor(st.session_state.history[-200:])
                st.session_state.last_result = predictor.predict_3so5tinh()
        else:
            if not neural_engine:
                st.warning("⚠️ AI chưa kết nối, dùng thuật toán cục bộ")
            if len(st.session_state.history) < 20:
                st.warning(f"⚠️ Cần 20 kỳ để dùng AI (hiện có {len(st.session_state.history)})")
            
            predictor = TitanPredictor(st.session_state.history[-200:] if st.session_state.history else [])
            st.session_state.last_result = predictor.predict_3so5tinh()
        st.rerun()

with col3:
    if st.button("🗑️ RESET BỘ NHỚ"):
        st.session_state.history = []
        st.session_state.prediction_history = []
        if os.path.exists(DB_FILE): 
            os.remove(DB_FILE)
        st.rerun()

# Hiển thị thống kê nhanh
if len(st.session_state.history) > 0:
    with st.expander("📊 Thống kê nhanh", expanded=False):
        last_10 = st.session_state.history[-10:] if len(st.session_state.history) >= 10 else st.session_state.history
        
        # Tần suất các số
        all_nums = ''.join(last_10)
        num_counts = Counter(all_nums)
        
        cols = st.columns(5)
        for i in range(5):
            with cols[i]:
                st.markdown(f"**Số {i}**")
                st.progress(num_counts.get(str(i), 0) / max(1, max(num_counts.values())))
                st.caption(f"{num_counts.get(str(i), 0)} lần")
        
        cols2 = st.columns(5)
        for i in range(5, 10):
            with cols2[i-5]:
                st.markdown(f"**Số {i}**")
                st.progress(num_counts.get(str(i), 0) / max(1, max(num_counts.values())))
                st.caption(f"{num_counts.get(str(i), 0)} lần")

# ================= HIỂN THỊ KẾT QUẢ =================
if "last_result" in st.session_state:
    res = st.session_state.last_result
    st.markdown("<div class='prediction-card'>", unsafe_allow_html=True)
    
    # Hiển thị phân tích
    st.markdown(f"<div class='logic-box'><b>💡 PHÂN TÍCH THUẬT TOÁN:</b><br>{res['logic']}</div>", unsafe_allow_html=True)
    
    # Hiển thị 4 số chủ lực
    st.markdown("<p style='text-align:center; font-size:14px; color:#888;'>🎯 4 SỐ CHỦ LỰC (ĐẶT CƯỢC CHÍNH)</p>", unsafe_allow_html=True)
    dan4_str = '  '.join(map(str, res['dan4']))
    st.markdown(f"<div class='num-display'>{dan4_str}</div>", unsafe_allow_html=True)
    
    # Giải thích cho 4 số
    if len(res['dan4']) == 4:
        st.caption(f"✨ Gợi ý kết hợp: {res['dan4'][0]}{res['dan4'][1]}{res['dan4'][2]}, {res['dan4'][0]}{res['dan4'][1]}{res['dan4'][3]}, {res['dan4'][0]}{res['dan4'][2]}{res['dan4'][3]}, {res['dan4'][1]}{res['dan4'][2]}{res['dan4'][3]}")
    
    # Hiển thị 3 số lót
    st.markdown("<p style='text-align:center; font-size:14px; color:#888; margin-top:25px;'>🛡️ 3 SỐ LÓT (GIỮ VỐN, BAO THÊM)</p>", unsafe_allow_html=True)
    dan3_str = '  '.join(map(str, res['dan3']))
    st.markdown(f"<div class='num-display' style='color:#f2cc60; text-shadow: 0 0 25px #f2cc60;'>{dan3_str}</div>", unsafe_allow_html=True)
    
    # Tạo dàn 7 số để sao chép
    all_numbers = res['dan4'] + res['dan3']
    copy_val = " ".join(all_numbers)
    
    # Form để dễ copy
    with st.form(key="copy_form"):
        st.text_input("📋 DÀN 7 SỐ (copy paste):", value=copy_val, key="copy_input")
        st.form_submit_button("📋 Copy", on_click=lambda: st.write("Đã copy!"))  # Streamlit tự xử lý copy
    
    st.markdown("</div>", unsafe_allow_html=True)
    
    # Hiển thị lịch sử dự đoán
    if st.session_state.prediction_history:
        with st.expander("📜 Lịch sử dự đoán gần nhất", expanded=False):
            for i, pred in enumerate(st.session_state.prediction_history[-5:]):
                st.markdown(f"**Lần {i+1}**: {' '.join(pred['dan4'])} + {' '.join(pred['dan3'])}")
                st.caption(f"_{pred['logic']}_")
                st.divider()

# Hiển thị hướng dẫn
with st.expander("📖 Hướng dẫn - Quy tắc 3 số 5 tính", expanded=False):
    st.markdown("""
    ### 🎯 QUY TẮC "3 SỐ 5 TÍNH"
    
    **Cách chơi:**
    - Chọn 3 số bất kỳ từ 0-9 (ví dụ: 1, 2, 6)
    - Kết quả xổ 5 số (hàng Chục ngàn, Ngàn, Trăm, Chục, Đơn vị)
    - **THẮNG** nếu 3 số bạn chọn đều xuất hiện trong 5 số kết quả (không cần đúng thứ tự)
    
    **Ví dụ:**
    - ✅ Chọn: 1,2,6 - Kết quả: 12864 → THẮNG (có đủ 1,2,6)
    - ❌ Chọn: 1,3,6 - Kết quả: 12662 → THUA (thiếu số 3)
    
    **Mẹo:**
    - Đánh dàn 7 số rồi chọn 3 số bất kỳ trong dàn để tạo vé
    - Kết hợp số chủ lực + số lót để tối ưu xác suất
    """)

st.markdown("<br><p style='text-align:center; font-size:11px; color:#444;'>TITAN v21.0 OMNI - Thuật toán độc quyền | Tự động cập nhật dữ liệu</p>", unsafe_allow_html=True)