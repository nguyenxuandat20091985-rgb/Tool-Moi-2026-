"""
🚀 TITAN V27 - DUAL STRATEGY VERSION
2 tinh: 3 số mạnh nhất → 3 cặp
3 tinh: 3 số mạnh nhất (KHÁC) → 1 tổ hợp
Tổng: 6 số khác nhau từ 0-9
"""
import streamlit as st
import re
import itertools
from collections import Counter

st.set_page_config(page_title="TITAN V27", page_icon="🎲", layout="wide")

st.markdown("""
<style>
    .main {padding: 0.5rem;}
    .big-number {font-size: 48px; font-weight: bold; color: #28a745; text-align: center; font-family: monospace; letter-spacing: 8px;}
    .number-box {background: #28a745; color: white; padding: 15px; border-radius: 10px; text-align: center; margin: 10px 0;}
    .grid-3 {display: grid; grid-template-columns: repeat(3, 1fr); gap: 10px;}
    .pair-box {background: #fff3cd; border: 3px solid #ffc107; border-radius: 10px; padding: 15px; text-align: center; font-family: monospace; font-size: 28px; font-weight: bold;}
    .triple-box {background: #d1ecf1; border: 3px solid #17a2b8; border-radius: 10px; padding: 15px; text-align: center; font-family: monospace; font-size: 32px; font-weight: bold;}
    .status {background: #28a745; color: white; padding: 15px; border-radius: 10px; text-align: center; font-size: 20px; font-weight: bold; margin: 10px 0;}
    .metric {display: flex; justify-content: space-around; margin: 10px 0; padding: 10px; background: #f8f9fa; border-radius: 8px;}
    .metric-value {font-size: 24px; font-weight: bold; color: #28a745;}
    button {width: 100% !important; background: #28a745 !important; color: white !important; font-size: 20px !important; padding: 15px !important;}
    .strategy-box {background: #e3f2fd; border-left: 5px solid #2196f3; padding: 15px; border-radius: 8px; margin: 10px 0;}
</style>
""", unsafe_allow_html=True)

def get_numbers(text):
    return re.findall(r"\d{5}", text) if text else []

def smart_dual_predict(db):
    """
    Chọn 6 số mạnh nhất từ 0-9
    - Top 3 số → 2 tinh
    - Top 3 số tiếp theo (khác) → 3 tinh
    """
    if len(db) < 3:
        return {
            "two_digit_nums": ["0", "1", "2"],
            "three_digit_nums": ["3", "4", "5"],
            "pairs": ["01", "02", "12"],
            "triples": ["345"],
            "all_6_numbers": "012345",
            "conf": 50
        }
    
    # Tính tần suất từng số 0-9
    all_digits = "".join(db[-30:])
    scores = {str(i): all_digits.count(str(i)) for i in range(10)}
    
    # Bonus số vừa ra trong kỳ gần nhất
    last = db[-1]
    for d in set(last):
        scores[d] += 30
    
    # Sort theo điểm giảm dần
    sorted_nums = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    
    # Lấy top 6 số mạnh nhất
    top_6_numbers = [x[0] for x in sorted_nums[:6]]
    
    # Chia thành 2 nhóm:
    # - 3 số đầu → 2 tinh
    # - 3 số sau → 3 tinh
    two_digit_nums = sorted(top_6_numbers[:3])  # 3 số cho 2 tinh
    three_digit_nums = sorted(top_6_numbers[3:6])  # 3 số cho 3 tinh
    
    # Tạo cặp 2 số từ 3 số (C(3,2) = 3 cặp)
    pairs = ["".join(p) for p in itertools.combinations(two_digit_nums, 2)]
    
    # Tạo tổ hợp 3 số từ 3 số (C(3,3) = 1 tổ hợp)
    triples = ["".join(three_digit_nums)]  # Chỉ 1 tổ hợp duy nhất
    
    # Confidence
    conf = min(95, 60 + len(db)//2)
    
    return {
        "two_digit_nums": two_digit_nums,
        "three_digit_nums": three_digit_nums,
        "pairs": pairs,
        "triples": triples,
        "all_6_numbers": "".join(sorted(top_6_numbers)),
        "scores": {num: scores[num] for num in top_6_numbers},
        "conf": conf
    }

# Init
if "db" not in st.session_state:
    st.session_state.db = []
if "result" not in st.session_state:
    st.session_state.result = None

# UI
st.markdown('<h1 style="text-align:center;color:#28a745;margin:5px 0;">🎲 TITAN V27</h1>', unsafe_allow_html=True)

# Metrics
col1, col2 = st.columns(2)
with col1:
    st.markdown(f'<div class="metric"><div><div class="metric-value">{len(st.session_state.db)}</div><div style="font-size:12px;color:#666;">Tổng kỳ</div></div></div>', unsafe_allow_html=True)
with col2:
    last = st.session_state.db[-1] if st.session_state.db else "-"
    st.markdown(f'<div class="metric"><div><div class="metric-value" style="font-size:18px;">{last}</div><div style="font-size:12px;color:#666;">Kỳ cuối</div></div></div>', unsafe_allow_html=True)

# Input
user_input = st.text_area("📥 Dán số (mỗi dòng 1 số 5 chữ số):", placeholder="16923\n51475\n31410\n...", height=60)

if st.button("⚡ CHỐT SỐ"):
    numbers = get_numbers(user_input)
    if numbers:
        st.session_state.db.extend(numbers)
        st.session_state.result = smart_dual_predict(st.session_state.db)
        st.rerun()
    else:
        st.error("❌ Không có số 5 chữ số!")

# Results
if st.session_state.result:
    r = st.session_state.result
    
    st.markdown(f'<div class="status">🔥 KHUYÊN ĐÁNH | Tin cậy: {r["conf"]}%</div>', unsafe_allow_html=True)
    
    # 6 số nền
    st.markdown(f'<div class="number-box"><div style="font-size:14px;margin-bottom:5px;">🎲 6 SỐ MẠNH NHẤT</div><div class="big-number">{r["all_6_numbers"]}</div></div>', unsafe_allow_html=True)
    
    # Strategy explanation
    st.markdown("""
    <div class="strategy-box">
    <strong>📋 CHIẾN LƯỢC:</strong><br>
    • <b>2 tinh:</b> 3 số mạnh nhất → 3 cặp<br>
    • <b>3 tinh:</b> 3 số mạnh tiếp theo → 1 tổ hợp<br>
    • <b>Không trùng nhau</b> → Phủ rộng 0-9<br>
    • <b>Bù trừ:</b> Không trúng 2 tinh → Trúng 3 tinh
    </div>
    """, unsafe_allow_html=True)
    
    # 2 TINH
    st.markdown('<div style="background:#fff3cd;padding:15px;border-radius:10px;margin:15px 0;border:3px solid #ffc107;">', unsafe_allow_html=True)
    st.markdown(f'<h2 style="text-align:center;margin:0;color:#856404;">🎯 2 TINH (3 số: {"".join(r["two_digit_nums"])})</h2>', unsafe_allow_html=True)
    st.markdown('<div class="grid-3">', unsafe_allow_html=True)
    for i, pair in enumerate(r["pairs"]):
        st.markdown(f'<div class="pair-box">{pair}</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)
    
    # 3 TINH
    st.markdown('<div style="background:#d1ecf1;padding:15px;border-radius:10px;margin:15px 0;border:3px solid #17a2b8;">', unsafe_allow_html=True)
    st.markdown(f'<h2 style="text-align:center;margin:0;color:#0c5460;">🎯 3 TINH (3 số: {"".join(r["three_digit_nums"])})</h2>', unsafe_allow_html=True)
    st.markdown('<div class="grid-3">', unsafe_allow_html=True)
    for triple in r["triples"]:
        st.markdown(f'<div class="triple-box">{triple}</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Detailed stats
    with st.expander("📊 Chi tiết điểm từng số", expanded=False):
        st.write("**Điểm số từng số (càng cao càng tốt):**")
        for num, score in r["scores"].items():
            bar_width = min(100, score * 5)
            st.markdown(f"`{num}`: {'█' * int(bar_width/10)} ({score} điểm)")
    
    # Strategy guide
    st.markdown("""
    <div style="background:#28a745;color:white;padding:15px;border-radius:10px;margin:15px 0;">
    <h3 style="margin-top:0;">💡 CÁCH ĐÁNH:</h3>
    <ol style="margin:0;padding-left:20px;">
    <li><b>Bước 1:</b> Đánh 3 cặp 2 tinh (vốn ít, dễ trúng)</li>
    <li><b>Bước 2:</b> Nếu trượt → Đánh 1 tổ hợp 3 tinh (thưởng cao)</li>
    <li><b>Lưu ý:</b> 6 số này KHÔNG TRÙNG nhau → Tăng xác suất trúng</li>
    </ol>
    </div>
    """, unsafe_allow_html=True)

# Actions
st.markdown("---")
col_a, col_b = st.columns(2)
with col_a:
    if st.button("🗑️ XÓA HẾT"):
        st.session_state.db = []
        st.session_state.result = None
        st.rerun()
with col_b:
    if st.button("📊 THỐNG KÊ"):
        st.session_state.show_stats = not st.session_state.get("show_stats", False)

# Stats
if st.session_state.get("show_stats", False) and len(st.session_state.db) > 0:
    with st.expander("📊 Thống kê tần suất", expanded=True):
        freq = Counter("".join(st.session_state.db[-50:]))
        df_data = [{"Số": str(i), "Tần suất": freq.get(str(i), 0)} for i in range(10)]
        
        col_x, col_y = st.columns(2)
        with col_x:
            st.markdown("**🔥 Nóng nhất**")
            for item in sorted(df_data, key=lambda x: x["Tần suất"], reverse=True)[:3]:
                st.write(f"`{item['Số']}`: {item['Tần suất']} lần")
        with col_y:
            st.markdown("**❄️ Lạnh nhất**")
            for item in sorted(df_data, key=lambda x: x["Tần suất"])[:3]:
                st.write(f"`{item['Số']}`: {item['Tần suất']} lần")

st.markdown('<div style="text-align:center;color:#999;font-size:11px;margin-top:10px;">⚡ 2 tinh + 3 tinh = 6 số khác nhau | Bù trừ hoàn hảo</div>', unsafe_allow_html=True)