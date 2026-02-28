import streamlit as st
from datetime import datetime
import time
import re

st.set_page_config(page_title="TITAN v34.1 - CHỮ TO", layout="centered", page_icon="⚡")

# --- CSS CHỮ TO + MÀU TƯƠNG PHẢN CAO ---
st.markdown("""
<style>
    .recommendation { 
        background: #FFFFFF;
        padding: 30px; 
        border-radius: 15px; 
        text-align: center;
        box-shadow: 0 4px 15px rgba(0,0,0,0.3); 
        margin: 10px 0;
        border: 3px solid #333333;
    }
    .title { 
        font-size: 2em; 
        font-weight: bold; 
        color: #000000;
        margin: 10px 0;
    }
    .label { 
        font-size: 1.3em; 
        font-weight: bold; 
        color: #333333;
        margin: 15px 0 5px 0;
    }
    .tai { 
        color: #FF0000; 
        font-weight: bold; 
        font-size: 3em;
        text-shadow: 2px 2px 4px #00000030;
        margin: 10px 0;
    }
    .xiu { 
        color: #0066CC; 
        font-weight: bold; 
        font-size: 3em;
        text-shadow: 2px 2px 4px #00000030;
        margin: 10px 0;
    }
    .bet-amount { 
        font-size: 2em; 
        color: #00AA00; 
        font-weight: bold;
        margin: 10px 0;
    }
    .odds { 
        font-size: 1.5em; 
        color: #333333; 
        font-weight: bold;
        margin: 10px 0;
    }
    .reason { 
        font-size: 1.3em; 
        color: #FF6600; 
        font-weight: bold;
        margin: 15px 0;
        background: #FFF3E0;
        padding: 10px;
        border-radius: 8px;
    }
    .confidence { 
        font-size: 1.3em; 
        color: #333333; 
        font-weight: bold;
        margin: 10px 0;
    }
    .countdown { 
        font-size: 1.5em; 
        color: #FF0000; 
        font-weight: bold;
        text-align: center;
        padding: 10px;
        background: #FFEEEE;
        border-radius: 8px;
        margin: 10px 0;
    }
</style>
""", unsafe_allow_html=True)

# --- SESSION STATE ---
if 'bankroll' not in st.session_state:
    st.session_state.bankroll = 500000
if 'period_count' not in st.session_state:
    st.session_state.period_count = 690

# --- HÀM PHÂN TÍCH ---
def quick_analyze(raw_text):
    lines = [re.sub(r'[^\d]', '', l.strip()) for l in raw_text.strip().split('\n')]
    valid = [l for l in lines if len(l) == 5]
    
    if len(valid) < 5:
        return None
    
    positions = ["Chục Ngàn", "Ngàn", "Trăm", "Chục", "Đơn Vị"]
    best_pick = None
    best_score = -1
    
    for pos_idx, pos_name in enumerate(positions):
        digits = [int(line[pos_idx]) for line in valid[:10]]
        tai_count = sum(1 for d in digits if d >= 5)
        tai_rate = tai_count / len(digits)
        
        if tai_rate >= 0.7:
            score = tai_rate
            pred = "XỈU"
            reason = f"Bệt Tài {tai_count}/10 → Bẻ cầu"
        elif tai_rate <= 0.3:
            score = 1 - tai_rate
            pred = "TÀI"
            reason = f"Bệt Xỉu {10-tai_count}/10 → Bẻ cầu"
        else:
            continue
        
        if score > best_score:
            best_score = score
            best_pick = {
                "position": pos_name,
                "bet": pred,
                "reason": reason,
                "confidence": int(score * 100),
                "bet_amount": min(20000, int(st.session_state.bankroll * 0.02))
            }
    
    if not best_pick:
        last_digit = int(valid[0][4])
        best_pick = {
            "position": "Đơn Vị",
            "bet": "TÀI" if last_digit < 5 else "XỈU",
            "reason": "Cầu nhảy → Theo kỳ trước ngược",
            "confidence": 55,
            "bet_amount": min(10000, int(st.session_state.bankroll * 0.01))
        }
    
    return best_pick

# --- GIAO DIỆN ---
st.title("⚡ TITAN v34.1 - CHỮ TO RÕ RÀNG")

# Countdown
now = datetime.now()
seconds = now.second
remaining = 60 - seconds if seconds < 30 else 30 - (seconds - 30)
st.markdown(f'<div class="countdown">🕒 KỲ TIẾP THEO SAU: {remaining:02d} GIÂY</div>', unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.header("💰 VỐN")
    st.session_state.bankroll = st.number_input("Vốn hiện tại (đ)", value=st.session_state.bankroll, step=10000)
    st.info(f"✅ Cược đề xuất: 1-2% vốn = {min(20000, int(st.session_state.bankroll*0.02)):,}đ")

# Form
with st.form("quick_form", clear_on_submit=False):
    raw = st.text_area(
        "📥 DÁN 10 KỲ GẦN NHẤT (Mới nhất trên cùng):",
        placeholder="87746\n56421\n69137\n...",
        height=150
    )
    col1, col2 = st.columns([3, 1])
    with col1:
        go = st.form_submit_button("⚡ PHÂN TÍCH NGAY", type="primary", use_container_width=True)
    with col2:
        st.form_submit_button("🗑️ XOÁ", use_container_width=True)

# Kết quả - SỬ DỤNG st.markdown VỚI unsafe_allow_html=True
if go and raw:
    rec = quick_analyze(raw)
    
    if rec:
        st.session_state.period_count += 1
        
        bet_class = "tai" if rec['bet'] == "TÀI" else "xiu"
        profit = int(rec['bet_amount'] * 0.985)
        
        # Tạo HTML string
        html_content = f"""
        <div class="recommendation">
            <div class="title">🎯 KHUYẾN NGHỊ KỲ {st.session_state.period_count}</div>
            <hr style="border: 2px solid #333333;">
            
            <div class="label">📍 VỊ TRÍ:</div>
            <div style="font-size: 2em; font-weight: bold; color: #FF6600;">{rec['position']}</div>
            
            <div class="label">🔴 ĐÁNH:</div>
            <div class="{bet_class}">{rec['bet']}</div>
            
            <div class="label">💰 CƯỢC:</div>
            <div class="bet-amount">{rec['bet_amount']:,}đ</div>
            
            <div class="odds">🎯 Odds: 1.985 → Thắng +{profit:,}đ</div>
            
            <hr style="border: 2px solid #333333;">
            
            <div class="reason">📊 {rec['reason']}</div>
            
            <div class="confidence">⚡ Độ tin cậy: {"█" * (rec['confidence']//10)}{"░" * (10 - rec['confidence']//10)} {rec['confidence']}%</div>
        </div>
        """
        
        # Render HTML - QUAN TRỌNG: unsafe_allow_html=True
        st.markdown(html_content, unsafe_allow_html=True)
        
        # Nút hành động
        c1, c2 = st.columns(2)
        with c1:
            if st.button("✅ THẮNG", type="primary", use_container_width=True, key="win"):
                st.session_state.bankroll += profit
                st.balloons()
                st.success(f"🎉 +{profit:,}đ")
                st.rerun()
        with c2:
            if st.button("❌ THUA", type="secondary", use_container_width=True, key="lose"):
                st.session_state.bankroll -= rec['bet_amount']
                st.error(f"💸 -{rec['bet_amount']:,}đ")
                st.rerun()
        
        # Cảnh báo
        if st.session_state.bankroll < 400000:
            st.error("🛑 DỪNG NGAY! Đã mất >20% vốn. Nghỉ ngơi nhé anh!")

    else:
        st.warning("⚠️ Dữ liệu chưa đủ 5 kỳ hợp lệ!")

# Footer
st.markdown("---")
st.caption("⚡ TITAN v34.1 | CHỮ TO - MÀU RÕ - 3 GIÂY QUYẾT ĐỊNH | Chơi có trách nhiệm 🙏")