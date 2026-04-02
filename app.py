import streamlit as st
import pandas as pd
import numpy as np
import json

# ================= CẤU HÌNH TRANG =================
st.set_page_config(
    page_title="AI 5D BET PRO V3.0",
    layout="wide",
    page_icon="🤖",
    initial_sidebar_state="collapsed"
)

# ================= CSS TỐI ƯU MOBILE =================
st.markdown("""
    <style>
    @media (max-width: 600px) {
        .main { padding: 10px; }
        .stTextInput > div > div > input, .stTextArea > div > div > textarea {
            font-size: 16px; padding: 10px; height: 45px;
        }
        .stButton > button {
            height: 48px; font-size: 16px; margin-top: 5px;
            width: 100%; border-radius: 8px;
        }
        .metric-card {
            background: #161b22; padding: 12px; border-radius: 10px;
            border: 1px solid #30363d; text-align: center; margin-bottom: 8px;
        }
        .pred-box {
            background: linear-gradient(135deg, #0f2027, #203a43, #2c5364);
            padding: 20px; border-radius: 15px; text-align: center;
            border: 2px solid #00ff88; margin: 15px 0;
        }
        .status-win { color: #00ff88; font-weight: bold; }
        .status-lose { color: #ff4b4b; font-weight: bold; }
        .status-wait { color: #ffd700; font-weight: bold; }
        .conf-bar {
            height: 6px; background: #30363d; border-radius: 3px; margin-top: 5px;
        }
        .conf-fill {
            height: 100%; background: #00ff88; border-radius: 3px; transition: width 0.3s;
        }
    }
    @media (min-width: 601px) {
        .metric-card { display: inline-block; width: 32%; margin-right: 2%; }
    }
    body { background-color: #0d1117; color: #e6edf3; font-family: system-ui, -apple-system, sans-serif; }
    .st-emotion-cache-1v0mbdv { padding: 10px; }
    </style>
""", unsafe_allow_html=True)

# ================= KHỞI TẠO STATE =================
if 'balance' not in st.session_state:
    st.session_state.balance = 100000
if 'history' not in st.session_state:
    st.session_state.history = []
if 'bet_step' not in st.session_state:
    st.session_state.bet_step = 0
if 'last_pred' not in st.session_state:
    st.session_state.last_pred = None
if 'last_conf' not in st.session_state:
    st.session_state.last_conf = 0.0
if 'stop_loss' not in st.session_state:
    st.session_state.stop_loss = 80000 # Cắt lỗ khi còn < 80k (mất 20k)

BET_LEVELS = [1000, 2000, 3000, 5000, 8000, 12000] # Gấp thếp Fibonacci nhẹ
COMMISSION = 0.05 # Phí sàn 5%
CONF_THRESHOLD = 0.58 # Ngưỡng AI tin tưởng

# ================= HÀM LOGIC =================
def parse_bulk_input(text):
    """Tách chuỗi nhập thành list các chuỗi 5 số"""
    raw = text.replace(",", " ").replace(";", " ").split()
    return [s for s in raw if len(s) == 5 and s.isdigit()]

def calc_result(num_str):
    digits = [int(c) for c in num_str]
    total = sum(digits)
    res_type = "TÀI" if total >= 23 else "XỈU"
    return total, res_type

def ai_engine(history_types):
    if len(history_types) < 10:
        return None, 0.0, "Chưa đủ 10 kỳ để học nhịp"
    
    # 1. Markov Transition Matrix
    trans = {"TÀI": {"TÀI": 0, "XỈU": 0}, "XỈU": {"TÀI": 0, "XỈU": 0}}
    for i in range(len(history_types)-1):
        c, n = history_types[i], history_types[i+1]
        trans[c][n] += 1
        
    last = history_types[-1]
    total_next = sum(trans[last].values())
    p_tai_markov = trans[last]["TÀI"] / total_next if total_next > 0 else 0.5
    p_xiu_markov = 1 - p_tai_markov
    
    # 2. Frequency Balance Correction (Luật số lớn)
    count_tai = history_types.count("TÀI")
    count_xiu = history_types.count("XỈU")
    balance_factor = 0.35
    p_tai_adj = p_tai_markov * (1 - balance_factor) + (count_xiu / (count_tai + count_xiu)) * balance_factor
    p_xiu_adj = 1 - p_tai_adj
    
    # 3. Streak Reversal Check
    streak = 1
    for i in range(len(history_types)-2, -1, -1):
        if history_types[i] == last: streak += 1
        else: break
    if streak >= 4:
        # Cầu bệt dài -> xác suất đảo tăng nhẹ
        rev_boost = 0.12
        p_tai_adj = p_tai_adj - rev_boost if last == "TÀI" else p_tai_adj + rev_boost
        p_xiu_adj = 1 - p_tai_adj
        
    # Quyết định & Độ tin cậy
    pred = "TÀI" if p_tai_adj > p_xiu_adj else "XỈU"
    conf = max(p_tai_adj, p_xiu_adj)
    conf = min(max(conf, 0.50), 0.92) # Clamp
    
    reason = "Đang học nhịp..."
    if conf >= CONF_THRESHOLD:
        reason = "Tín hiệu mạnh" if conf > 0.7 else "Tín hiệu trung bình"
    else:
        reason = "Tín hiệu yếu - Khuyến cáo CHỜ"
        
    return pred, conf, reason

def update_bankroll(current_bet, won):
    step = st.session_state.bet_step
    bal = st.session_state.balance
    
    if won:
        # Thắng: cộng lời, reset cấp độ
        profit = current_bet * (1 - COMMISSION)
        bal += profit
        step = 0
        status = "THẮNG ✅"
    else:
        # Thua: trừ tiền, lên cấp
        bal -= current_bet
        step += 1
        status = "THUA ❌"
        
    # Bảo vệ vốn: Reset nếu vượt cấp hoặc vốn dưới stop_loss
    if step >= len(BET_LEVELS) or bal < st.session_state.stop_loss:
        step = 0
        status += " | ⚠️ RESET VỐN"
        
    return bal, step, status

# ================= GIAO DIỆN CHÍNH =================
st.title("🤖 AI 5D BET PRO V3.0")
st.caption("Hệ thống học nhịp nhà cái • Quản lý vốn Fibonacci • Tối ưu Mobile")

# Metrics
c1, c2, c3 = st.columns(3)
c1.markdown(f'<div class="metric-card"><h4>💰 Vốn hiện tại</h4><h2>{st.session_state.balance:,.0f}đ</h2></div>', unsafe_allow_html=True)
c2.markdown(f'<div class="metric-card"><h4>📈 Cấp cược tiếp</h4><h2>{BET_LEVELS[min(st.session_state.bet_step, len(BET_LEVELS)-1)]:,.0f}đ</h2></div>', unsafe_allow_html=True)
win_cnt = sum(1 for h in st.session_state.history if "THẮNG" in h.get("Trạng Thái", ""))
total = len(st.session_state.history)
win_rate = (win_cnt/total*100) if total > 0 else 0
c3.markdown(f'<div class="metric-card"><h4>🎯 Winrate</h4><h2>{win_rate:.1f}%</h2></div>', unsafe_allow_html=True)

# --- NHẬP DỮ LIỆU BULK (10+ KỲ) ---
st.subheader("📥 Nhập dữ liệu ban đầu (≥10 kỳ)")
bulk_input = st.text_area("Dán kết quả cũ (cách nhau dấu cách/phẩy/dòng mới), VD: `12345 56789 00112`", height=80)
if st.button("⏳ LOAD & HỌC NHỊP"):
    nums = parse_bulk_input(bulk_input)
    if len(nums) >= 10:
        for n in nums:
            total, rtype = calc_result(n)
            st.session_state.history.append({
                "Kỳ": len(st.session_state.history)+1, "Kết Quả": n, "Tổng": total,
                "Hệ Thống Ra": rtype, "AI Dự Đoán": "-", "Trạng Thái": "Nạp dữ liệu", "P/L": 0
            })
        st.success(f"✅ Đã nạp {len(nums)} kỳ thành công. AI đã sẵn sàng học nhịp!")
        st.rerun()
    else:
        st.error("❌ Cần nhập đủ 10 kỳ trở lên để AI khởi động.")

# --- NHẬP KỲ LIVE & ĐỐI SOÁT ---
st.subheader("🎲 Nhập kỳ vừa ra & Đối soát")
col_inp, col_btn = st.columns([3, 1])
with col_inp:
    live_input = st.text_input("5 số kỳ hiện tại:", max_chars=5, placeholder="VD: 45891")
with col_btn:
    process_btn = st.button("🔍 PHÂN TÍCH", type="primary", use_container_width=True)

if process_btn:
    if len(live_input) != 5 or not live_input.isdigit():
        st.error("Lỗi: Nhập chính xác 5 chữ số.")
    elif len(st.session_state.history) < 10:
        st.warning("⏳ Vui lòng nạp đủ 10 kỳ ở phần trên trước.")
    else:
        total, res_type = calc_result(live_input)
        
        # Đối soát
        status, pl = "N/A", 0
        current_bet = BET_LEVELS[min(st.session_state.bet_step, len(BET_LEVELS)-1)]
        
        if st.session_state.last_pred:
            won = (st.session_state.last_pred == res_type)
            if won:
                pl = current_bet * (1 - COMMISSION)
            else:
                pl = -current_bet
            st.session_state.balance, st.session_state.bet_step, status = update_bankroll(current_bet, won)
        
        # Lưu history
        st.session_state.history.append({
            "Kỳ": len(st.session_state.history)+1, "Kết Quả": live_input, "Tổng": total,
            "Hệ Thống Ra": res_type, "AI Dự Đoán": st.session_state.last_pred or "N/A",
            "Trạng Thái": status, "P/L": pl
        })
        
        # AI dự đoán kỳ TIẾP THEO
        hist_types = [h["Hệ Thống Ra"] for h in st.session_state.history]
        st.session_state.last_pred, st.session_state.last_conf, reason_msg = ai_engine(hist_types)
        
        st.rerun()

# --- KHUNG DỰ ĐOÁN ---
if st.session_state.last_pred and len(st.session_state.history) >= 10:
    pred = st.session_state.last_pred
    conf = st.session_state.last_conf
    color = "#00ff88" if conf >= CONF_THRESHOLD else "#ffd700"
    action = "ĐÁNH" if conf >= CONF_THRESHOLD else "CHỜ / QUAN SÁT"
    
    st.markdown(f"""
        <div class="pred-box">
            <h3 style="margin:0; color:#aaa;">DỰ ĐOÁN KỲ TIẾP THEO</h3>
            <h1 style="margin:10px 0; color:{color}; font-size:48px;">{pred}</h1>
            <p style="margin:0; color:#ccc;">Độ tin cậy AI: {conf*100:.1f}%</p>
            <div class="conf-bar"><div class="conf-fill" style="width:{conf*100}%; background:{color};"></div></div>
            <p style="margin-top:10px; font-size:18px;">👉 Hành động: <span style="color:{color}; font-weight:bold;">{action}</span></p>
        </div>
    """, unsafe_allow_html=True)

# --- BẢNG LỊCH SỬ (MOBILE OPTIMIZED) ---
if st.session_state.history:
    st.subheader("📋 Nhật ký đối soát")
    df = pd.DataFrame(st.session_state.history).iloc[::-1].head(20)
    
    def color_rows(row):
        colors = []
        for v in row:
            if isinstance(v, str):
                if "THẮNG" in v: colors.append('color: #00ff88; font-weight:bold')
                elif "THUA" in v: colors.append('color: #ff4b4b; font-weight:bold')
                else: colors.append('')
            elif isinstance(v, (int, float)):
                colors.append('color: #00ff88' if v > 0 else 'color: #ff4b4b' if v < 0 else '')
            else: colors.append('')
        return colors
        
    st.dataframe(
        df.style.apply(color_rows, axis=1).format({"P/L": "{:,.0f}đ"}),
        use_container_width=True, height=300, hide_index=True
    )
    
    total_pl = sum(h.get("P/L", 0) for h in st.session_state.history)
    st.metric("📊 Tổng P/L phiên", f"{total_pl:,.0f}đ", delta=f"{'Lời' if total_pl>=0 else 'Lỗ'}")

# --- CONTROLS ---
st.divider()
col_r, col_e = st.columns(2)
if col_r.button("🗑️ Reset Toàn Bộ", use_container_width=True):
    st.session_state.balance = 100000
    st.session_state.history = []
    st.session_state.bet_step = 0
    st.session_state.last_pred = None
    st.session_state.last_conf = 0.0
    st.rerun()
if col_e.button("📤 Xuất CSV", use_container_width=True):
    df_exp = pd.DataFrame(st.session_state.history)
    csv = df_exp.to_csv(index=False).encode('utf-8')
    st.download_button("Tải nhật ký CSV", csv, "ai_5d_log.csv", "text/csv")

# ================= CẢNH BÁO TRÁCH NHIỆM =================
st.markdown("""
    <div style="background:#2a1c1c; border:1px solid #ff4b4b; padding:12px; border-radius:8px; margin-top:20px; font-size:12px; color:#ff9999;">
        ⚠️ <b>Lưu ý quan trọng:</b> Xổ số là trò chơi xác suất ngẫu nhiên. AI chỉ hỗ trợ phân tích xu hướng thống kê, <b>không đảm bảo thắng 100%</b>. 
        Hãy tuân thủ quản lý vốn, không tất tay, và chỉ chơi với số tiền có thể chấp nhận mất. Chơi có trách nhiệm.
    </div>
""", unsafe_allow_html=True)