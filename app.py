import streamlit as st
import re, pandas as pd
import numpy as np
from collections import Counter

# --- CẤU HÌNH ---
LUCKY_OX = [0, 2, 5, 6, 7, 8]
st.set_page_config(page_title="TITAN V48 - POSITION MATRIX", page_icon="🎯", layout="wide")

# --- CSS ---
st.markdown("""
<style>
    .stApp {background-color: #030303; color: #FFD700;}
    .big-num {font-size: 42px; font-weight: bold; color: #00FFCC; text-align: center; font-family: 'Courier New', monospace; letter-spacing: 6px;}
    .box {background: linear-gradient(135deg, #0d1b1b, #030303); color: #FFD700; padding: 20px; border-radius: 15px; text-align: center; border: 1px solid #FFD700; margin-bottom: 15px; box-shadow: 0 4px 20px rgba(255, 215, 0, 0.15);}
    .pos-box {background: linear-gradient(135deg, #1a1a2e, #030303); color: #FFF; padding: 15px; border-radius: 12px; text-align: center; border: 1px solid #444; margin: 5px;}
    .pos-hot {border-color: #FF0000; background: linear-gradient(135deg, #2e1a1a, #030303);}
    .pos-cold {border-color: #00FFCC; background: linear-gradient(135deg, #1a2e2e, #030303);}
    .item {background: linear-gradient(135deg, #00FFCC, #008B8B); color: #000; padding: 15px; border-radius: 10px; text-align: center; font-size: 28px; font-weight: bold;}
    .item-3 {background: linear-gradient(135deg, #FFD700, #FFA500); color: #000;}
    .warning-box {background: rgba(255, 0, 0, 0.3); border: 2px solid #FF0000; color: #FF6666; padding: 15px; border-radius: 10px; margin: 15px 0; text-align: center; font-weight: bold;}
    .success-box {background: rgba(0, 255, 0, 0.2); border: 2px solid #00FF00; color: #00FF00; padding: 15px; border-radius: 10px; margin: 15px 0; text-align: center; font-weight: bold;}
    .digit {display: inline-block; width: 50px; height: 50px; line-height: 50px; border-radius: 50%; background: #333; margin: 3px; font-weight: bold;}
    .digit-hot {background: #FF0000; color: white;}
    .digit-cold {background: #00FFCC; color: black;}
    h1 {text-align: center; color: #FFD700; text-shadow: 0 0 20px rgba(255, 215, 0, 0.5);}
    .position-title {color: #00FFCC; font-size: 18px; font-weight: bold; margin-bottom: 10px;}
</style>
""", unsafe_allow_html=True)

# --- HÀM XỬ LÝ ---
def get_nums(text):
    clean_text = re.sub(r'\s+', '', text)
    return [n for n in re.findall(r"\d{5}", clean_text) if n]

def analyze_position(db, pos):
    """Phân tích từng vị trí (0-4)"""
    if len(db) < 10: return None
    digits = [num[pos] for num in db[-30:]]
    counter = Counter(digits)
    
    # Tìm số nóng (về nhiều nhất trong 10 kỳ gần)
    recent_10 = [num[pos] for num in db[-10:]]
    recent_counter = Counter(recent_10)
    
    # Tìm số gan (không về lâu nhất)
    gan_dict = {}
    for d in "0123456789":
        gan = 0
        for num in reversed(db):
            if num[pos] != d:
                gan += 1
            else:
                break
        gan_dict[d] = gan
    
    hot_nums = [d for d, c in recent_counter.most_common(3)]
    cold_nums = sorted(gan_dict.keys(), key=lambda x: gan_dict[x], reverse=True)[:3]
    
    return {
        "hot": hot_nums,
        "cold": cold_nums,
        "gan": gan_dict,
        "frequency": counter
    }

def predict_v48_position(db):
    if len(db) < 15: return None
    
    positions = {}
    for pos in range(5):
        positions[pos] = analyze_position(db, pos)
    
    # Dự đoán từng vị trí
    predictions = {}
    for pos in range(5):
        pos_data = positions[pos]
        # Ưu tiên số nóng nếu đang trong chu kỳ lặp
        recent_hot = pos_data["hot"][0] if pos_data["hot"] else "0"
        gan_pick = pos_data["cold"][0] if pos_data["cold"] else "0"
        
        # Quyết định chọn nóng hay lạnh dựa trên pattern
        recent_10 = [num[pos] for num in db[-10:]]
        unique_recent = len(set(recent_10))
        
        if unique_recent <= 5:  # Đang lặp nhiều → Theo cầu nóng
            predictions[pos] = {"pick": recent_hot, "type": "HOT", "confidence": 75}
        else:  # Đang đa dạng → Theo số gan
            predictions[pos] = {"pick": gan_pick, "type": "GAN", "confidence": 60}
    
    # Tạo số dự đoán
    predicted_num = "".join([predictions[pos]["pick"] for pos in range(5)])
    avg_confidence = sum(p["confidence"] for p in predictions.values()) // 5
    
    # Phân tích cặp 2 tinh từ các vị trí
    pair_suggestions = []
    hot_digits = []
    for pos in range(5):
        hot_digits.extend(positions[pos]["hot"][:2])
    
    hot_counter = Counter(hot_digits)
    for digit, count in hot_counter.most_common(5):
        pair_suggestions.append(digit)
    
    return {
        "positions": positions,
        "predictions": predictions,
        "predicted_num": predicted_num,
        "confidence": avg_confidence,
        "hot_digits": hot_counter.most_common(5),
        "last_actual": db[-1] if db else None
    }

# --- GIAO DIỆN ---
st.markdown('<h1>🎯 TITAN V48 - POSITION MATRIX</h1>', unsafe_allow_html=True)
st.markdown('<p style="text-align:center; color:#888;">Phân tích từng vị trí - Tăng độ chính xác</p>', unsafe_allow_html=True)

# Cảnh báo trung thực
st.markdown("""
<div class='warning-box'>
⚠️ LƯU Ý QUAN TRỌNG: Không có tool nào dự đoán chính xác 100%.<br>
Tool này phân tích xác suất thống kê, anh em nên quản lý vốn cẩn thận.
</div>
""", unsafe_allow_html=True)

if "history" not in st.session_state: st.session_state.history = []

user_input = st.text_area("📥 Dán bảng kết quả (Kỳ mới nhất ở dưới cùng):", height=200, 
                          value="87558\n34979\n87136\n26404\n71990\n07298\n00443\n02917\n28485\n69200\n57769\n59597\n13385\n76881\n87203\n16695\n37485\n00325\n94144\n56726\n20115\n77579\n38010\n29580\n52771\n15670\n21391")

col1, col2 = st.columns(2)
with col1:
    if st.button("🚀 PHÂN TÍCH VỊ TRÍ"):
        nums = get_nums(user_input)
        if len(nums) >= 15:
            # Kiểm tra kết quả kỳ trước
            if "last_pred" in st.session_state and len(nums) > 1:
                lp = st.session_state.last_pred
                last_actual = nums[-1]
                predicted = lp["predicted_num"]
                matches = sum(1 for i in range(5) if predicted[i] == last_actual[i])
                
                st.session_state.history.insert(0, {
                    "Kỳ": last_actual,
                    "Dự Đoán": predicted,
                    "Đúng": f"{matches}/5",
                    "KQ": "🔥 WIN" if matches >= 3 else "❌"
                })
            
            st.session_state.last_pred = predict_v48_position(nums)
            st.rerun()
        else:
            st.warning("Cần ít nhất 15 kỳ dữ liệu!")

with col2:
    if st.button("🗑️ LÀM MỚI"):
        st.session_state.clear()
        st.rerun()

# --- HIỂN THỊ KẾT QUẢ ---
if "last_pred" in st.session_state:
    res = st.session_state.last_pred
    
    # Số dự đoán
    st.markdown(f"""
    <div class='box'>
    🔮 SỐ DỰ ĐOÁN (5 VỊ TRÍ): <br>
    <span class='big-num'>{res["predicted_num"]}</span>
    <br><br>
    📊 ĐỘ TIN CẬY: {res["confidence"]}%
    </div>
    """, unsafe_allow_html=True)
    
    # Phân tích từng vị trí
    st.markdown("<h3 style='color:#00FFCC; text-align:center;'>📍 PHÂN TÍCH CHI TIẾT 5 VỊ TRÍ</h3>", unsafe_allow_html=True)
    cols = st.columns(5)
    
    pos_names = ["Chục Ngàn", "Ngàn", "Trăm", "Chục", "Đơn Vị"]
    for pos in range(5):
        with cols[pos]:
            pred = res["predictions"][pos]
            pos_data = res["positions"][pos]
            
            hot_color = "pos-hot" if pred["type"] == "HOT" else "pos-cold"
            
            st.markdown(f"""
            <div class='pos-box {hot_color}'>
            <div class='position-title'>{pos_names[pos]}</div>
            <div style='font-size:32px; font-weight:bold; color:#FFD700;'>{pred["pick"]}</div>
            <div style='font-size:11px; margin-top:5px;'>{pred["type"]} - {pred["confidence"]}%</div>
            <div style='font-size:10px; color:#888; margin-top:5px;'>Nóng: {",".join(pos_data["hot"][:2]) if pos_data["hot"] else "-"}</div>
            </div>
            """, unsafe_allow_html=True)
    
    # Các số nóng tổng thể
    st.markdown("<h3 style='color:#FFD700; text-align:center; margin-top:30px;'>🔥 CÁC SỐ NÓNG TỔNG THỂ</h3>", unsafe_allow_html=True)
    hot_html = ""
    for digit, count in res["hot_digits"]:
        hot_html += f'<span class="digit digit-hot">{digit}</span>'
    st.markdown(f"<div style='text-align:center; font-size:20px;'>{hot_html}</div>", unsafe_allow_html=True)
    
    # 2 TINH GỢI Ý
    st.markdown("<div class='box' style='margin-top:20px;'>🎯 2 TINH GỢI Ý (TỪ SỐ NÓNG)</div>", unsafe_allow_html=True)
    hot_nums = [d for d, c in res["hot_digits"]]
    c1, c2, c3 = st.columns(3)
    suggestions = []
    for i in range(min(3, len(hot_nums)-1)):
        for j in range(i+1, min(4, len(hot_nums))):
            suggestions.append(f"{hot_nums[i]},{hot_nums[j]}")
    
    for i, sug in enumerate(suggestions[:3]):
        with [c1, c2, c3][i]:
            st.markdown(f"<div class='item'>{sug}</div>", unsafe_allow_html=True)

# --- LỊCH SỬ ---
if st.session_state.history:
    st.divider()
    st.subheader("📋 Lịch Sử Đối Soát")
    df_history = pd.DataFrame(st.session_state.history).head(10)
    st.table(df_history)
    
    # Thống kê
    if len(st.session_state.history) >= 3:
        wins = sum(1 for h in st.session_state.history if "WIN" in h["KQ"])
        rate = wins / len(st.session_state.history) * 100
        st.markdown(f"<div style='text-align:center; padding:15px; background:#1a1a2e; border-radius:10px;'>"
                   f"📊 Tỷ Lệ Thắng: <span style='color:#00FFCC; font-size:24px; font-weight:bold;'>{wins}/{len(st.session_state.history)} ({rate:.1f}%)</span></div>", 
                   unsafe_allow_html=True)