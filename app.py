streamlit>=1.30.0
google-generativeai>=0.8.0
pandas>=2.0.0
numpy>=1.24.0

➡➡➡➡➡➡➡➡➡➡➡➡➡➡➡➡➡➡➡

# ================= IMPORT THƯ VIỆN =================
import streamlit as st
import google.generativeai as genai
import re
import json
import pandas as pd
import numpy as np
from collections import Counter, defaultdict
from datetime import datetime
import time

# ================= CẤU HÌNH =================
try:
    API_KEY = st.secrets["GEMINI_API_KEY"]
except:
    st.error("⚠️ Chưa cấu hình API Key trong Secrets!")
    st.stop()

# ================= UI STYLE GỐC CỦA ANH =================
st.set_page_config(page_title="TITAN v31.0 - 3 SỐ 5 TINH", layout="wide")
st.markdown("""
    <style>
    .stApp { background: #010409; color: #e6edf3; }
    .prediction-card {
        background: #0d1117; border: 1px solid #30363d;
        border-radius: 12px; padding: 25px; margin-top: 20px;
        box-shadow: 0 4px 20px rgba(0,0,0,0.5);
    }
    .num-box {
        font-size: 80px; font-weight: 900; color: #ff5858;
        text-align: center; letter-spacing: 12px; border-right: 2px solid #30363d;
        text-shadow: 0 0 15px rgba(255,88,88,0.4);
    }
    .lot-box {
        font-size: 55px; font-weight: 700; color: #58a6ff;
        text-align: center; letter-spacing: 8px; padding-left: 20px;
    }
    .status-bar { padding: 15px; border-radius: 10px; text-align: center; font-weight: bold; font-size: 20px; margin-bottom: 15px; }
    .warning-box { background: #331010; color: #ff7b72; padding: 10px; border-radius: 5px; border: 1px solid #6e2121; text-align: center; margin-top: 10px; }
    .quota-ok { background: #064e3b; border: 1px solid #10b981; padding: 10px; border-radius: 8px; margin: 10px 0; text-align: center; }
    .quota-warn { background: #422006; border: 1px solid #9a6700; padding: 10px; border-radius: 8px; margin: 10px 0; text-align: center; }
    .rule-box { background: #1f2937; border-left: 4px solid #3b82f6; padding: 15px; margin: 15px 0; border-radius: 8px; }
    .win-check { background: #064e3b; border: 1px solid #10b981; padding: 15px; border-radius: 8px; margin: 10px 0; }
    .lose-check { background: #7c2d12; border: 1px solid #fbbf24; padding: 15px; border-radius: 8px; margin: 10px 0; }
    </style>
""", unsafe_allow_html=True)

st.markdown("<h1 style='text-align: center; color: #58a6ff;'>🎯 TITAN v31.0 - 3 SỐ 5 TINH</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: #8b949e;'>Chọn 3 số • Trúng khi đủ 3 số trong kết quả 5 chữ số • Không cần đúng vị trí</p>", unsafe_allow_html=True)

# ================= QUẢN LÝ QUOTA API =================
@st.cache_resource
def init_quota():
    return {'count': 0, 'last_reset': datetime.now().date(), 'limit': 18}
quota = init_quota()

def check_quota():
    today = datetime.now().date()
    if quota['last_reset'] != today:
        quota['count'] = 0
        quota['last_reset'] = today
    return quota['limit'] - quota['count']

def use_quota():
    quota['count'] += 1

# ================= KHỞI TẠO AI =================
@st.cache_resource
def setup_neural():
    try:
        genai.configure(api_key=API_KEY)
        models = [m.name for m in genai.list_models() if 'generateContent' in m.supported_generation_methods]
        preferred = ['models/gemini-1.5-flash', 'models/gemini-pro', 'models/gemini-1.0-pro']
        selected = next((m for m in preferred if m in models), models[0] if models else None)
        return genai.GenerativeModel(selected) if selected else None, selected
    except:
        return None, None

neural_engine, model_used = setup_neural()

# ================= QUẢN LÝ DỮ LIỆU =================
def load_db(uploaded_file):
    if uploaded_file:
        try:
            data = json.load(uploaded_file)
            return data if isinstance(data, list) else []
        except:
            return []
    return []

def save_db_json(data):
    return json.dumps(data[-3000:], ensure_ascii=False).encode('utf-8')

def clean_input(raw_text, existing):
    """Làm sạch: chỉ giữ số 5 chữ số, loại trùng, báo số bị loại"""
    cleaned = re.sub(r'[\s\t]+', ' ', raw_text.strip())
    matches = re.findall(r'\b\d{5}\b', cleaned)
    unique = list(dict.fromkeys(matches))
    new_nums = [n for n in unique if n not in existing]
    rejected = [m for m in re.findall(r'\b\d{3,7}\b', cleaned) if len(re.sub(r'\D','',m)) != 5]
    return {
        'found': len(matches), 'unique': len(unique), 'new': len(new_nums),
        'dup': len(matches)-len(unique), 'rejected': list(set(rejected))[:5],
        'numbers': new_nums
    }

# Session state
for key in ["history", "last_prediction", "show_debug", "last_clean", "check_result"]:
    if key not in st.session_state:
        st.session_state[key] = None if key == "last_prediction" else ([] if key == "history" else False)

# ================= THUẬT TOÁN PHÁT HIỆN RỦI RO =================
def detect_risk(history, window=20):
    if len(history) < window:
        return {"score": 0, "warnings": [], "level": "OK"}
    recent = history[-window:]
    all_d = "".join(recent)
    freq = Counter(all_d)
    warnings = []
    score = 0
    
    # Số ra quá nhiều
    most = freq.most_common(1)
    if most and most[0][1] > 15:
        warnings.append(f"Số {most[0][0]} ra {most[0][1]}/{window} kỳ")
        score += 30
    
    # Cầu bệt vị trí
    for pos in range(5):
        seq = [int(n[pos]) if len(n)>pos else 0 for n in recent]
        streak = max_streak = 1
        for i in range(1, len(seq)):
            if seq[i] == seq[i-1]:
                streak += 1
                max_streak = max(max_streak, streak)
            else:
                streak = 1
        if max_streak >= 4:
            warnings.append(f"Vị {pos} bệt {max_streak}")
            score += 25
    
    # Entropy thấp = quá đều
    total = len(all_d)
    if total > 0:
        entropy = -sum((c/total)*np.log2(c/total) for c in freq.values() if c>0)
        if entropy < 2.8:
            warnings.append(f"Pattern quá đều (entropy={entropy:.2f})")
            score += 25
    
    level = "🔴 DỪNG" if score >= 60 else "🟡 CẨN THẬN" if score >= 40 else "🟢 OK"
    return {"score": score, "warnings": warnings, "level": level}

# ================= KIỂM TRA TRÚNG THƯỞNG (LUẬT 3 SỐ 5 TINH) =================
def check_win(selected_3, result_5digit):
    """
    ✅ Kiểm tra trúng theo luật 3 số 5 tinh:
    - selected_3: list 3 số [1,2,6]
    - result_5digit: string "12864"
    - Return: True nếu result chứa đủ cả 3 số (không cần đúng vị trí)
    """
    if len(selected_3) != 3 or len(result_5digit) != 5:
        return False
    result_digits = set(result_5digit)
    return all(str(s) in result_digits for s in selected_3)

def analyze_recent_wins(history, predictions_log, window=20):
    """Phân tích tỷ lệ trúng của các dự đoán gần đây"""
    if not history or not predictions_log:
        return None
    recent = history[-window:]
    results = []
    for pred in predictions_log[-window:]:
        if 'main_3' in pred and 'result' in pred:
            selected = [int(d) for d in str(pred['main_3']) if d.isdigit()]
            if len(selected) == 3:
                won = check_win(selected, pred['result'])
                results.append(won)
    if results:
        return {"total": len(results), "won": sum(results), "rate": round(sum(results)/len(results)*100, 1)}
    return None

# ================= DỰ ĐOÁN FALLBACK (KHÔNG CẦN AI) =================
def fallback_predict(history):
    """Dự đoán bằng thống kê khi AI lỗi/hết quota"""
    all_d = "".join(history[-50:] if len(history)>=50 else history)
    freq = Counter(all_d)
    # Chọn 3 số có tần suất cao nhất, đảm bảo khác nhau
    top = [int(x[0]) for x in freq.most_common(10)]
    main_3 = list(dict.fromkeys(top))[:3]  # Loại trùng, lấy 3 số đầu
    support = [str(x) for x in list(dict.fromkeys(top))[3:7]]
    return {
        "main_3": "".join(str(x) for x in main_3),
        "support_4": "".join(support) if len(support)>=4 else "".join(support)+"".join(str(x) for x in top[7:11])[:4-len(support)],
        "decision": "THEO DÕI",
        "confidence": 70,
        "logic": "Thống kê tần suất thuần túy",
        "is_fallback": True
    }

# ================= SIDEBAR =================
with st.sidebar:
    st.header("⚙️ Cài đặt")
    st.session_state.show_debug = st.checkbox("🐛 Debug", value=False)
    
    st.divider()
    uploaded = st.file_uploader("📂 Nạp DB", type="json")
    if uploaded:
        st.session_state.history = load_db(uploaded)
        st.success(f"✅ {len(st.session_state.history)} kỳ")
    
    st.divider()
    if st.session_state.history:
        st.download_button("💾 Tải DB", save_db_json(st.session_state.history), 
                          f"titan_{datetime.now().strftime('%Y%m%d')}.json", "application/json")
    
    st.divider()
    st.metric("📊 Tổng kỳ", len(st.session_state.history or []))
    st.metric("🔌 API calls", quota['count'])
    
    if st.button("🗑️ Xóa DB"):
        st.session_state.history = []
        st.session_state.last_prediction = None
        st.rerun()
    
    st.divider()
    if model_used:
        st.success(f"✅ {model_used.split('/')[-1]}")
    else:
        st.error("❌ AI lỗi")

# ================= HIỂN THỊ LUẬT CHƠI =================
st.markdown("""
<div class='rule-box'>
<b>📋 Luật 3 số 5 tinh:</b><br>
• Chọn 3 số từ 0-9 (ví dụ: 1, 2, 6)<br>
• Kết quả là số 5 chữ số: [Chục ngàn][Ngàn][Trăm][Chục][Đơn vị]<br>
• ✅ <b>Trúng:</b> Kết quả chứa <u>đủ cả 3 số</u> đã chọn (không cần đúng vị trí)<br>
• ❌ <b>Trượt:</b> Thiếu dù 1 số trong 3 số đã chọn<br>
• Ví dụ: Chọn <code>1,2,6</code> → Kết quả <code>12864</code> = ✅ Trúng (có 1,2,6)<br>
• Ví dụ: Chọn <code>1,3,6</code> → Kết quả <code>12662</code> = ❌ Trượt (thiếu 3)
</div>
""", unsafe_allow_html=True)

# ================= QUOTA STATUS =================
remaining = check_quota()
if remaining > 0:
    st.markdown(f"<div class='quota-ok'>✅ Quota: <strong>{remaining}/{quota['limit']}</strong> request còn lại</div>", unsafe_allow_html=True)
else:
    st.markdown("<div class='quota-warn'>⚠️ <strong>HẾT QUOTA</strong> → Dùng fallback thống kê</div>", unsafe_allow_html=True)

# ================= NHẬP LIỆU =================
col1, col2 = st.columns([3, 1])
with col1:
    raw = st.text_area("📡 Dán kết quả (5 số/dòng)", height=120, placeholder="32880\n21808...")
with col2:
    st.metric("Kỳ trong DB", len(st.session_state.history or []))
    if st.button("🔍 Xem trước", use_container_width=True) and raw:
        st.session_state.last_clean = clean_input(raw, st.session_state.history or [])
    if st.button("🚀 Lưu & Xử lý", type="primary", use_container_width=True) and raw:
        cleaned = clean_input(raw, st.session_state.history or [])
        st.session_state.last_clean = cleaned
        if cleaned['new'] > 0:
            if not st.session_state.history:
                st.session_state.history = []
            st.session_state.history.extend(cleaned['numbers'])
            st.session_state.history = (st.session_state.history or [])[-3000:]
            st.success(f"✅ Lưu {cleaned['new']} kỳ mới!")
            st.info(f"📊 Tìm: {cleaned['found']} | Riêng: {cleaned['unique']} | Trùng: {cleaned['dup']}")
            st.rerun()
        else:
            st.warning("⚠️ Số đã có trong DB!")

# Hiển thị kết quả làm sạch
if st.session_state.last_clean:
    c1,c2,c3,c4 = st.columns(4)
    d = st.session_state.last_clean
    c1.metric("🔍 Tìm thấy", d['found'])
    c2.metric("✅ Riêng", d['unique'])
    c3.metric("➕ Mới", d['new'])
    c4.metric("🗑️ Trùng", d['dup'])
    if d['rejected']:
        with st.expander("🚫 Số bị loại"):
            for r in d['rejected']:
                st.write(f"- `{r}`")

# ================= PHÂN TÍCH & DỰ ĐOÁN =================
st.markdown("---")
st.subheader("🔮 DỰ ĐOÁN 3 SỐ 5 TINH")

if (st.session_state.history or []) and len(st.session_state.history) >= 15:
    if st.button("🎯 CHẠY DỰ ĐOÁN", type="secondary", use_container_width=True):
        risk = detect_risk(st.session_state.history)
        
        # Nếu risk cao → không gọi AI
        if risk['score'] >= 60:
            st.session_state.last_prediction = {
                "main_3": "000", "support_4": "0000", "decision": "DỪNG",
                "confidence": 99, "logic": "Risk cao - Pattern bất thường",
                "risk": risk, "is_fallback": True
            }
            st.warning("⚠️ Risk cao → Không dự đoán")
            st.rerun()
        
        # Kiểm tra quota
        if check_quota() <= 0 or neural_engine is None:
            st.warning("⚠️ Hết quota/AI lỗi → Dùng fallback")
            result = fallback_predict(st.session_state.history)
            result['risk'] = risk
            st.session_state.last_prediction = result
            st.rerun()
        
        # Gọi AI
        with st.spinner("🤖 Titan đang phân tích..."):
            prompt = f"""TITAN v31.0 - Chuyên gia 3 số 5 tinh.
LUẬT: Chọn 3 số 0-9. Trúng nếu kết quả 5 chữ số chứa ĐỦ cả 3 số (không cần đúng vị trí).
DATA: Lịch sử 50 kỳ: {(st.session_state.history or [])[-50:]}
Risk: {risk['score']}/100 - {risk['level']} | Warnings: {risk['warnings']}
NHIỆM VỤ:
1. Nếu Risk>=60: decision="DỪNG"
2. Nếu Risk<60: Chọn 3 số CHÍNH (main_3) có xác suất cao nhất sẽ XUẤT HIỆN trong kết quả
3. Chọn 4 số LÓT (support_4) để backup
4. decision: "ĐÁNH" hoặc "THEO DÕI"
5. confidence: 0-100
JSON: {{"main_3":"123","support_4":"4567","decision":"ĐÁNH","confidence":85,"logic":"Giải thích ngắn"}}
LƯU Ý: main_3 phải là 3 chữ số KHÁC NHAU từ 0-9."""
            
            try:
                response = neural_engine.generate_content(
                    prompt,
                    generation_config=genai.types.GenerationConfig(temperature=0.2, top_p=0.9, max_output_tokens=512)
                )
                text = response.text.strip()
                json_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', text, re.DOTALL)
                if not json_match:
                    json_match = re.search(r'\{.*\}', text, re.DOTALL)
                
                if json_match:
                    result = json.loads(json_match.group(1))
                    # Validate & chuẩn hóa main_3
                    main = str(result.get('main_3',''))
                    digits = [d for d in main if d.isdigit()]
                    unique_digits = list(dict.fromkeys(digits))[:3]  # Lấy 3 số khác nhau đầu tiên
                    if len(unique_digits) < 3:
                        # Bổ sung từ support hoặc thống kê
                        all_d = "".join((st.session_state.history or [])[-50:])
                        freq = Counter(all_d)
                        for d in [str(x[0]) for x in freq.most_common(10)]:
                            if d not in unique_digits:
                                unique_digits.append(d)
                            if len(unique_digits) >= 3:
                                break
                    result['main_3'] = "".join(unique_digits[:3])
                    result['support_4'] = str(result.get('support_4',''))[:4].zfill(4)
                    result['risk'] = risk
                    result['is_fallback'] = False
                    st.session_state.last_prediction = result
                    use_quota()
                    st.success(f"✅ OK! Quota còn: {check_quota()}")
                    st.rerun()
                raise ValueError("No JSON")
            except Exception as e:
                err = str(e)
                if '429' in err or 'quota' in err.lower():
                    st.warning("⚠️ Hết quota → Fallback")
                    result = fallback_predict(st.session_state.history)
                    result['risk'] = risk
                    result['quota_error'] = True
                    st.session_state.last_prediction = result
                    st.rerun()
                st.error(f"❌ Lỗi: {err[:150]}")
                result = fallback_predict(st.session_state.history)
                result['risk'] = risk
                st.session_state.last_prediction = result
                st.info("⚠️ Dùng fallback")

elif st.session_state.history:
    st.info(f"💡 Cần ≥15 kỳ (có {len(st.session_state.history)})")

# ================= HIỂN THỊ KẾT QUẢ =================
if st.session_state.last_prediction:
    res = st.session_state.last_prediction
    risk = res.get('risk', {})
    risk_score = risk.get('score', 0)
    
    st.markdown("---")
    
    # Hiển thị risk
    if risk_score >= 60:
        st.markdown(f"<div class='status-bar' style='background:#da3633'>🚨 RỦI RO: {risk_score}/100 - {risk.get('level')}<br>{' | '.join(risk.get('warnings',[]))}</div>", unsafe_allow_html=True)
        st.error("**KHUYẾN NGHỊ: DỪNG CHƠI!** Pattern bất thường.")
    elif risk_score >= 40:
        st.markdown(f"<div class='status-bar' style='background:#d29922'>⚠️ RỦI RO: {risk_score}/100 - {risk.get('level')}</div>", unsafe_allow_html=True)
        st.warning("**CẨN THẬN** - Đánh nhỏ.")
    else:
        st.markdown(f"<div class='status-bar' style='background:#238636'>✅ RỦI RO: {risk_score}/100 - {risk.get('level')}</div>", unsafe_allow_html=True)
    
    if risk_score < 60:
        decision = res.get('decision','N/A')
        conf = res.get('confidence',0)
        is_fb = res.get('is_fallback', False)
        
        colors = {"ĐÁNH":("#238636","✅"),"THEO DÕI":("#d29922","⏳"),"DỪNG":("#da3633","🛑")}
        bg, icon = colors.get(decision, ("#30363d","❓"))
        badge = "🔄 FALLBACK" if is_fb else "🤖 AI"
        
        st.markdown(f"""
            <div class='status-bar' style='background:{bg};color:white'>
                {icon} {decision} | Độ tin: {conf}% | {badge}
            </div>
        """, unsafe_allow_html=True)

        st.markdown("<div class='prediction-card'>", unsafe_allow_html=True)
        
        # Hiển thị 3 số chính + 4 số lót
        c1, c2 = st.columns([1.5, 1])
        with c1:
            st.markdown("<p style='color:#8b949e;text-align:center'>🔥 3 SỐ CHÍNH (Chọn để cược)</p>", unsafe_allow_html=True)
            main_nums = list(res.get('main_3','???'))
            st.markdown(f"""
                <div style='display:flex;justify-content:center;gap:15px'>
                    <div style='font-size:70px;font-weight:900;color:#ff5858'>{main_nums[0] if len(main_nums)>0 else '?'}</div>
                    <div style='font-size:70px;font-weight:900;color:#ff5858'>{main_nums[1] if len(main_nums)>1 else '?'}</div>
                    <div style='font-size:70px;font-weight:900;color:#ff5858'>{main_nums[2] if len(main_nums)>2 else '?'}</div>
                </div>
            """, unsafe_allow_html=True)
            st.markdown("<p style='text-align:center;color:#8b949e;font-size:14px'>✅ Trúng nếu kết quả 5 chữ số chứa đủ cả 3 số này</p>", unsafe_allow_html=True)
        with c2:
            st.markdown("<p style='color:#8b949e;text-align:center'>🛡️ 4 SỐ LÓT (Backup)</p>", unsafe_allow_html=True)
            st.markdown(f"<div class='lot-box'>{res.get('support_4','????')}</div>", unsafe_allow_html=True)
        
        st.divider()
        
        # Logic + hướng dẫn
        col_log, col_copy = st.columns([2,1])
        with col_log:
            st.write(f"💡 **Logic:** {res.get('logic','N/A')}")
            if conf < 75:
                st.markdown("<div class='warning-box'>⚠️ Độ tin thấp - Nên đánh nhỏ hoặc chờ</div>", unsafe_allow_html=True)
        with col_copy:
            # Dàn 7 số để tham khảo
            all_nums = "".join(sorted(set(res.get('main_3','')+res.get('support_4',''))))
            st.text_input("📋 Dàn 7 số:", all_nums)
            st.markdown("<p style='font-size:12px;color:#8b949e'>Copy dàn này để chơi bao lô nếu muốn</p>", unsafe_allow_html=True)
        
        st.markdown("</div>", unsafe_allow_html=True)
        
        # ✅ TÍNH NĂNG MỚI: KIỂM TRA TRÚNG VỚI KẾT QUẢ THỰC
        st.markdown("### ✅ KIỂM TRA TRÚNG THƯỞNG")
        st.markdown("<div class='rule-box'>Nhập kết quả thực tế để kiểm tra xem 3 số chính có trúng không theo luật 3 số 5 tinh</div>", unsafe_allow_html=True)
        
        col_chk1, col_chk2 = st.columns([2,1])
        with col_chk1:
            check_input = st.text_input("🎲 Nhập kết quả 5 chữ số để kiểm tra:", placeholder="Ví dụ: 12864", max_chars=5)
        with col_chk2:
            if st.button("🔍 Kiểm tra trúng", use_container_width=True) and check_input and len(check_input)==5 and check_input.isdigit():
                main_digits = [int(d) for d in res.get('main_3','') if d.isdigit()]
                if len(main_digits) == 3:
                    won = check_win(main_digits, check_input)
                    st.session_state.check_result = {"input": check_input, "selected": main_digits, "won": won}
                else:
                    st.error("❌ 3 số chính không hợp lệ")
        
        if st.session_state.check_result:
            cr = st.session_state.check_result
            if cr['won']:
                st.markdown(f"""
                    <div class='win-check'>
                    🎉 <b>TRÚNG THƯỞNG!</b><br>
                    • Bạn chọn: <code>{cr['selected']}</code><br>
                    • Kết quả: <code>{cr['input']}</code><br>
                    • Kết quả chứa đủ cả 3 số: {', '.join(str(s) for s in cr['selected'])} → ✅
                    </div>
                """, unsafe_allow_html=True)
            else:
                missing = [s for s in cr['selected'] if str(s) not in cr['input']]
                st.markdown(f"""
                    <div class='lose-check'>
                    ❌ <b>KHÔNG TRÚNG</b><br>
                    • Bạn chọn: <code>{cr['selected']}</code><br>
                    • Kết quả: <code>{cr['input']}</code><br>
                    • Thiếu số: {', '.join(str(m) for m in missing)} → ❌
                    </div>
                """, unsafe_allow_html=True)

# ================= BIỂU ĐỒ & THỐNG KÊ =================
st.markdown("---")
with st.expander("📊 Thống kê 50 kỳ gần"):
    if st.session_state.history:
        col1, col2 = st.columns(2)
        with col1:
            st.write("##### Tần suất số (0-9)")
            all_d = "".join((st.session_state.history or [])[-50:])
            st.bar_chart(pd.Series(Counter(all_d)).sort_index(), color="#58a6ff")
        with col2:
            st.write("##### Top số nóng")
            freq = Counter(all_d).most_common(5)
            for num, count in freq:
                st.write(f"- Số <b>{num}</b>: {count} lần")

# ================= FOOTER =================
st.markdown("---")
st.caption("""
⚠️ **LƯU Ý:** Tool tham khảo. Nhà cái có thể điều khiển kết quả. Quản lý vốn chặt, dừng đúng lúc!
🎯 **Luật 3 số 5 tinh:** Trúng khi kết quả 5 chữ số chứa ĐỦ cả 3 số đã chọn (không cần đúng vị trí)
🔌 **Quota API:** ~20 request/ngày (free tier). Tool tự chuyển fallback khi hết.
🔐 **Bảo mật:** API Key lưu trong Secrets, không lộ trên GitHub.
""")
st.caption(f"🕐 {datetime.now().strftime('%d/%m/%Y %H:%M')} | TITAN v31.0 - 3 SỐ 5 TINH")

