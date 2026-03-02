# ================= IMPORT THƯ VIỆN =================
import streamlit as st
import google.generativeai as genai
import re
import json
import os
import pandas as pd
import numpy as np
from collections import Counter, defaultdict
from datetime import datetime
import time

# ================= CẤU HÌNH =================
try:
    API_KEY = st.secrets["GEMINI_API_KEY"]
except:
    st.error("⚠️ Chưa cấu hình API Key!")
    st.stop()

# ================= QUOTA MANAGEMENT =================
@st.cache_resource
def init_quota():
    return {'count': 0, 'last_reset': datetime.now().date(), 'limit': 15}

quota = init_quota()

def check_quota():
    today = datetime.now().date()
    if quota['last_reset'] != today:
        quota['count'] = 0
        quota['last_reset'] = today
    return quota['limit'] - quota['count']

def use_quota(n=1):
    quota['count'] += n

# ================= KHỞI TẠO AI =================
@st.cache_resource
def setup_neural():
    try:
        genai.configure(api_key=API_KEY)
        models = [m.name for m in genai.list_models() if 'generateContent' in m.supported_generation_methods]
        selected = next((m for m in ['models/gemini-1.5-flash', 'models/gemini-pro'] if m in models), None)
        return genai.GenerativeModel(selected) if selected else None, selected
    except:
        return None, None

neural_engine, model_used = setup_neural()

# ================= QUẢN LÝ DỮ LIỆU =================
def load_db():
    if "history" in st.session_state and st.session_state.history:
        return st.session_state.history
    if os.path.exists("titan_v33.json"):
        try:
            with open("titan_v33.json", "r") as f:
                return json.load(f)
        except:
            return []
    return []

def save_db(data):
    try:
        with open("titan_v33.json", "w") as f:
            json.dump(data[-3000:], f)
    except:
        pass

def clean_data(raw):
    matches = re.findall(r'\b\d{5}\b', raw.strip())
    return list(dict.fromkeys(matches))

# ================= 3 THUẬT TOÁN AI ĐỘC LẬP =================

def ai_statistical_analysis(history):
    """AI 1: Phân tích thống kê nâng cao"""
    if len(history) < 10:
        return {"main": "000", "support": "0000", "confidence": 50}
    
    all_d = "".join(history[-100:])
    freq = Counter(all_d)
    
    # Tần suất có trọng số (kỳ gần nặng hơn)
    weighted_freq = defaultdict(float)
    for i, num in enumerate(history[-50:]):
        weight = 1 + (i / 50)
        for d in num:
            weighted_freq[d] += weight
    
    # Top số nóng
    hot = [str(x[0]) for x in sorted(weighted_freq.items(), key=lambda k: k[1], reverse=True)[:7]]
    
    # Phân tích theo vị trí
    pos_freq = [defaultdict(float) for _ in range(5)]
    for num in history[-50:]:
        for i, d in enumerate(num[:5]):
            pos_freq[i][d] += 1
    
    # Chọn số xuất hiện nhiều nhất ở mỗi vị trí
    main_candidates = []
    for i in range(3):  # 3 số chính
        pos = i % 5
        if pos_freq[pos]:
            top = max(pos_freq[pos].items(), key=lambda k: k[1])[0]
            if top not in main_candidates:
                main_candidates.append(top)
    
    main_3 = "".join(main_candidates[:3]).ljust(3, '0')[:3]
    support_4 = "".join(hot[3:7]).ljust(4, '0')[:4]
    
    return {
        "main": main_3,
        "support": support_4,
        "confidence": min(85, 60 + len(history) // 10),
        "method": "Thống kê trọng số"
    }

def ai_pattern_recognition(history):
    """AI 2: Nhận diện pattern/cầu"""
    if len(history) < 15:
        return {"main": "000", "support": "0000", "confidence": 50}
    
    recent = history[-30:]
    patterns = {"bệt": [], "nhịp": [], "dao": []}
    
    # Phát hiện cầu bệt
    for pos in range(5):
        seq = [n[pos] if len(n) > pos else '0' for n in recent]
        for i in range(len(seq) - 2):
            if seq[i] == seq[i+1] == seq[i+2]:
                if seq[i] not in patterns["bệt"]:
                    patterns["bệt"].append(seq[i])
    
    # Phát hiện cầu nhịp 2
    for pos in range(5):
        seq = [n[pos] if len(n) > pos else '0' for n in recent]
        for i in range(len(seq) - 3):
            if seq[i] == seq[i+2] and seq[i] != seq[i+1]:
                if seq[i] not in patterns["nhịp"]:
                    patterns["nhịp"].append(seq[i])
    
    # Số về nhiều nhất 10 kỳ gần
    last_10 = "".join(history[-10:])
    hot_last = [str(x[0]) for x in Counter(last_10).most_common(7)]
    
    # Kết hợp pattern
    main_nums = list(dict.fromkeys(patterns["bệt"] + patterns["nhịp"]))[:3]
    while len(main_nums) < 3:
        for h in hot_last:
            if h not in main_nums:
                main_nums.append(h)
            if len(main_nums) >= 3:
                break
    
    support_nums = [h for h in hot_last if h not in main_nums][:4]
    while len(support_nums) < 4:
        support_nums.append('0')
    
    return {
        "main": "".join(main_nums[:3]),
        "support": "".join(support_nums[:4]),
        "confidence": 75 if patterns["bệt"] or patterns["nhịp"] else 65,
        "method": "Nhận diện cầu",
        "patterns": patterns
    }

def ai_gemini_analysis(history, risk_info):
    """AI 3: Gemini phân tích sâu"""
    if neural_engine is None or check_quota() <= 0:
        return None
    
    try:
        prompt = f"""
        TITAN v33.0 - Chuyên gia xổ số cao cấp.
        
        DATA:
        - 100 kỳ gần: {history[-100:] if len(history)>=100 else history}
        - Risk: {risk_info['score']}/100 - {risk_info['level']}
        - Warnings: {risk_info['warnings']}
        
        NHIỆM VỤ:
        1. Phân tích xu hướng tổng thể
        2. Chọn 3 số CHÍNH có xác suất cao nhất
        3. Chọn 4 số LÓT backup
        4. Decision: "ĐÁNH" nếu confidence >= 75, ngược lại "THEO DÕI"
        
        JSON: {{"main":"123","support":"4567","decision":"ĐÁNH","confidence":85,"logic":"Giải thích ngắn gọn"}}
        """
        
        response = neural_engine.generate_content(
            prompt,
            generation_config=genai.types.GenerationConfig(temperature=0.3, max_output_tokens=512)
        )
        
        text = response.text.strip()
        json_match = re.search(r'\{.*\}', text, re.DOTALL)
        
        if json_match:
            result = json.loads(json_match.group(0))
            result['method'] = "Gemini AI"
            return result
        
        return None
    except:
        return None

def multi_ai_consensus(history, risk_info):
    """✅ TỔNG HỢP 3 AI - RA QUYẾT TẬP THỂ"""
    results = []
    
    # AI 1: Statistical
    stat_result = ai_statistical_analysis(history)
    results.append(stat_result)
    
    # AI 2: Pattern
    pattern_result = ai_pattern_recognition(history)
    results.append(pattern_result)
    
    # AI 3: Gemini (nếu có quota)
    if check_quota() > 2:
        gemini_result = ai_gemini_analysis(history, risk_info)
        if gemini_result:
            results.append(gemini_result)
            use_quota(2)
    
    # ✅ BẦU CHỌN: Số xuất hiện nhiều nhất trong các đề xuất
    all_main = "".join([r['main'] for r in results])
    all_support = "".join([r['support'] for r in results])
    
    main_freq = Counter(all_main)
    support_freq = Counter(all_support)
    
    # Chọn 3 số chính (ưu tiên số được nhiều AI đề xuất)
    final_main = [str(x[0]) for x in main_freq.most_common(3)]
    while len(final_main) < 3:
        for d in '0123456789':
            if d not in final_main:
                final_main.append(d)
    
    # Chọn 4 số lót
    final_support = [str(x[0]) for x in support_freq.most_common(4)]
    while len(final_support) < 4:
        for d in '0123456789':
            if d not in final_support and d not in final_main:
                final_support.append(d)
    
    # Tính confidence trung bình
    avg_conf = sum(r['confidence'] for r in results) / len(results)
    
    # Quyết định dựa trên risk + consensus
    if risk_info['score'] >= 60:
        decision = "DỪNG"
        final_conf = 95
    elif risk_info['score'] >= 40:
        decision = "THEO DÕI"
        final_conf = min(avg_conf, 70)
    else:
        decision = "ĐÁNH" if avg_conf >= 75 else "THEO DÕI"
        final_conf = avg_conf
    
    return {
        "main_3": "".join(final_main[:3]),
        "support_4": "".join(final_support[:4]),
        "decision": decision,
        "confidence": round(final_conf),
        "logic": f"Multi-AI Consensus ({len(results)} models) | Risk: {risk_info['score']}/100",
        "color": "Green" if decision == "ĐÁNH" else "Red" if decision == "DỪNG" else "Yellow",
        "ai_details": results,
        "is_fallback": False
    }

# ================= RISK DETECTION =================
def detect_risk(history):
    if len(history) < 15:
        return {"score": 0, "warnings": [], "level": "OK"}
    
    recent = history[-20:]
    all_d = "".join(recent)
    freq = Counter(all_d)
    warnings = []
    score = 0
    
    # Số ra quá nhiều
    most = freq.most_common(1)
    if most and most[0][1] > 15:
        warnings.append(f"Số {most[0][0]} ra {most[0][1]}/20 kỳ")
        score += 30
    
    # Cầu bệt
    for pos in range(5):
        seq = [n[pos] if len(n)>pos else '0' for n in recent]
        streak = 1
        for i in range(1, len(seq)):
            if seq[i] == seq[i-1]:
                streak += 1
            else:
                streak = 1
        if streak >= 4:
            warnings.append(f"Vị {pos} bệt {streak}")
            score += 25
    
    level = "🔴 DỪNG" if score >= 60 else "🟡 CẨN THẬN" if score >= 40 else "🟢 OK"
    return {"score": score, "warnings": warnings, "level": level}

# ================= KHỞI TẠO SESSION =================
for key in ["history", "last_prediction", "last_clean", "auto_analyzed"]:
    if key not in st.session_state:
        st.session_state[key] = [] if key == "history" else None if key == "last_prediction" else None if key == "last_clean" else False

# ================= UI STYLE =================
st.set_page_config(page_title="TITAN v33.0 SPEED", layout="wide")
st.markdown("""
    <style>
    .stApp { background: #010409; color: #e6edf3; }
    .prediction-card { background: #0d1117; border: 1px solid #30363d; border-radius: 12px; padding: 20px; margin: 10px 0; }
    .num-box { font-size: 70px; font-weight: 900; color: #ff5858; text-align: center; letter-spacing: 10px; }
    .lot-box { font-size: 50px; font-weight: 700; color: #58a6ff; text-align: center; letter-spacing: 5px; }
    .status-bar { padding: 12px; border-radius: 8px; text-align: center; font-weight: bold; font-size: 18px; margin: 10px 0; }
    .quota-bar { background: #064e3b; border: 1px solid #10b981; padding: 8px; border-radius: 6px; text-align: center; margin: 10px 0; font-size: 14px; }
    .tab-btn { background: #1f2937; border: 1px solid #374151; padding: 10px 20px; border-radius: 8px; cursor: pointer; margin: 0 5px; }
    .tab-btn.active { background: #3b82f6; border-color: #60a5fa; }
    </style>
""", unsafe_allow_html=True)

st.markdown("<h2 style='text-align:center;color:#58a6ff'>⚡ TITAN v33.0 SPEED - MULTI-AI</h2>", unsafe_allow_html=True)

# Quota status
remaining = check_quota()
st.markdown(f"<div class='quota-bar'>🔌 Quota: <strong>{remaining}/{quota['limit']}</strong> | 🤖 Model: <code>{model_used.split('/')[-1] if model_used else 'Offline'}</code></div>", unsafe_allow_html=True)

# ================= TAB NAVIGATION =================
tab1, tab2, tab3 = st.columns([1, 1, 1])

with tab1:
    st.markdown("### 📥 NHẬP DỮ LIỆU")
    
    raw_input = st.text_area("📡 Dán kết quả (5 số/dòng):", height=100, placeholder="32880\n21808...", key="input_area")
    
    # ✅ AUTO-ANALYZE: Khi bấm nút, tự động lưu + phân tích
    if st.button("💾 THÊM & PHÂN TÍCH TỰ ĐỘNG", type="primary", use_container_width=True):
        if raw_input:
            with st.spinner("⚡ Đang xử lý..."):
                # 1. Làm sạch dữ liệu
                clean_nums = clean_data(raw_input)
                new_nums = [n for n in clean_nums if n not in (st.session_state.history or [])]
                
                if new_nums:
                    # 2. Lưu vào history
                    if not st.session_state.history:
                        st.session_state.history = []
                    st.session_state.history.extend(new_nums)
                    st.session_state.history = st.session_state.history[-3000:]
                    save_db(st.session_state.history)
                    
                    # 3. Tự động phân tích ngay
                    risk = detect_risk(st.session_state.history)
                    result = multi_ai_consensus(st.session_state.history, risk)
                    
                    st.session_state.last_prediction = result
                    st.session_state.last_clean = {
                        'found': len(clean_nums),
                        'new': len(new_nums),
                        'dup': len(clean_nums) - len(new_nums)
                    }
                    
                    st.success(f"✅ Đã thêm {len(new_nums)} kỳ & phân tích xong!")
                    st.rerun()
                else:
                    st.warning("⚠️ Số đã có trong DB!")
        else:
            st.error("❌ Vui lòng nhập dữ liệu!")
    
    # Hiển thị kết quả làm sạch
    if st.session_state.last_clean:
        c1, c2, c3 = st.columns(3)
        d = st.session_state.last_clean
        c1.metric("🔍 Tìm thấy", d['found'])
        c2.metric("➕ Mới", d['new'])
        c3.metric("🗑️ Trùng", d['dup'])

with tab2:
    st.markdown("### 🎯 KẾT QUẢ DỰ ĐOÁN")
    
    if st.session_state.last_prediction:
        res = st.session_state.last_prediction
        
        # Risk warning
        if 'risk' in res or (isinstance(res, dict) and 'logic' in res and 'Risk' in res.get('logic', '')):
            # Extract risk from logic if not in dict
            risk_text = res.get('logic', '')
            if 'Risk:' in risk_text:
                risk_part = risk_text.split('Risk:')[1].strip()
                risk_score = int(risk_part.split('/')[0]) if '/' in risk_part else 0
                
                if risk_score >= 60:
                    st.markdown("<div class='status-bar' style='background:#da3633'>🚨 RỦI RO CAO - DỪNG CHƠI</div>", unsafe_allow_html=True)
                elif risk_score >= 40:
                    st.markdown("<div class='status-bar' style='background:#d29922'>⚠️ RỦI RO TB - CẨN THẬN</div>", unsafe_allow_html=True)
        
        # Status bar
        color = res.get('color', 'green')
        bg = "#238636" if color == "Green" else "#da3633" if color == "Red" else "#d29922"
        st.markdown(f"<div class='status-bar' style='background:{bg}'>{res.get('decision','')} | Độ tin: {res.get('confidence',0)}%</div>", unsafe_allow_html=True)

        # Numbers display
        st.markdown("<div class='prediction-card'>", unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("<p style='text-align:center;color:#8b949e'>🔥 3 SỐ CHÍNH</p>", unsafe_allow_html=True)
            st.markdown(f"<div class='num-box'>{res.get('main_3','000')}</div>", unsafe_allow_html=True)
        with c2:
            st.markdown("<p style='text-align:center;color:#8b949e'>🛡️ 4 SỐ LÓT</p>", unsafe_allow_html=True)
            st.markdown(f"<div class='lot-box'>{res.get('support_4','0000')}</div>", unsafe_allow_html=True)
        
        st.divider()
        st.write(f"💡 **Logic:** {res.get('logic','')}")
        
        # AI details
        if 'ai_details' in res:
            with st.expander("🤖 Chi tiết phân tích từ các AI"):
                for i, ai in enumerate(res['ai_details'], 1):
                    st.write(f"**AI {i} - {ai.get('method','')}**: Main={ai.get('main','')}, Support={ai.get('support','')}, Conf={ai.get('confidence',0)}%")
        
        # Copy dàn
        full_dan = "".join(sorted(set(res.get('main_3','') + res.get('support_4',''))))
        st.text_input("📋 Dàn 7 số:", full_dan)
        st.markdown("</div>", unsafe_allow_html=True)
    else:
        st.info("👈 Chưa có dữ liệu. Vào tab 'Nhập dữ liệu' để bắt đầu!")

with tab3:
    st.markdown("### 📊 THỐNG KÊ & DATABASE")
    
    st.write(f"**📈 Tổng kỳ:** {len(st.session_state.history or [])}")
    
    if st.session_state.history:
        # Download/Upload
        col1, col2 = st.columns(2)
        with col1:
            json_data = json.dumps(st.session_state.history[-3000:], ensure_ascii=False).encode('utf-8')
            st.download_button("💾 Tải DB", json_data, f"titan_{datetime.now().strftime('%Y%m%d')}.json", "application/json")
        with col2:
            uploaded = st.file_uploader("📂 Nạp DB", type="json")
            if uploaded:
                try:
                    st.session_state.history = json.load(uploaded)
                    st.success(f"✅ Đã nạp {len(st.session_state.history)} kỳ")
                    st.rerun()
                except:
                    st.error("❌ File không hợp lệ")
        
        st.divider()
        
        # Stats
        with st.expander("📊 Thống kê tần suất (50 kỳ gần)", expanded=True):
            all_d = "".join(st.session_state.history[-50:])
            freq = Counter(all_d)
            
            col1, col2 = st.columns(2)
            with col1:
                st.write("##### Biểu đồ")
                st.bar_chart(pd.Series(freq).sort_index(), color="#58a6ff")
            with col2:
                st.write("##### Top số nóng")
                for num, count in freq.most_common(5):
                    st.write(f"- Số **{num}**: {count} lần")
        
        # Recent results
        with st.expander("📜 Lịch sử 20 kỳ gần"):
            recent = st.session_state.history[-20:][::-1]
            for i, num in enumerate(recent, 1):
                st.write(f"{i}. **{num}**")
        
        # Clear button
        if st.button("🗑️ Xóa toàn bộ dữ liệu"):
            st.session_state.history = []
            st.session_state.last_prediction = None
            if os.path.exists("titan_v33.json"):
                os.remove("titan_v33.json")
            st.rerun()

# ================= FOOTER =================
st.markdown("---")
st.caption(f"""
⚡ **TITAN v33.0 SPEED** | Multi-AI Consensus (Statistical + Pattern + Gemini)  
🔌 Quota: ~15 request/ngày | Tự động phân tích khi thêm dữ liệu  
📊 Tab layout: Nhập liệu → Kết quả → Thống kê (Không cần scroll)
""")
st.caption(f"🕐 {datetime.now().strftime('%d/%m/%Y %H:%M')}")