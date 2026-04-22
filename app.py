# =============================================================================
# 💎 AI-QUANTUM PRO 2026 - GEMINI AI ENHANCED
# =============================================================================
import streamlit as st
import requests
from bs4 import BeautifulSoup
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
from streamlit_autorefresh import st_autorefresh
from collections import Counter
import json
import time

# Gemini AI
try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except:
    GEMINI_AVAILABLE = False

# =============================================================================
# 🔧 CẤU HÌNH
# =============================================================================
st.set_page_config(
    page_title="💎 AI-QUANTUM PRO 2026 - Gemini Enhanced",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

st_autorefresh(interval=120000, key="live_update", limit=None)

# =============================================================================
# 🎨 CSS
# =============================================================================
st.markdown("""
<style>
    .main { background-color: #0a0a0f; color: #ffffff; }
    .header-gold {
        background: linear-gradient(135deg, #D4AF37 0%, #FFD700 50%, #B8962E 100%);
        padding: 20px; border-radius: 15px; text-align: center; 
        color: #000; font-weight: bold; margin-bottom: 20px;
    }
    .stat-card {
        background: linear-gradient(135deg, #1a1a2e, #16213e);
        border: 2px solid #D4AF37; border-radius: 12px;
        padding: 15px; text-align: center; margin: 5px;
    }
    .stat-value { font-size: 28px; font-weight: bold; color: #FFD700; }
    .stat-label { font-size: 12px; color: #888; margin-top: 5px; }
    .bet-number {
        background: linear-gradient(135deg, #ff4b4b, #ff6b6b);
        color: white; padding: 10px 20px; border-radius: 8px;
        font-size: 24px; font-weight: bold; display: inline-block;
        margin: 5px; box-shadow: 0 4px 15px rgba(255, 75, 75, 0.4);
    }
    .cold-number {
        background: linear-gradient(135deg, #4a90e2, #67b26f);
        color: white; padding: 8px 15px; border-radius: 8px;
        font-size: 18px; font-weight: bold; display: inline-block;
        margin: 3px;
    }
    .pred-box {
        border: 2px solid #D4AF37; border-radius: 12px;
        padding: 20px; background: linear-gradient(135deg, #111, #1a1a2e);
        text-align: center; margin: 8px 0;
    }
    .gemini-badge {
        background: linear-gradient(135deg, #4285f4, #34a853);
        color: white; padding: 5px 15px; border-radius: 20px;
        font-size: 12px; font-weight: bold; display: inline-block;
        margin: 5px;
    }
    .disclaimer {
        background: rgba(255, 107, 107, 0.15);
        border-left: 4px solid #ff6b6b;
        padding: 12px; border-radius: 0 8px 8px 0;
        margin: 15px 0; font-size: 13px;
    }
</style>
""", unsafe_allow_html=True)

# =============================================================================
# 💾 STATISTICS & HISTORY
# =============================================================================
def init_data():
    if 'statistics' not in st.session_state:
        st.session_state.statistics = {
            'predictions': [],
            'total_predictions': 0,
            'wins': 0,
            'losses': 0,
            'win_rate': 0.0
        }
    if 'historical_results' not in st.session_state:
        st.session_state.historical_results = []

def save_historical_result(data):
    """Lưu kết quả vào lịch sử để phân tích"""
    if data and data.get("Đặc Biệt"):
        record = {
            'date': datetime.now().strftime("%d/%m/%Y"),
            'time': datetime.now().strftime("%H:%M"),
            'db': data["Đặc Biệt"],
            'g1': data.get("Giải Nhất", ""),
            'all_numbers': extract_all_numbers(data)
        }
        # Tránh trùng lặp
        if not any(r['date'] == record['date'] and r['time'] == record['time'] 
                   for r in st.session_state.historical_results):
            st.session_state.historical_results.append(record)
            # Giữ 30 ngày gần nhất
            if len(st.session_state.historical_results) > 30:
                st.session_state.historical_results.pop(0)

def extract_all_numbers(data):
    """Trích xuất tất cả số 2 chữ số từ kết quả"""
    numbers = []
    for key, value in data.items():
        if key == "time":
            continue
        if isinstance(value, list):
            for num in value:
                if num and num != "...":
                    numbers.append(num[-2:])
        elif value and value != "...":
            numbers.append(value[-2:])
    return numbers

def analyze_so_bet(historical_data, days=7):
    """
    Phân tích số bệt - số lâu chưa về
    Returns: Dict với số và số ngày chưa về
    """
    all_possible = [f"{i:02d}" for i in range(100)]
    
    # Lấy kết quả những ngày gần nhất
    recent_results = historical_data[-days:] if len(historical_data) >= days else historical_data
    
    # Tập hợp tất cả số đã về
    appeared_numbers = set()
    for record in recent_results:
        appeared_numbers.update(record.get('all_numbers', []))
    
    # Số chưa về (số bệt/gan)
    cold_numbers = [num for num in all_possible if num not in appeared_numbers]
    
    # Tính số ngày chưa về cho từng số
    days_missing = {}
    for num in all_possible:
        days_count = 0
        for record in reversed(recent_results):
            if num in record.get('all_numbers', []):
                break
            days_count += 1
        if days_count > 0:
            days_missing[num] = days_count
    
    # Sắp xếp theo số ngày giảm dần
    coldest = sorted(days_missing.items(), key=lambda x: x[1], reverse=True)
    
    return {
        'cold_numbers': cold_numbers[:20],  # 20 số lâu chưa về nhất
        'total_cold': len(cold_numbers),
        'appeared_count': len(appeared_numbers)
    }

def analyze_frequency(historical_data):
    """Phân tích tần suất xuất hiện"""
    all_numbers = []
    for record in historical_data:
        all_numbers.extend(record.get('all_numbers', []))
    
    counter = Counter(all_numbers)
    hot_numbers = counter.most_common(10)
    
    return {
        'hot_numbers': hot_numbers,
        'total_draws': len(historical_data)
    }

# =============================================================================
# 🤖 GEMINI AI INTEGRATION
# =============================================================================
def setup_gemini(api_key):
    """Cấu hình Gemini AI"""
    if GEMINI_AVAILABLE and api_key:
        try:
            genai.configure(api_key=api_key)
            return genai.GenerativeModel('gemini-pro')
        except Exception as e:
            st.error(f"❌ Lỗi Gemini API: {e}")
            return None
    return None

def get_gemini_prediction(model, historical_data, bet_analysis, freq_analysis):
    """
    Sử dụng Gemini AI để phân tích và dự đoán
    """
    if not model:
        return None
    
    try:
        # Chuẩn bị dữ liệu cho AI
        recent_results = historical_data[-10:] if len(historical_data) >= 10 else historical_data
        
        prompt = f"""
Bạn là chuyên gia phân tích xổ số với AI. Hãy phân tích dữ liệu sau và đưa ra dự đoán:

**Dữ liệu lịch sử (10 kỳ gần nhất):**
{json.dumps(recent_results, indent=2, ensure_ascii=False)}

**Phân tích số bệt (lâu chưa về):**
{json.dumps(bet_analysis, indent=2, ensure_ascii=False)}

**Phân tích tần suất:**
{json.dumps(freq_analysis, indent=2, ensure_ascii=False)}

**Yêu cầu:**
1. Phân tích xu hướng và pattern
2. Đề xuất 1 bạch thủ (1 số 2 chữ số)
3. Đề xuất 1 song thủ (2 số 2 chữ số)
4. Đề xuất 1 dàn đề 10 số
5. Giải thích lý do chọn

Trả lời theo format JSON:
{{
    "bach_thu": "XX",
    "song_thu": ["XX", "YY"],
    "dan_de": ["XX", "YY", "ZZ", ...],
    "confidence": 0-100,
    "reasoning": "Giải thích ngắn gọn"
}}
"""
        
        response = model.generate_content(prompt)
        
        # Parse JSON từ response
        response_text = response.text.strip()
        # Loại bỏ markdown code blocks nếu có
        if response_text.startswith("```"):
            response_text = response_text.split("```")[1]
            if response_text.startswith("json"):
                response_text = response_text[4:]
        response_text = response_text.strip()
        
        prediction = json.loads(response_text)
        return prediction
        
    except Exception as e:
        st.warning(f"⚠️ Gemini AI gặp lỗi: {e}")
        return None

# =============================================================================
# 📡 SCRAPING
# =============================================================================
@st.cache_data(ttl=300)
def get_live_xsmb():
    url = "https://xosodaiphat.com/xsmb-xổ-số-miền-bắc.html"
    headers = {'User-Agent': 'Mozilla/5.0'}
    
    try:
        res = requests.get(url, headers=headers, timeout=15)
        soup = BeautifulSoup(res.content, 'html.parser')
        
        def get_txt(cls):
            item = soup.find("span", class_=cls)
            return item.text.strip() if item else "..."
        
        return {
            "Đặc Biệt": get_txt("special-temp"),
            "Giải Nhất": get_txt("g1-temp"),
            "Giải Nhì": [get_txt(f"g2_{i}-temp") for i in range(2)],
            "Giải Ba": [get_txt(f"g3_{i}-temp") for i in range(6)],
            "Giải Tư": [get_txt(f"g4_{i}-temp") for i in range(4)],
            "Giải Năm": [get_txt(f"g5_{i}-temp") for i in range(6)],
            "Giải Sáu": [get_txt(f"g6_{i}-temp") for i in range(3)],
            "Giải Bảy": [get_txt(f"g7_{i}-temp") for i in range(4)],
            "time": datetime.now().strftime("%H:%M:%S %d/%m")
        }
    except:
        return None

# =============================================================================
# 🎲 PREDICTION ENGINE
# =============================================================================
def generate_predictions_with_ai(historical_data, bet_analysis, gemini_pred=None):
    """
    Kết hợp phân tích thống kê + AI để dự đoán
    """
    # Phân tích tần suất
    freq_analysis = analyze_frequency(historical_data)
    
    # Lấy số nóng và số lạnh
    hot_numbers = [num for num, count in freq_analysis['hot_numbers'][:5]]
    cold_numbers = [num for num, days in bet_analysis['cold_numbers'][:5]]
    
    # Nếu có Gemini prediction, ưu tiên sử dụng
    if gemini_pred:
        return {
            "bach_thu": gemini_pred.get('bach_thu', hot_numbers[0] if hot_numbers else "00"),
            "song_thu": tuple(gemini_pred.get('song_thu', hot_numbers[:2])),
            "xien_2": f"{gemini_pred.get('bach_thu', '00')} - {gemini_pred.get('song_thu', ['00'])[0]}",
            "dan_de": gemini_pred.get('dan_de', hot_numbers + cold_numbers)[:10],
            "hot_numbers": hot_numbers,
            "cold_numbers": cold_numbers,
            "ai_confidence": gemini_pred.get('confidence', 0),
            "ai_reasoning": gemini_pred.get('reasoning', ''),
            'source': 'Gemini AI'
        }
    
    # Fallback: Statistical prediction
    bt = hot_numbers[0] if hot_numbers else f"{np.random.randint(0,100):02d}"
    st_list = hot_numbers[1:3] if len(hot_numbers) > 1 else [f"{np.random.randint(0,100):02d}" for _ in range(2)]
    
    dan_de = list(set(hot_numbers[:3] + cold_numbers[:3]))
    while len(dan_de) < 10:
        dan_de.append(f"{np.random.randint(0,100):02d}")
    dan_de = sorted(dan_de)[:10]
    
    return {
        "bach_thu": bt,
        "song_thu": tuple(st_list),
        "xien_2": f"{bt} - {st_list[0]}",
        "dan_de": dan_de,
        "hot_numbers": hot_numbers,
        "cold_numbers": cold_numbers,
        'source': 'Thống kê'
    }

# =============================================================================
# 🚀 MAIN APP
# =============================================================================
def main():
    init_data()
    
    # API Key input
    with st.sidebar:
        st.markdown("### ⚙️ CẤU HÌNH")
        api_key = st.text_input(
            "Gemini API Key", 
            value="AIzaSyARQk3lpoHnK51LQ62OR4vciH0XMFFIZjg",
            type="password",
            help="API key cho Gemini AI"
        )
        
        st.divider()
        
        # Initialize Gemini
        gemini_model = setup_gemini(api_key) if api_key else None
        
        if gemini_model:
            st.success("✅ Gemini AI đã kích hoạt")
            st.markdown('<span class="gemini-badge">🤖 AI ACTIVE</span>', 
                       unsafe_allow_html=True)
        else:
            st.warning("⚠️ Gemini AI chưa kích hoạt")
        
        st.divider()
        
        st.markdown("### 📊 THỐNG KÊ")
        stats = st.session_state.statistics
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f'''
            <div class="stat-card">
                <div class="stat-value win">{stats['wins']}</div>
                <div class="stat-label">Thắng</div>
            </div>
            ''', unsafe_allow_html=True)
        with col2:
            st.markdown(f'''
            <div class="stat-card">
                <div class="stat-value loss">{stats['losses']}</div>
                <div class="stat-label">Thua</div>
            </div>
            ''', unsafe_allow_html=True)
        
        wr = stats['win_rate']
        st.markdown(f'''
        <div class="stat-card" style="margin-top: 10px;">
            <div class="stat-value" style="color: {'#00ff88' if wr >= 50 else '#ff4b4b'}">
                {wr:.1f}%
            </div>
            <div class="stat-label">Tỷ lệ thắng</div>
        </div>
        ''', unsafe_allow_html=True)
        
        st.divider()
        
        if st.button("🔄 Xóa lịch sử", use_container_width=True):
            st.session_state.statistics = {
                'predictions': [], 'total_predictions': 0,
                'wins': 0, 'losses': 0, 'win_rate': 0.0
            }
            st.session_state.historical_results = []
            st.rerun()
    
    # HEADER
    st.markdown('''
    <div class="header-gold">
        <h1 style="margin:0;">💎 AI-QUANTUM PRO 2026</h1>
        <p style="margin:5px 0 0;">🎯 Gemini AI Enhanced • Số Bệt Detection • Real-Time</p>
    </div>
    ''', unsafe_allow_html=True)
    
    # TABS
    tab1, tab2, tab3, tab4 = st.tabs([
        "🎯 Dự Đoán AI", 
        "📊 Phân Tích Số Bệt", 
        "📡 Kết Quả", 
        "📈 Lịch Sử"
    ])
    
    with tab1:
        st.markdown("###  DỰ ĐOÁN VỚI GEMINI AI")
        
        # Get live data
        data = get_live_xsmb()
        if data:
            save_historical_result(data)
        
        # Phân tích
        bet_analysis = analyze_so_bet(st.session_state.historical_results, days=7)
        freq_analysis = analyze_frequency(st.session_state.historical_results)
        
        # Gemini prediction
        gemini_pred = None
        if gemini_model and len(st.session_state.historical_results) >= 3:
            with st.spinner("🤖 Gemini AI đang phân tích..."):
                gemini_pred = get_gemini_prediction(
                    gemini_model, 
                    st.session_state.historical_results,
                    bet_analysis,
                    freq_analysis
                )
        
        # Generate final predictions
        predictions = generate_predictions_with_ai(
            st.session_state.historical_results,
            bet_analysis,
            gemini_pred
        )
        
        # Display AI confidence
        if 'ai_confidence' in predictions and predictions['ai_confidence']:
            st.markdown(f'''
            <div style="background: linear-gradient(135deg, #4285f4, #34a853);
                        padding: 15px; border-radius: 10px; text-align: center;
                        margin: 15px 0;">
                <div style="font-size: 24px; font-weight: bold; color: white;">
                    🤖 Gemini AI Confidence: {predictions['ai_confidence']}%
                </div>
                <div style="color: white; margin-top: 5px;">
                    {predictions.get('ai_reasoning', 'Đang phân tích...')}
                </div>
            </div>
            ''', unsafe_allow_html=True)
        
        # Display predictions
        c1, c2, c3 = st.columns(3)
        
        with c1:
            st.markdown(f'''
            <div class="pred-box">
                <div style="color:#aaa; font-size:13px;">🎯 BẠCH THỦ</div>
                <div class="bet-number">{predictions['bach_thu']}</div>
                <div style="font-size:11px; color:#666; margin-top:10px;">
                    Nguồn: {predictions.get('source', 'Thống kê')}
                </div>
            </div>
            ''', unsafe_allow_html=True)
        
        with c2:
            st.markdown(f'''
            <div class="pred-box">
                <div style="color:#aaa; font-size:13px;">🎯 SONG THỦ</div>
                <div style="font-size:28px; color:#FFD700; font-weight:bold; margin:10px 0;">
                    {predictions['song_thu'][0]} - {predictions['song_thu'][1]}
                </div>
            </div>
            ''', unsafe_allow_html=True)
        
        with c3:
            st.markdown(f'''
            <div class="pred-box">
                <div style="color:#aaa; font-size:13px;">🎯 XIÊN 2</div>
                <div style="font-size:24px; color:#FFD700; font-weight:bold; margin:10px 0;">
                    {predictions['xien_2']}
                </div>
            </div>
            ''', unsafe_allow_html=True)
        
        # Dàn đề
        st.markdown(f'''
        <div class="pred-box">
            <div style="color:#aaa; font-size:14px; margin-bottom:10px;">
                📋 DÀN ĐỀ 10 SỐ (AI + Thống kê)
            </div>
            <div style="font-size:20px; color:#fff; letter-spacing:2px;">
                {', '.join(predictions['dan_de'])}
            </div>
        </div>
        ''', unsafe_allow_html=True)
        
        # Check result form
        st.markdown("---")
        st.markdown("### ✅ KIỂM TRA KẾT QUẢ")
        
        col_check1, col_check2 = st.columns(2)
        with col_check1:
            check_bt = st.text_input("Bạch Thủ", max_chars=2, placeholder="VD: 76")
        with col_check2:
            check_st = st.text_input("Song Thủ", max_chars=7, placeholder="VD: 09-90")
        
        if st.button("🎯 Kiểm Tra", use_container_width=True, type="primary"):
            if check_bt or check_st:
                all_loto = extract_all_numbers(data) if data else []
                
                if check_bt:
                    is_win = check_bt in all_loto
                    if is_win:
                        st.success(f"🎉 Bạch thủ {check_bt} TRÚNG!")
                        st.session_state.statistics['wins'] += 1
                    else:
                        st.error(f"❌ Bạch thủ {check_bt} trượt")
                        st.session_state.statistics['losses'] += 1
                    
                    st.session_state.statistics['total_predictions'] += 1
                    st.rerun()
        
        st.markdown("---")
        st.markdown('''
        <div class="disclaimer">
            ⚠️ <b>LƯU Ý</b>: Gemini AI phân tích dựa trên dữ liệu lịch sử. 
            Xổ số là may rủi ngẫu nhiên. Chơi có trách nhiệm!
        </div>
        ''', unsafe_allow_html=True)
    
    with tab2:
        st.markdown("### 📊 PHÂN TÍCH SỐ BỆT (SỐ GAN)")
        
        bet_analysis = analyze_so_bet(st.session_state.historical_results, days=7)
        freq_analysis = analyze_frequency(st.session_state.historical_results)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown(f'''
            <div class="stat-card">
                <div class="stat-value">{bet_analysis['total_cold']}</div>
                <div class="stat-label">Số chưa về (7 ngày)</div>
            </div>
            ''', unsafe_allow_html=True)
            
            st.markdown("#### 🔥 Số nóng (hay về nhất)")
            for i, (num, count) in enumerate(freq_analysis['hot_numbers'][:10], 1):
                st.markdown(f"{i}. **{num}** - {count} lần")
        
        with col2:
            st.markdown("#### ❄️ Số bệt/lạnh (lâu chưa về)")
            cold_html = '<div style="margin: 10px 0;">'
            for num, days in bet_analysis['cold_numbers'][:15]:
                color = "#ff4b4b" if days >= 5 else "#ffa500" if days >= 3 else "#4a90e2"
                cold_html += f'''
                <span class="cold-number" style="background: {color};">
                    {num} ({days} ngày)
                </span>
                '''
            cold_html += '</div>'
            st.markdown(cold_html, unsafe_allow_html=True)
        
        # Detailed table
        st.markdown("---")
        st.markdown("#### 📋 Chi tiết số bệt")
        
        if bet_analysis['cold_numbers']:
            df_cold = pd.DataFrame(bet_analysis['cold_numbers'][:20], 
                                  columns=['Số', 'Số ngày chưa về'])
            st.dataframe(df_cold, use_container_width=True, hide_index=True)
    
    with tab3:
        st.markdown("### 📡 KẾT QUẢ TRỰC TIẾP")
        
        data = get_live_xsmb()
        
        if data is None:
            st.error("❌ Không tải được kết quả")
        else:
            col1, col2 = st.columns([3, 1])
            with col1:
                st.caption(f"🕐 Cập nhật: **{data.get('time', 'N/A')}**")
            with col2:
                if st.button("🔄 Làm mới", use_container_width=True):
                    st.cache_data.clear()
                    st.rerun()
            
            db = data.get("Đặc Biệt", "....")
            st.markdown(f'''
            <div style="background: linear-gradient(135deg, #1a1a2e, #16213e); 
                        border: 2px solid #D4AF37; border-radius: 15px; 
                        padding: 20px; text-align: center; margin: 10px 0;">
                <div style="font-size: 18px; color: #aaa; margin-bottom: 10px;">
                    🏆 ĐẶC BIỆT
                </div>
                <div style="font-size: 42px; color: #ff4b4b; font-weight: bold; 
                            letter-spacing: 6px;">{db}</div>
            </div>
            ''', unsafe_allow_html=True)
    
    with tab4:
        st.markdown("### 📈 LỊCH SỬ DỰ ĐOÁN")
        
        if st.session_state.statistics['predictions']:
            df = pd.DataFrame(st.session_state.statistics['predictions'])
            display_df = df[['date', 'type', 'numbers', 'result_numbers', 'is_win']]
            display_df = display_df.sort_values('date', ascending=False)
            
            st.dataframe(
                display_df,
                use_container_width=True,
                hide_index=True,
                column_config={
                    "date": "Ngày",
                    "type": "Loại",
                    "numbers": "Số",
                    "result_numbers": "Kết quả",
                    "is_win": st.column_config.CheckboxColumn("Trúng?")
                }
            )
        else:
            st.info("📭 Chưa có lịch sử")
    
    # FOOTER
    st.markdown("---")
    st.markdown('<div style="text-align: center; color: #666; padding: 20px;">'
                '💎 AI-QUANTUM PRO 2026 • Gemini AI Enhanced<br>'
                'Chơi xổ số có trách nhiệm - 18+</div>',
                unsafe_allow_html=True)

if __name__ == "__main__":
    main()