# =============================================================================
# 💎 AI-QUANTUM PRO 2026 - GEMINI ENHANCED
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
import google.generativeai as genai

# =============================================================================
# 🔧 CẤU HÌNH
# =============================================================================
st.set_page_config(
    page_title="💎 AI-QUANTUM PRO 2026 - GEMINI",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Gemini API Configuration
GEMINI_API_KEY = "AIzaSyARQk3lpoHnK51LQ62OR4vciH0XMFFIZjg"
genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel('gemini-pro')

st_autorefresh(interval=120000, key="live_update")

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
        box-shadow: 0 4px 20px rgba(212, 175, 55, 0.6);
    }
    .stat-card {
        background: linear-gradient(135deg, #1a1a2e, #16213e);
        border: 2px solid #D4AF37; border-radius: 12px;
        padding: 15px; text-align: center; margin: 5px;
    }
    .stat-value { font-size: 28px; font-weight: bold; color: #FFD700; }
    .stat-label { font-size: 12px; color: #888; margin-top: 5px; }
    .gan-number {
        background: rgba(255, 75, 75, 0.3);
        border: 1px solid #ff4b4b;
        padding: 5px 10px; border-radius: 5px;
        margin: 2px; display: inline-block;
    }
    .hot-number {
        background: rgba(0, 255, 136, 0.3);
        border: 1px solid #00ff88;
        padding: 5px 10px; border-radius: 5px;
        margin: 2px; display: inline-block;
    }
    .pred-box {
        border: 2px solid #D4AF37; border-radius: 12px;
        padding: 20px; background: linear-gradient(135deg, #111, #1a1a2e);
        text-align: center; margin: 8px 0;
    }
    .ai-badge {
        background: linear-gradient(135deg, #4285f4, #34a853);
        color: white; padding: 3px 10px; border-radius: 12px;
        font-size: 11px; font-weight: bold; margin-left: 5px;
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
# 💾 AUTO HISTORY MANAGEMENT
# =============================================================================
def init_history():
    """Khởi tạo lịch sử tự động"""
    if 'daily_history' not in st.session_state:
        st.session_state.daily_history = {
            'predictions': [],
            'daily_stats': {},
            'last_updated': None
        }
    
    # Load từ file nếu có (persistent storage)
    try:
        import os
        if os.path.exists('history.json'):
            with open('history.json', 'r', encoding='utf-8') as f:
                st.session_state.daily_history = json.load(f)
    except:
        pass

def save_daily_prediction(date_str, pred_type, numbers, result_numbers, is_win, confidence=0):
    """Lưu dự đoán tự động theo ngày"""
    today = datetime.now().strftime("%Y-%m-%d")
    
    pred = {
        'date': date_str,
        'type': pred_type,
        'numbers': numbers,
        'result_numbers': result_numbers,
        'is_win': is_win,
        'confidence': confidence,
        'timestamp': datetime.now().isoformat()
    }
    
    st.session_state.daily_history['predictions'].append(pred)
    
    # Update daily stats
    if today not in st.session_state.daily_history['daily_stats']:
        st.session_state.daily_history['daily_stats'][today] = {
            'total': 0, 'wins': 0, 'losses': 0
        }
    
    st.session_state.daily_history['daily_stats'][today]['total'] += 1
    if is_win:
        st.session_state.daily_history['daily_stats'][today]['wins'] += 1
    else:
        st.session_state.daily_history['daily_stats'][today]['losses'] += 1
    
    st.session_state.daily_history['last_updated'] = datetime.now().isoformat()
    save_history_to_file()

def save_history_to_file():
    """Lưu lịch sử ra file JSON"""
    try:
        with open('history.json', 'w', encoding='utf-8') as f:
            json.dump(st.session_state.daily_history, f, ensure_ascii=False, indent=2)
    except Exception as e:
        st.error(f"❌ Không thể lưu lịch sử: {e}")

def get_loto_gan(all_results, days_threshold=7):
    """Phát hiện số bệt/loto gan - số lâu chưa về"""
    counter = Counter(all_results)
    all_possible = [f"{i:02d}" for i in range(100)]
    
    # Số chưa xuất hiện hoặc rất ít
    gan_numbers = []
    for num in all_possible:
        count = counter.get(num, 0)
        if count == 0 or count < 2:
            gan_numbers.append((num, count))
    
    # Sort theo số lần xuất hiện (ít nhất lên đầu)
    gan_numbers.sort(key=lambda x: x[1])
    return gan_numbers[:10]  # Top 10 số gan nhất

def get_hot_numbers(all_results):
    """Số nóng - xuất hiện nhiều"""
    counter = Counter(all_results)
    return counter.most_common(10)

# =============================================================================
# 📡 SCRAPING
# =============================================================================
@st.cache_data(ttl=300)
def get_live_xsmb():
    """Crawl kết quả XSMB"""
    urls = [
        "https://xosodaiphat.com/xsmb-xổ-số-miền-bắc.html",
        "https://xoso.com.vn/xsmb.html"
    ]
    
    headers = {'User-Agent': 'Mozilla/5.0'}
    
    for url in urls:
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
            continue
    
    return None

# =============================================================================
# 🧠 GEMINI AI PREDICTION
# =============================================================================
def gemini_predict(hot_numbers, gan_numbers, recent_results):
    """
    Sử dụng Gemini AI để phân tích và dự đoán
    """
    try:
        prompt = f"""
        Bạn là chuyên gia phân tích xổ số với 20 năm kinh nghiệm.
        
        Dữ liệu hiện tại:
        - Số NÓNG (xuất hiện nhiều): {', '.join([n[0] for n in hot_numbers[:5]])}
        - Số LẠNH/GAN (lâu chưa về): {', '.join([n[0] for n in gan_numbers[:5]])}
        - Kết quả gần đây: {', '.join(recent_results[-10:])}
        
        Hãy phân tích và đưa ra dự đoán:
        1. Bạch thủ (1 số 2 chữ số) - độ tin cậy %
        2. Song thủ (2 số) - độ tin cậy %
        3. Dàn đề 10 số
        
        Phân tích dựa trên:
        - Quy luật xác suất
        - Chu kỳ lặp lại
        - Xu hướng gần đây
        
        Trả lời theo format JSON:
        {{
            "bach_thu": "76",
            "bach_thu_confidence": 75,
            "song_thu": ["09", "90"],
            "song_thu_confidence": 65,
            "dan_de": ["01", "12", "23", "34", "45", "56", "67", "78", "89", "90"],
            "analysis": "Giải thích ngắn gọn lý do chọn các số này"
        }}
        """
        
        response = model.generate_content(prompt)
        
        # Parse JSON từ response
        try:
            # Tìm JSON trong response
            import re
            json_match = re.search(r'\{.*\}', response.text, re.DOTALL)
            if json_match:
                result = json.loads(json_match.group())
                return result
        except:
            pass
        
        # Fallback nếu parse failed
        return {
            "bach_thu": hot_numbers[0][0] if hot_numbers else "76",
            "bach_thu_confidence": 70,
            "song_thu": [hot_numbers[1][0] if len(hot_numbers) > 1 else "09", 
                        gan_numbers[0][0] if gan_numbers else "90"],
            "song_thu_confidence": 60,
            "dan_de": [f"{i:02d}" for i in range(10)],
            "analysis": "Phân tích AI không khả dụng"
        }
        
    except Exception as e:
        st.error(f"⚠️ Gemini AI error: {e}")
        return None

# =============================================================================
# 🎲 TRUYỀN THỐNG + AI PREDICTION
# =============================================================================
def advanced_predictions(all_loto, data):
    """
    Hệ thống dự đoán đa lớp:
    1. Phân tích tần suất truyền thống
    2. Phát hiện số gan
    3. Gemini AI analysis
    4. Kết hợp kết quả
    """
    
    # 1. Phân tích cơ bản
    hot = get_hot_numbers(all_loto)
    gan = get_loto_gan(all_loto)
    
    # 2. Gemini AI prediction
    recent_results = all_loto[-30:] if len(all_loto) >= 30 else all_loto
    ai_pred = gemini_predict(hot, gan, recent_results)
    
    # 3. Truyền thống prediction
    traditional_bt = hot[0][0] if hot else f"{np.random.randint(0,100):02d}"
    traditional_st = [
        hot[1][0] if len(hot) > 1 else f"{np.random.randint(0,100):02d}",
        gan[0][0] if gan else f"{np.random.randint(0,100):02d}"
    ]
    
    # 4. Kết hợp AI + Truyền thống
    if ai_pred:
        final_bt = ai_pred.get('bach_thu', traditional_bt)
        final_st = ai_pred.get('song_thu', traditional_st)
        final_dan = ai_pred.get('dan_de', [f"{i:02d}" for i in range(10)])
        ai_confidence = ai_pred.get('bach_thu_confidence', 70)
        analysis = ai_pred.get('analysis', '')
    else:
        final_bt = traditional_bt
        final_st = traditional_st
        final_dan = [n[0] for n in hot[:5]] + [n[0] for n in gan[:5]]
        ai_confidence = 0
        analysis = "Sử dụng thuật toán truyền thống"
    
    # Dàn đề 10 số
    dan_de = list(set(final_dan))
    while len(dan_de) < 10:
        dan_de.append(f"{np.random.randint(0,100):02d}")
    dan_de = sorted(dan_de)[:10]
    
    return {
        "bach_thu": final_bt,
        "song_thu": final_st if isinstance(final_st, list) else list(final_st),
        "xien_2": f"{final_bt} - {final_st[0] if isinstance(final_st, list) else final_st}",
        "dan_de": dan_de,
        "hot_numbers": hot,
        "gan_numbers": gan,
        "ai_confidence": ai_confidence,
        "analysis": analysis,
        "ai_used": ai_pred is not None
    }

# =============================================================================
# 🚀 MAIN APP
# =============================================================================
def main():
    init_history()
    
    # HEADER
    st.markdown('''
    <div class="header-gold">
        <h1 style="margin:0;">💎 AI-QUANTUM PRO 2026</h1>
        <p style="margin:5px 0 0;">🎯 Gemini AI Enhanced • Số Gan • Thống Kê Tự Động</p>
    </div>
    ''', unsafe_allow_html=True)
    
    # SIDEBAR - AUTO STATISTICS
    with st.sidebar:
        st.markdown("### 📊 THỐNG KÊ TỰ ĐỘNG")
        
        # Tính tổng stats
        total = len(st.session_state.daily_history['predictions'])
        wins = sum(1 for p in st.session_state.daily_history['predictions'] if p['is_win'])
        losses = total - wins
        win_rate = (wins / total * 100) if total > 0 else 0
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f'''
            <div class="stat-card">
                <div class="stat-value" style="color: #00ff88">{wins}</div>
                <div class="stat-label">Thắng</div>
            </div>
            ''', unsafe_allow_html=True)
        with col2:
            st.markdown(f'''
            <div class="stat-card">
                <div class="stat-value" style="color: #ff4b4b">{losses}</div>
                <div class="stat-label">Thua</div>
            </div>
            ''', unsafe_allow_html=True)
        
        st.markdown(f'''
        <div class="stat-card" style="margin-top: 10px;">
            <div class="stat-value" style="color: {'#00ff88' if win_rate >= 50 else '#ff4b4b'}">
                {win_rate:.1f}%
            </div>
            <div class="stat-label">Tỷ lệ thắng</div>
            <div class="stat-label">Tổng: {total} lượt</div>
        </div>
        ''', unsafe_allow_html=True)
        
        st.divider()
        
        # Daily stats
        st.markdown("### 📅 Thống kê theo ngày")
        if st.session_state.daily_history['daily_stats']:
            for date, stats in sorted(st.session_state.daily_history['daily_stats'].items(), reverse=True)[:7]:
                day_rate = (stats['wins'] / stats['total'] * 100) if stats['total'] > 0 else 0
                st.caption(f"{date}: {stats['wins']}/{stats['total']} ({day_rate:.0f}%)")
        else:
            st.caption("Chưa có dữ liệu")
        
        st.divider()
        
        if st.button("🔄 Xóa toàn bộ lịch sử", use_container_width=True):
            st.session_state.daily_history = {
                'predictions': [],
                'daily_stats': {},
                'last_updated': None
            }
            try:
                import os
                if os.path.exists('history.json'):
                    os.remove('history.json')
            except:
                pass
            st.rerun()
        
        # Export
        if st.session_state.daily_history['predictions']:
            df_exp = pd.DataFrame(st.session_state.daily_history['predictions'])
            csv = df_exp.to_csv(index=False, encoding='utf-8-sig')
            st.download_button(
                label="📥 Tải lịch sử CSV",
                data=csv,
                file_name=f"ai_quantum_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv",
                use_container_width=True
            )
    
    # TABS
    tab1, tab2, tab3, tab4 = st.tabs([
        "🎯 Dự Đoán AI", 
        "📡 Kết Quả", 
        "📈 Lịch Sử Tự Động",
        "🌐 Website"
    ])
    
    with tab1:
        st.markdown("### 🧠 DỰ ĐOÁN GEMINI AI + THUẬT TOÁN")
        
        data = get_live_xsmb()
        if not 
            st.warning("⚠️ Không tải được kết quả. Sử dụng dữ liệu mẫu.")
            data = {
                "Đặc Biệt": "48076",
                "Giải Nhất": "66442",
                "Giải Nhì": ["97779", "94665"],
                "time": datetime.now().strftime("%H:%M:%S %d/%m")
            }
        
        # Extract all loto
        all_loto = []
        for k, v in data.items():
            if k == "time":
                continue
            if isinstance(v, list):
                for x in v:
                    if x and x != "...":
                        all_loto.append(x[-2:])
            elif v and v != "...":
                all_loto.append(v[-2:])
        
        # Generate predictions
        with st.spinner("🤖 Gemini AI đang phân tích..."):
            preds = advanced_predictions(all_loto, data)
        
        # Display AI Badge
        if preds['ai_used']:
            st.success(f"✅ Gemini AI đã phân tích • Độ tin cậy: {preds['ai_confidence']}%")
        
        # Display Analysis
        if preds['analysis']:
            st.info(f"🔍 **Phân tích AI**: {preds['analysis']}")
        
        # Số GAN (Loto gan)
        st.markdown("#### 🔴 SỐ GAN - LÂU CHƯA VỀ")
        gan_html = ""
        for num, count in preds['gan_numbers'][:10]:
            gan_html += f'<span class="gan-number">{num} ({count} lần)</span> '
        st.markdown(gan_html, unsafe_allow_html=True)
        
        # Số NÓNG
        st.markdown("#### 🟢 SỐ NÓNG - XUẤT HIỆN NHIỀU")
        hot_html = ""
        for num, count in preds['hot_numbers'][:10]:
            hot_html += f'<span class="hot-number">{num} ({count} lần)</span> '
        st.markdown(hot_html, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Display Predictions
        c1, c2, c3 = st.columns(3)
        
        with c1:
            st.markdown(f'''
            <div class="pred-box">
                <div style="color:#aaa; font-size:13px;">
                    🎯 BẠCH THỦ <span class="ai-badge">AI</span>
                </div>
                <div style="font-size:40px; color:#FFD700; font-weight:bold; margin:10px 0;">
                    {preds['bach_thu']}
                </div>
                <div style="font-size:12px; color:#00ff88;">
                    Độ tin cậy: {preds['ai_confidence']}%
                </div>
            </div>
            ''', unsafe_allow_html=True)
        
        with c2:
            st_text = " - ".join(preds['song_thu']) if isinstance(preds['song_thu'], list) else preds['song_thu']
            st.markdown(f'''
            <div class="pred-box">
                <div style="color:#aaa; font-size:13px;">
                    🎯 SONG THỦ <span class="ai-badge">AI</span>
                </div>
                <div style="font-size:28px; color:#FFD700; font-weight:bold; margin:10px 0;">
                    {st_text}
                </div>
            </div>
            ''', unsafe_allow_html=True)
        
        with c3:
            st.markdown(f'''
            <div class="pred-box">
                <div style="color:#aaa; font-size:13px;">🎯 XIÊN 2</div>
                <div style="font-size:24px; color:#FFD700; font-weight:bold; margin:10px 0;">
                    {preds['xien_2']}
                </div>
            </div>
            ''', unsafe_allow_html=True)
        
        # Dàn đề
        st.markdown(f'''
        <div class="pred-box" style="margin-top: 10px;">
            <div style="color:#aaa; font-size:14px; margin-bottom:10px;">
                📋 DÀN ĐỀ 10 SỐ <span class="ai-badge">AI + STATISTICS</span>
            </div>
            <div style="font-size:20px; color:#fff; letter-spacing:2px;">
                {', '.join(preds['dan_de'])}
            </div>
        </div>
        ''', unsafe_allow_html=True)
        
        # Auto-check
        st.markdown("---")
        st.markdown("### ✅ KIỂM TRA & LƯU TỰ ĐỘNG")
        
        col1, col2 = st.columns(2)
        with col1:
            check_num = st.text_input("Nhập số dự đoán", max_chars=10, 
                                     placeholder="VD: 76 hoặc 09-90")
        with col2:
            check_type = st.selectbox("Loại", ["Bạch Thủ", "Song Thủ", "Dàn Đề"])
        
        if st.button("💾 Kiểm Tra & Lưu Tự Động", type="primary", use_container_width=True):
            if check_num:
                is_win = check_num in all_loto or any(n in check_num for n in all_loto)
                
                if is_win:
                    st.success(f"🎉 TRÚNG! Số {check_num} có trong kết quả")
                    st.balloons()
                else:
                    st.error(f"❌ Trượt! Số {check_num} không có")
                
                # Auto save to history
                save_daily_prediction(
                    datetime.now().strftime("%d/%m %H:%M"),
                    check_type,
                    check_num,
                    "Trúng" if is_win else "Trượt",
                    is_win,
                    preds['ai_confidence'] if preds['ai_used'] else 0
                )
                
                st.info("✅ Đã lưu tự động vào lịch sử")
                time.sleep(1)
                st.rerun()
            else:
                st.warning("⚠️ Vui lòng nhập số")
        
        st.markdown("---")
        st.markdown('''
        <div class="disclaimer">
            ⚠️ <b>LƯU Ý</b>: Gemini AI phân tích dựa trên thống kê và xác suất. 
            Xổ số là trò chơi may rủi. Không có gì đảm bảo 100%. 
            <b>Chơi có trách nhiệm!</b>
        </div>
        ''', unsafe_allow_html=True)
    
    with tab2:
        st.markdown("### 📡 KẾT QUẢ TRỰC TIẾP")
        
        data = get_live_xsmb()
        if not 
            data = {"Đặc Biệt": "48076", "time": datetime.now().strftime("%H:%M:%S %d/%m")}
        
        st.markdown(f'''
        <div style="background: linear-gradient(135deg, #1a1a2e, #16213e); 
                    border: 2px solid #D4AF37; border-radius: 15px; 
                    padding: 30px; text-align: center;">
            <div style="font-size: 20px; color: #aaa; margin-bottom: 15px;">🏆 ĐẶC BIỆT</div>
            <div style="font-size: 48px; color: #ff4b4b; font-weight: bold; 
                        letter-spacing: 8px; text-shadow: 0 0 20px rgba(255,75,75,0.5);">
                {data.get('Đặc Biệt', '....')}
            </div>
            <div style="margin-top: 15px; color: #888;">
                Cập nhật: {data.get('time', 'N/A')}
            </div>
        </div>
        ''', unsafe_allow_html=True)
        
        if st.button("🔄 Làm mới kết quả", use_container_width=True):
            st.cache_data.clear()
            st.rerun()
    
    with tab3:
        st.markdown("### 📈 LỊCH SỬ TỰ ĐỘNG THEO NGÀY")
        
        if st.session_state.daily_history['predictions']:
            df = pd.DataFrame(st.session_state.daily_history['predictions'])
            
            # Summary
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Tổng lượt", len(df))
            col2.metric("Thắng", sum(df['is_win']))
            col3.metric("Thua", sum(~df['is_win']))
            col4.metric("Tỷ lệ", f"{sum(df['is_win'])/len(df)*100:.1f}%")
            
            st.divider()
            
            # Filter by date
            dates = sorted(df['date'].unique(), reverse=True)
            selected_date = st.selectbox("Chọn ngày", dates)
            
            df_filtered = df[df['date'] == selected_date] if selected_date else df
            
            # Display
            st.dataframe(
                df_filtered[['date', 'type', 'numbers', 'result_numbers', 'confidence', 'is_win']],
                use_container_width=True,
                hide_index=True,
                column_config={
                    "date": "Ngày/Giờ",
                    "type": "Loại",
                    "numbers": "Số dự đoán",
                    "result_numbers": "Kết quả",
                    "confidence": st.column_config.ProgressColumn(
                        "Độ tin cậy AI",
                        min_value=0,
                        max_value=100,
                    ),
                    "is_win": st.column_config.CheckboxColumn("Trúng?")
                }
            )
            
            # Chart
            if not df.empty:
                st.markdown("#### 📊 Biểu đồ thắng/thua")
                df_chart = df.copy()
                df_chart['is_win_num'] = df_chart['is_win'].map({True: 1, False: 0})
                st.line_chart(df_chart.set_index('date')['is_win_num'])
        else:
            st.info("📭 Chưa có lịch sử. Hãy bắt đầu dự đoán!")
    
    with tab4:
        st.markdown("### 🌐 WEBSITE XOSODAIPHAT")
        st.markdown('''
        <div style="border: 2px solid #D4AF37; border-radius: 15px; 
                    overflow: hidden; height: 800px;">
            <iframe src="https://xosodaiphat.com/xsmb-xổ-số-miền-bắc.html" 
                    style="width:100%; height:100%; border:none;"
                    sandbox="allow-same-origin allow-scripts">
            </iframe>
        </div>
        ''', unsafe_allow_html=True)
    
    # FOOTER
    st.markdown("---")
    st.markdown('''
    <div style="text-align: center; color: #666; padding: 20px;">
        💎 AI-QUANTUM PRO 2026 • Gemini AI Enhanced<br>
        <b>Chơi xổ số có trách nhiệm - 18+ only</b>
    </div>
    ''', unsafe_allow_html=True)

if __name__ == "__main__":
    main()