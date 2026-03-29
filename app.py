import streamlit as st
import json, os, re, math, random
import pandas as pd
import numpy as np
from collections import Counter, defaultdict
from itertools import combinations
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# === CẤU HÌNH ===
DB_FILE = "titan_v32_data.json"
NVIDIA_API_KEY = "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"
GEMINI_API_KEY = "AIzaSyD1-XMO6FsA9ZgAf2P6nIiXLPp8moTPMrc"
LUCKY_OX = [0, 2, 5, 6, 7, 8]

st.set_page_config(page_title="TITAN V52 - QUANTUM AI", page_icon="🧠", layout="centered")

# === CSS CAO CẤP ===
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700;900&display=swap');
    
    .stApp {
        background: linear-gradient(180deg, #0a0a0a 0%, #1a0a2e 50%, #0a0a0a 100%);
        color: #00ffff;
        font-family: 'Orbitron', sans-serif;
    }
    
    .main-header {
        font-size: 32px;
        font-weight: 900;
        text-align: center;
        background: linear-gradient(90deg, #00ffff, #ff00ff, #00ffff);
        background-size: 200% auto;
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        animation: shine 3s linear infinite;
        text-shadow: 0 0 30px rgba(0, 255, 255, 0.5);
    }
    
    @keyframes shine {
        to { background-position: 200% center; }
    }
    
    .metric-box {
        background: linear-gradient(135deg, #1a0a2e, #0a0a1a);
        border: 2px solid #00ffff;
        border-radius: 15px;
        padding: 15px;
        text-align: center;
        box-shadow: 0 0 20px rgba(0, 255, 255, 0.3);
        margin: 10px 0;
    }
    
    .prediction-card {
        background: linear-gradient(135deg, #1a0a2e, #2e0a4a);
        border-radius: 15px;
        padding: 20px;
        margin: 10px 0;
        border: 3px solid #ff00ff;
        box-shadow: 0 0 30px rgba(255, 0, 255, 0.4);
        text-align: center;
    }
    
    .big-number {
        font-size: 56px;
        font-weight: 900;
        letter-spacing: 12px;
        color: #00ffff;
        text-shadow: 0 0 20px #00ffff;
    }
    
    .heatmap-cell {
        display: inline-block;
        width: 40px;
        height: 40px;
        margin: 3px;
        border-radius: 8px;
        text-align: center;
        line-height: 40px;
        font-weight: bold;
        font-size: 16px;
    }
    
    .tab-button {
        background: linear-gradient(135deg, #1a0a2e, #0a0a1a);
        border: 2px solid #333;
        color: #888;
        padding: 12px 20px;
        border-radius: 10px;
        margin: 5px;
        cursor: pointer;
        transition: all 0.3s;
    }
    
    .tab-button.active {
        border-color: #00ffff;
        color: #00ffff;
        box-shadow: 0 0 15px rgba(0, 255, 255, 0.5);
    }
    
    .confidence-bar {
        height: 25px;
        background: #0a0a1a;
        border-radius: 12px;
        overflow: hidden;
        margin: 10px 0;
    }
    
    .confidence-fill {
        height: 100%;
        background: linear-gradient(90deg, #ff0000, #ffff00, #00ff00);
        display: flex;
        align-items: center;
        justify-content: center;
        font-weight: bold;
        color: #000;
        transition: width 0.5s;
    }
    
    .history-win { color: #00ff00; font-weight: 900; }
    .history-lose { color: #ff0040; font-weight: 900; }
    
    .tag {
        display: inline-block;
        padding: 5px 12px;
        border-radius: 15px;
        font-size: 10px;
        font-weight: bold;
        margin: 3px;
    }
    .tag-hot { background: #ff0040; color: white; }
    .tag-gan { background: #00ffff; color: black; }
    .tag-bet { background: #ffff00; color: black; }
    
    button {
        border-radius: 10px;
        font-weight: bold;
    }
    
    .stTextInput input, .stTextArea textarea {
        background: #0a0a1a;
        border: 2px solid #333;
        color: #00ffff;
    }
</style>
""", unsafe_allow_html=True)

# === DATABASE MANAGER ===

class DatabaseManager:
    def __init__(self, filename):
        self.filename = filename
        self.data = self._load()
    
    def _load(self):
        if os.path.exists(self.filename):
            with open(self.filename, 'r', encoding='utf-8') as f:
                return json.load(f)
        return {
            'results': [],
            'predictions': [],
            'model_weights': {'frequency': 1.0, 'markov': 1.0, 'ml': 1.0, 'pattern': 1.0},
            'performance': {'total': 0, 'wins': 0, 'win_rate': 0}
        }
    
    def save(self):
        with open(self.filename, 'w', encoding='utf-8') as f:
            json.dump(self.data, f, indent=2, ensure_ascii=False)
    
    def add_result(self, number):
        if number not in self.data['results']:
            self.data['results'].append(number)
            self.save()
    
    def add_prediction(self, pred_data):
        self.data['predictions'].insert(0, pred_data)
        if len(self.data['predictions']) > 100:
            self.data['predictions'] = self.data['predictions'][:100]
        self.save()
    
    def update_performance(self, is_win):
        self.data['performance']['total'] += 1
        if is_win:
            self.data['performance']['wins'] += 1
        self.data['performance']['win_rate'] = (
            self.data['performance']['wins'] / self.data['performance']['total']
            if self.data['performance']['total'] > 0 else 0
        )
        self.save()
    
    def adjust_weights(self, strategy, success):
        factor = 1.1 if success else 0.9
        self.data['model_weights'][strategy] *= factor
        self.data['model_weights'][strategy] = max(0.5, min(2.0, self.data['model_weights'][strategy]))
        self.save()

# === AI ENGINE ===

class QuantumAIEngine:
    def __init__(self, db_manager):
        self.db = db_manager
        self.scaler = StandardScaler()
        self.rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.lr_model = LogisticRegression(random_state=42)
        self.markov_matrix = np.zeros((10, 10))
        
    def _extract_features(self, db, target_digit, position=0):
        """Trích xuất đặc trưng cho ML"""
        if len(db) < 20:
            return [0] * 10
        
        features = []
        recent = db[-20:]
        
        # 1. Tần suất
        all_digits = "".join(recent)
        freq = all_digits.count(str(target_digit)) / len(all_digits)
        features.append(freq)
        
        # 2. Khoảng cách lần xuất hiện gần nhất
        gap = 0
        for num in reversed(recent):
            if str(target_digit) in num:
                break
            gap += 1
        features.append(gap / 20)
        
        # 3. Xuất hiện theo vị trí
        pos_count = sum(1 for num in recent if len(num) > position and num[position] == str(target_digit))
        features.append(pos_count / len(recent))
        
        # 4. Pattern trước đó
        pattern_score = 0
        for i in range(len(recent) - 1):
            if str(target_digit) in recent[i+1]:
                prev_digits = recent[i]
                if str(target_digit) in prev_digits:
                    pattern_score += 1
        features.append(pattern_score / max(len(recent) - 1, 1))
        
        # 5. Chu kỳ
        positions = [i for i, num in enumerate(db) if str(target_digit) in num]
        if len(positions) >= 2:
            gaps = [positions[i+1] - positions[i] for i in range(len(positions)-1)]
            avg_gap = sum(gaps) / len(gaps)
            features.append(1 / (avg_gap + 1))
        else:
            features.append(0)
        
        # 6. Số lần xuất hiện liên tiếp (bệt)
        streak = 0
        for num in reversed(recent):
            if str(target_digit) in num:
                streak += 1
            else:
                break
        features.append(streak / 5)
        
        # 7. Digital root correlation
        dr_target = target_digit % 9
        dr_matches = 0
        for num in recent:
            dr_num = sum(int(d) for d in num) % 9
            if dr_num == dr_target:
                dr_matches += 1
        features.append(dr_matches / len(recent))
        
        # 8-10. Position specific features
        for pos in range(3):
            pos_freq = sum(1 for num in recent if len(num) > pos and num[pos] == str(target_digit))
            features.append(pos_freq / len(recent))
        
        return features
    
    def _build_markov_matrix(self, db):
        """Xây dựng ma trận Markov"""
        self.markov_matrix = np.zeros((10, 10))
        
        for i in range(len(db) - 1):
            curr_digits = [int(d) for d in db[i]]
            next_digits = [int(d) for d in db[i+1]]
            
            for cd in curr_digits:
                for nd in next_digits:
                    self.markov_matrix[cd][nd] += 1
        
        # Normalize
        row_sums = self.markov_matrix.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1
        self.markov_matrix = self.markov_matrix / row_sums
    
    def _calculate_markov_probability(self, digit, db):
        """Tính xác suất Markov"""
        if len(db) < 2:
            return 0.1
        
        last_num = db[-1]
        prob = 0
        for d in last_num:
            prob += self.markov_matrix[int(d)][digit]
        
        return prob / len(last_num)
    
    def _detect_pattern(self, db, digit):
        """Phát hiện pattern đặc biệt"""
        if len(db) < 10:
            return 0
        
        patterns = {
            'alternating': 0,
            'cluster': 0,
            'cycle': 0
        }
        
        # Alternating pattern
        appearances = [1 if str(digit) in num else 0 for num in db[-15:]]
        for i in range(len(appearances) - 1):
            if appearances[i] != appearances[i+1]:
                patterns['alternating'] += 1
        
        # Cluster pattern
        cluster_count = 0
        in_cluster = False
        for app in appearances:
            if app == 1:
                if not in_cluster:
                    cluster_count += 1
                in_cluster = True
            else:
                in_cluster = False
        patterns['cluster'] = cluster_count
        
        # Cycle detection
        positions = [i for i, num in enumerate(db) if str(digit) in num]
        if len(positions) >= 3:
            gaps = [positions[i+1] - positions[i] for i in range(len(positions)-1)]
            if len(set(gaps)) == 1:
                patterns['cycle'] = 1
        
        return sum(patterns.values()) / 3
    
    def predict(self, db):
        """Dự đoán với ensemble learning"""
        if len(db) < 15:
            return None
        
        self._build_markov_matrix(db)
        
        # Chuẩn bị data cho ML
        X_train = []
        y_train = []
        
        for i in range(len(db) - 1):
            for digit in range(10):
                features = self._extract_features(db[:i+1], digit)
                X_train.append(features)
                y_train.append(1 if str(digit) in db[i+1] else 0)
        
        # Train models
        if len(X_train) > 10:
            X_train = np.array(X_train)
            y_train = np.array(y_train)
            
            try:
                self.rf_model.fit(X_train, y_train)
                self.lr_model.fit(X_train, y_train)
            except:
                pass
        
        # Calculate probabilities for each digit
        digit_probs = {}
        weights = self.db.data['model_weights']
        
        for digit in range(10):
            probs = []
            
            # 1. Frequency-based
            recent_str = "".join(db[-30:])
            freq_prob = recent_str.count(str(digit)) / len(recent_str)
            probs.append(freq_prob * weights['frequency'])
            
            # 2. Markov
            markov_prob = self._calculate_markov_probability(digit, db)
            probs.append(markov_prob * weights['markov'])
            
            # 3. ML prediction
            if len(X_train) > 10:
                features = self._extract_features(db, digit)
                try:
                    rf_prob = self.rf_model.predict_proba([features])[0][1]
                    lr_prob = self.lr_model.predict_proba([features])[0][1]
                    ml_prob = (rf_prob + lr_prob) / 2
                except:
                    ml_prob = 0.5
                probs.append(ml_prob * weights['ml'])
            
            # 4. Pattern
            pattern_score = self._detect_pattern(db, digit)
            probs.append(pattern_score * weights['pattern'])
            
            # Weighted average
            total_weight = sum(weights.values())
            digit_probs[digit] = sum(probs) / total_weight
        
        # Select top 2 digits with strategy
        sorted_digits = sorted(digit_probs.items(), key=lambda x: x[1], reverse=True)
        
        # Strategy selection
        strategy = self._select_strategy(db, digit_probs)
        
        if strategy == 'hot':
            # Take top 2 hot numbers
            selected = [d for d, p in sorted_digits[:2]]
        elif strategy == 'hot_cold':
            # 1 hot, 1 medium
            selected = [sorted_digits[0][0], sorted_digits[5][0]]
        elif strategy == 'balanced':
            # Take 2 with highest combined probability
            selected = [sorted_digits[0][0], sorted_digits[1][0]]
        else:
            selected = [sorted_digits[0][0], sorted_digits[1][0]]
        
        # Calculate confidence
        confidence = (digit_probs[selected[0]] + digit_probs[selected[1]]) / 2 * 100
        confidence = min(95, max(40, confidence))
        
        return {
            'numbers': selected,
            'probabilities': {str(d): f"{digit_probs[d]*100:.1f}%" for d in range(10)},
            'confidence': confidence,
            'strategy': strategy,
            'all_probs': digit_probs
        }
    
    def _select_strategy(self, db, digit_probs):
        """Tự động chọn chiến lược"""
        if len(db) < 30:
            return 'hot'
        
        # Calculate entropy
        probs = list(digit_probs.values())
        mean_prob = sum(probs) / len(probs)
        variance = sum((p - mean_prob) ** 2 for p in probs) / len(probs)
        
        # Check recent performance
        recent_preds = self.db.data['predictions'][:10]
        if recent_preds:
            win_rate = sum(1 for p in recent_preds if p.get('result') == 'win') / len(recent_preds)
        else:
            win_rate = 0.5
        
        if variance > 0.01:
            return 'hot'  # Clear pattern
        elif win_rate < 0.3:
            return 'hot_cold'  # Change strategy
        else:
            return 'balanced'

# === XỬ LÝ DỮ LIỆU ===

def get_nums(text):
    clean_text = re.sub(r'\s+', '', text)
    return [n for n in re.findall(r"\d{5}", clean_text) if n]

# === GIAO DIỆN ===

st.markdown('<h1 class="main-header">🧠 TITAN V52 - QUANTUM AI</h1>', unsafe_allow_html=True)
st.markdown('<p style="text-align:center; color:#888; font-size:11px;">AI Ensemble Learning | Markov Chain | Pattern Recognition | Self-Learning</p>', unsafe_allow_html=True)

# Initialize
if 'db_manager' not in st.session_state:
    st.session_state.db_manager = DatabaseManager(DB_FILE)
if 'ai_engine' not in st.session_state:
    st.session_state.ai_engine = QuantumAIEngine(st.session_state.db_manager)
if 'current_tab' not in st.session_state:
    st.session_state.current_tab = 'input'
if 'last_prediction' not in st.session_state:
    st.session_state.last_prediction = None

db_manager = st.session_state.db_manager
ai_engine = st.session_state.ai_engine

# Tab Navigation
tabs = ['📥 Nhập Kết Quả', '🎯 Bạch Thủ', '🔥 Số Bệt', '⏰ Số Gan', '📊 Lịch Sử']
col_tabs = st.columns(len(tabs))
for i, tab in enumerate(tabs):
    if col_tabs[i].button(tab, key=f"tab_{i}", use_container_width=True):
        st.session_state.current_tab = tab.lower().split()[1].lower()

st.markdown("---")

# === TAB 1: NHẬP KẾT QUẢ ===
if st.session_state.current_tab == 'nhập':
    st.markdown("### 📥 Nhập Kết Quả Mới")
    
    user_input = st.text_area("Nhập số 5 chữ số (mỗi dòng 1 số):", height=100, 
                              placeholder="84890\n07119\n33627")
    
    if st.button("💾 Lưu Kết Quả", type="primary", use_container_width=True):
        nums = get_nums(user_input)
        if nums:
            for num in nums:
                db_manager.add_result(num)
            st.success(f"✅ Đã lưu {len(nums)} kết quả!")
            
            # Auto-check prediction
            if st.session_state.last_prediction and nums:
                last_num = nums[-1]
                pred_nums = st.session_state.last_prediction['numbers']
                is_win = all(str(d) in last_num for d in pred_nums)
                
                db_manager.update_performance(is_win)
                db_manager.add_prediction({
                    'date': datetime.now().isoformat(),
                    'result_number': last_num,
                    'predicted': pred_nums,
                    'result': 'win' if is_win else 'lose',
                    'confidence': st.session_state.last_prediction['confidence']
                })
                
                # Adjust weights
                strategy = st.session_state.last_prediction['strategy']
                db_manager.adjust_weights(strategy, is_win)
                
                if is_win:
                    st.balloons()
                    st.success(f"🔥 WIN! Cặp {pred_nums[0]}-{pred_nums[1]} về!")
                else:
                    st.error(f"❌ Thua. Cặp {pred_nums[0]}-{pred_nums[1]} không về.")
                
                st.session_state.last_prediction = None
        else:
            st.warning("⚠️ Không tìm thấy số hợp lệ!")
    
    # Show recent results
    if db_manager.data['results']:
        st.markdown("### 📋 Kết Quả Gần Đây")
        recent = db_manager.data['results'][-10:][::-1]
        for i, num in enumerate(recent):
            st.markdown(f"`{num}`", unsafe_allow_html=True)

# === TAB 2: BẠCH THỦ ===
elif st.session_state.current_tab == 'bạch':
    st.markdown("### 🎯 Dự Đoán Bạch Thủ 2 Số")
    
    if len(db_manager.data['results']) < 15:
        st.warning(f"⚠️ Cần ít nhất 15 kỳ (hiện có: {len(db_manager.data['results'])})")
    else:
        if st.button("🤖 AI Dự Đoán", type="primary", use_container_width=True):
            prediction = ai_engine.predict(db_manager.data['results'])
            if prediction:
                st.session_state.last_prediction = prediction
                
                st.markdown(f"""
                <div class="prediction-card">
                    <div style="font-size:14px; color:#888;">CẶP SỐ VIP</div>
                    <div class="big-number">{prediction['numbers'][0]} - {prediction['numbers'][1]}</div>
                    <div style="margin-top:15px;">
                        <div class="confidence-bar">
                            <div class="confidence-fill" style="width:{prediction['confidence']}%;">
                                {prediction['confidence']:.1f}%
                            </div>
                        </div>
                        <div style="font-size:12px; color:#888; margin-top:5px;">
                            Chiến lược: <b style="color:#00ffff;">{prediction['strategy'].upper()}</b>
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
                # Show probabilities
                st.markdown("### 📊 Xác Suất Các Số")
                probs_df = pd.DataFrame([
                    {'Số': d, 'Xác Suất': prediction['probabilities'][str(d)]}
                    for d in range(10)
                ])
                st.dataframe(probs_df, use_container_width=True, hide_index=True)
                
                # Heatmap
                st.markdown("### 🔥 Heatmap Nhiệt Độ")
                heatmap_html = '<div style="text-align:center; padding:20px;">'
                for d in range(10):
                    prob = float(prediction['probabilities'][str(d)].replace('%', ''))
                    if prob > 15:
                        color = f'background: linear-gradient(135deg, #ff0040, #ff0080); color: white;'
                    elif prob > 10:
                        color = f'background: linear-gradient(135deg, #00ffff, #0080ff); color: black;'
                    else:
                        color = f'background: #333; color: #888;'
                    
                    heatmap_html += f'<div class="heatmap-cell" style="{color}">{d}</div>'
                heatmap_html += '</div>'
                st.markdown(heatmap_html, unsafe_allow_html=True)

# === TAB 3: SỐ BỆT ===
elif st.session_state.current_tab == 'bệt':
    st.markdown("### 🔥 Phân Tích Số Bệt (Lặp Liên Tiếp)")
    
    if len(db_manager.data['results']) < 5:
        st.warning("Chưa đủ dữ liệu")
    else:
        # Calculate streaks
        streaks = {str(d): 0 for d in range(10)}
        for d in range(10):
            for num in reversed(db_manager.data['results']):
                if str(d) in num:
                    streaks[str(d)] += 1
                else:
                    break
        
        # Display
        st.markdown("### Số Đang Bệt (Giảm Dần)")
        sorted_streaks = sorted(streaks.items(), key=lambda x: x[1], reverse=True)
        
        for digit, streak in sorted_streaks:
            if streak > 0:
                bar_width = streak * 10
                st.markdown(f"""
                <div style="margin:10px 0;">
                    <div style="display:flex; justify-content:space-between; margin-bottom:5px;">
                        <span style="font-size:20px; font-weight:900; color:#ffff00;">Số {digit}</span>
                        <span style="font-size:16px; color:#888;">{streak} kỳ</span>
                    </div>
                    <div style="background:#1a0a2e; height:20px; border-radius:10px; overflow:hidden;">
                        <div style="background:linear-gradient(90deg, #ffff00, #ff0040); 
                                    width:{min(bar_width, 100)}%; height:100%;"></div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

# === TAB 4: SỐ GAN ===
elif st.session_state.current_tab == 'gan':
    st.markdown("### ⏰ Phân Tích Số Gan (Lâu Chưa Ra)")
    
    if len(db_manager.data['results']) < 5:
        st.warning("Chưa đủ dữ liệu")
    else:
        # Calculate gaps
        gaps = {str(d): 0 for d in range(10)}
        for d in range(10):
            for num in reversed(db_manager.data['results']):
                if str(d) in num:
                    break
                gaps[str(d)] += 1
        
        # Display
        st.markdown("### Số Gan Nhất (Giảm Dần)")
        sorted_gaps = sorted(gaps.items(), key=lambda x: x[1], reverse=True)
        
        for digit, gap in sorted_gaps:
            if gap > 0:
                color = '#ff0040' if gap > 10 else ('#ffff00' if gap > 5 else '#00ffff')
                st.markdown(f"""
                <div class="metric-box" style="border-color:{color};">
                    <div style="font-size:24px; font-weight:900; color:{color};">Số {digit}</div>
                    <div style="font-size:16px; color:#888; margin-top:5px;">{gap} kỳ chưa về</div>
                </div>
                """, unsafe_allow_html=True)

# === TAB 5: LỊCH SỬ ===
elif st.session_state.current_tab == 'lịch':
    st.markdown("### 📊 Lịch Sử Dự Đoán")
    
    if db_manager.data['predictions']:
        df = pd.DataFrame(db_manager.data['predictions'][:50])
        
        # Format
        if not df.empty:
            df['date'] = pd.to_datetime(df['date']).dt.strftime('%Y-%m-%d %H:%M')
            df = df.rename(columns={
                'date': 'Thời Gian',
                'result_number': 'Kết Quả',
                'predicted': 'Dự Đoán',
                'result': 'Kết Quả',
                'confidence': 'Độ Tin'
            })
            
            def color_result(val):
                if val == 'win':
                    return 'color: #00ff00; font-weight: 900'
                return 'color: #ff0040; font-weight: 900'
            
            st.dataframe(
                df.style.applymap(color_result, subset=['Kết Quả']),
                use_container_width=True,
                hide_index=True
            )
        
        # Stats
        perf = db_manager.data['performance']
        st.markdown(f"""
        <div class="metric-box">
            <div style="font-size:14px; color:#888;">THỐNG KÊ TỔNG QUÁT</div>
            <div style="display:grid; grid-template-columns:1fr 1fr; gap:15px; margin-top:10px;">
                <div>
                    <div style="font-size:12px; color:#888;">Tổng Kỳ</div>
                    <div style="font-size:28px; font-weight:900; color:#00ffff;">{perf['total']}</div>
                </div>
                <div>
                    <div style="font-size:12px; color:#888;">Thắng</div>
                    <div style="font-size:28px; font-weight:900; color:#00ff00;">{perf['wins']}</div>
                </div>
            </div>
            <div class="confidence-bar" style="margin-top:15px;">
                <div class="confidence-fill" style="width:{perf['win_rate']*100}%; background:{'#00ff00' if perf['win_rate'] >= 0.4 else '#ff0040'};">
                    {perf['win_rate']*100:.1f}%
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.info("📭 Chưa có lịch sử dự đoán")

# === FOOTER ===
st.markdown("""
<div style="text-align:center; color:#444; font-size:10px; margin-top:30px; padding-top:15px; border-top:1px solid #333;">
    TITAN V52 - QUANTUM AI | AI Ensemble Learning | Self-Optimizing<br>
    <i>Lưu ý: Tool hỗ trợ phân tích - Không đảm bảo 100% - Quản lý vốn thông minh</i>
</div>
""", unsafe_allow_html=True)