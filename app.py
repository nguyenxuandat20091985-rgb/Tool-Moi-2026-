import streamlit as st
import json
import os
import numpy as np
import pandas as pd
from collections import Counter, defaultdict
from itertools import combinations
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings('ignore')

# ==================== CẤU HÌNH ====================
NVIDIA_API_KEY = "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"
GEMINI_API_KEY = "AIzaSyD1-XMO6FsA9ZgAf2P6nIiXLPp8moTPMrc"
DB_FILE = "titan_v32_data.json"

# ==================== DATA MANAGEMENT ====================
class DataManager:
    def __init__(self, db_file):
        self.db_file = db_file
        self.data = self.load_data()
    
    def load_data(self):
        if os.path.exists(self.db_file):
            with open(self.db_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        return {
            'results': [],
            'predictions': [],
            'model_weights': {
                'frequency': 0.3,
                'markov': 0.3,
                'bayesian': 0.2,
                'pattern': 0.2
            },
            'accuracy_history': []
        }
    
    def save_data(self):
        with open(self.db_file, 'w', encoding='utf-8') as f:
            json.dump(self.data, f, indent=2, ensure_ascii=False)
    
    def add_result(self, number):
        if number not in self.data['results']:
            self.data['results'].append(number)
            self.save_data()
    
    def add_prediction(self, pred_pair, actual_number, is_correct):
        self.data['predictions'].append({
            'prediction': pred_pair,
            'actual': actual_number,
            'correct': is_correct,
            'timestamp': len(self.data['predictions'])
        })
        self.data['accuracy_history'].append(is_correct)
        if len(self.data['accuracy_history']) > 100:
            self.data['accuracy_history'] = self.data['accuracy_history'][-100:]
        self.save_data()
    
    def update_weights(self, strategy, performance):
        current = self.data['model_weights'][strategy]
        if performance > 0.4:
            self.data['model_weights'][strategy] = min(0.5, current + 0.05)
        else:
            self.data['model_weights'][strategy] = max(0.1, current - 0.05)
        self.save_data()

# ==================== AI ENGINES ====================
class FrequencyModel:
    def __init__(self, data):
        self.data = data
        self.weights = {}
    
    def train(self):
        counter = Counter()
        for num in self.data:
            digits = set(num)
            for d in digits:
                counter[d] += 1
        
        total = len(self.data)
        self.weights = {d: count/total for d, count in counter.items()}
        return self.weights
    
    def predict(self, top_n=2):
        if not self.weights:
            self.train()
        sorted_digits = sorted(self.weights.items(), key=lambda x: x[1], reverse=True)
        return [d for d, _ in sorted_digits[:top_n]], self.weights

class MarkovChain:
    def __init__(self, data):
        self.data = data
        self.transition_matrix = defaultdict(lambda: defaultdict(int))
    
    def train(self):
        for i in range(len(self.data) - 1):
            curr_digits = set(self.data[i])
            next_digits = set(self.data[i+1])
            
            for d1 in curr_digits:
                for d2 in next_digits:
                    self.transition_matrix[d1][d2] += 1
        
        # Normalize
        for d1 in self.transition_matrix:
            total = sum(self.transition_matrix[d1].values())
            if total > 0:
                for d2 in self.transition_matrix[d1]:
                    self.transition_matrix[d1][d2] /= total
    
    def predict(self, last_number, top_n=2):
        if not self.transition_matrix:
            self.train()
        
        last_digits = set(last_number)
        combined_probs = defaultdict(float)
        
        for d in last_digits:
            if d in self.transition_matrix:
                for next_d, prob in self.transition_matrix[d].items():
                    combined_probs[next_d] += prob
        
        if not combined_probs:
            return ['0', '1']
        
        sorted_digits = sorted(combined_probs.items(), key=lambda x: x[1], reverse=True)
        return [d for d, _ in sorted_digits[:top_n]], combined_probs

class BayesianModel:
    def __init__(self, data):
        self.data = data
        self.prior = {}
        self.likelihood = defaultdict(lambda: defaultdict(float))
    
    def train(self):
        # Calculate prior probability for each digit
        counter = Counter()
        for num in self.data:
            for d in set(num):
                counter[d] += 1
        
        total_nums = len(self.data)
        self.prior = {d: count/total_nums for d, count in counter.items()}
        
        # Calculate likelihood (digit appears given position patterns)
        for num in self.data:
            digits = list(set(num))
            for i, d in enumerate(digits):
                if i+1 < len(digits):
                    self.likelihood[d][digits[i+1]] += 1
        
        # Normalize likelihood
        for d in self.likelihood:
            total = sum(self.likelihood[d].values())
            if total > 0:
                for next_d in self.likelihood[d]:
                    self.likelihood[d][next_d] /= total
    
    def predict(self, last_number, top_n=2):
        if not self.prior:
            self.train()
        
        last_digits = set(last_number)
        posterior = {}
        
        for d in '0123456789':
            # Prior
            prior_prob = self.prior.get(d, 0.01)
            
            # Likelihood from last digits
            likelihood = 1.0
            for ld in last_digits:
                if ld in self.likelihood and d in self.likelihood[ld]:
                    likelihood *= self.likelihood[ld][d]
                else:
                    likelihood *= 0.1
            
            posterior[d] = prior_prob * likelihood
        
        sorted_digits = sorted(posterior.items(), key=lambda x: x[1], reverse=True)
        return [d for d, _ in sorted_digits[:top_n]], posterior

class PatternDetector:
    def __init__(self, data):
        self.data = data
        self.patterns = {
            'hot_numbers': [],
            'cold_numbers': [],
            'pairs': Counter(),
            'cycles': {}
        }
    
    def analyze(self):
        if len(self.data) < 10:
            return self.patterns
        
        # Hot/Cold numbers (last 20 results)
        recent = self.data[-20:] if len(self.data) >= 20 else self.data
        counter = Counter()
        for num in recent:
            for d in set(num):
                counter[d] += 1
        
        sorted_nums = sorted(counter.items(), key=lambda x: x[1], reverse=True)
        self.patterns['hot_numbers'] = [d for d, _ in sorted_nums[:3]]
        self.patterns['cold_numbers'] = [d for d in '0123456789' if d not in counter]
        
        # Pairs frequency
        for num in self.data:
            digits = sorted(set(num))
            for pair in combinations(digits, 2):
                self.patterns['pairs'][pair] += 1
        
        # Cycle detection
        for d in '0123456789':
            gaps = []
            last_idx = -1
            for i, num in enumerate(self.data):
                if d in num:
                    if last_idx != -1:
                        gaps.append(i - last_idx)
                    last_idx = i
            if gaps:
                self.patterns['cycles'][d] = int(np.mean(gaps))
        
        return self.patterns
    
    def predict(self, top_n=2):
        if not self.patterns['hot_numbers']:
            self.analyze()
        
        # Score each digit
        scores = {}
        for d in '0123456789':
            score = 0
            
            # Hot number bonus
            if d in self.patterns['hot_numbers']:
                score += 3
            
            # Cold number penalty
            if d in self.patterns['cold_numbers']:
                score -= 2
            
            # Cycle analysis
            if d in self.patterns['cycles']:
                avg_gap = self.patterns['cycles'][d]
                if avg_gap <= 3:
                    score += 2  # Frequent
                elif avg_gap > 7:
                    score += 1  # Due to appear
        
        sorted_digits = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return [d for d, _ in sorted_digits[:top_n]], scores

# ==================== MAIN AI ENGINE ====================
class TitanAI:
    def __init__(self, data_manager):
        self.dm = data_manager
        self.results = data_manager.data['results']
        self.weights = data_manager.data['model_weights']
        
        self.freq_model = FrequencyModel(self.results)
        self.markov_model = MarkovChain(self.results)
        self.bayesian_model = BayesianModel(self.results)
        self.pattern_detector = PatternDetector(self.results)
        
        self.ensemble_model = None
        self._train_ensemble()
    
    def _train_ensemble(self):
        if len(self.results) < 20:
            return
        
        # Prepare training data
        X = []
        y = []
        
        for i in range(len(self.results) - 1):
            curr = self.results[i]
            next_num = self.results[i+1]
            next_digits = set(next_num)
            
            # Features: current digits + frequency features
            curr_digits = list(curr)
            features = [int(d) for d in curr_digits]
            
            # Add frequency features
            freq = self.freq_model.train()
            for d in '0123456789':
                features.append(freq.get(d, 0))
            
            # Labels: which digits appear next
            label = [1 if d in next_digits else 0 for d in '0123456789']
            
            X.append(features)
            y.append(label)
        
        if len(X) > 10:
            X = np.array(X)
            y = np.array(y)
            
            self.ensemble_model = GradientBoostingClassifier(n_estimators=50, max_depth=3, random_state=42)
            # Train for each digit position
            self.digit_classifiers = []
            for i in range(10):
                clf = RandomForestClassifier(n_estimators=30, max_depth=5, random_state=42)
                clf.fit(X, y[:, i])
                self.digit_classifiers.append(clf)
    
    def predict(self, last_number=None):
        if not last_number and self.results:
            last_number = self.results[-1]
        elif not last_number:
            return None, {}
        
        # Get predictions from each model
        freq_pred, freq_scores = self.freq_model.predict(5)
        markov_pred, markov_scores = self.markov_model.predict(last_number, 5)
        bayesian_pred, bayesian_scores = self.bayesian_model.predict(last_number, 5)
        pattern_pred, pattern_scores = self.pattern_detector.predict(5)
        
        # Ensemble scoring
        final_scores = defaultdict(float)
        
        for d in '0123456789':
            score = 0
            
            # Frequency model
            if d in freq_scores:
                score += self.weights['frequency'] * freq_scores[d] * 10
            
            # Markov model
            if d in markov_scores:
                score += self.weights['markov'] * markov_scores[d] * 10
            
            # Bayesian model
            if d in bayesian_scores:
                score += self.weights['bayesian'] * bayesian_scores[d] * 10
            
            # Pattern model
            if d in pattern_scores:
                score += self.weights['pattern'] * pattern_scores[d]
            
            # ML ensemble
            if self.digit_classifiers:
                try:
                    curr_digits = list(last_number)
                    features = [int(dig) for dig in curr_digits]
                    freq = self.freq_model.train()
                    for digit in '0123456789':
                        features.append(freq.get(digit, 0))
                    
                    prob = self.digit_classifiers[int(d)].predict_proba([features])[0][1]
                    score += prob * 5
                except:
                    pass
            
            final_scores[d] = score
        
        # Sort and get top 2
        sorted_digits = sorted(final_scores.items(), key=lambda x: x[1], reverse=True)
        top_2 = [d for d, _ in sorted_digits[:2]]
        
        # Calculate confidence
        total_score = sum(final_scores.values())
        confidence = (final_scores[top_2[0]] + final_scores[top_2[1]]) / total_score * 100 if total_score > 0 else 50
        
        return top_2, {
            'scores': dict(final_scores),
            'freq_pred': freq_pred,
            'markov_pred': markov_pred,
            'bayesian_pred': bayesian_pred,
            'pattern_pred': pattern_pred,
            'confidence': min(95, max(30, confidence))
        }
    
    def learn(self, actual_number, predicted_pair):
        is_correct = all(d in actual_number for d in predicted_pair)
        self.dm.add_prediction(predicted_pair, actual_number, is_correct)
        
        # Update weights based on performance
        recent_accuracy = np.mean(self.dm.data['accuracy_history'][-20:]) if self.dm.data['accuracy_history'] else 0.5
        
        if is_correct and recent_accuracy < 0.4:
            # Boost all weights slightly
            for strategy in self.weights:
                self.weights[strategy] = min(0.5, self.weights[strategy] + 0.02)
        elif not is_correct and recent_accuracy > 0.6:
            # Reduce weights slightly
            for strategy in self.weights:
                self.weights[strategy] = max(0.1, self.weights[strategy] - 0.02)
        
        self.dm.data['model_weights'] = self.weights
        self.dm.save_data()
        
        # Retrain if enough new data
        if len(self.results) % 10 == 0:
            self._train_ensemble()
        
        return is_correct, recent_accuracy

# ==================== UI COMPONENTS ====================
def render_header():
    st.markdown("""
    <style>
    .main-header {
        font-size: 2.5em;
        font-weight: bold;
        background: linear-gradient(90deg, #00ffff, #ff00ff, #00ffff);
        background-size: 200% auto;
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        animation: shine 3s linear infinite;
        text-align: center;
    }
    @keyframes shine {
        to { background-position: 200% center; }
    }
    .metric-box {
        background: linear-gradient(135deg, #1a1a2e, #0f0f1e);
        border: 1px solid #00ffff;
        border-radius: 10px;
        padding: 15px;
        margin: 10px 0;
        text-align: center;
    }
    .prediction-card {
        background: linear-gradient(135deg, #0f0f1e, #1a1a2e);
        border: 2px solid #ff00ff;
        border-radius: 15px;
        padding: 20px;
        margin: 15px 0;
        text-align: center;
    }
    .big-number {
        font-size: 3em;
        font-weight: bold;
        color: #00ffff;
        text-shadow: 0 0 20px #00ffff;
    }
    .tag {
        display: inline-block;
        padding: 5px 10px;
        border-radius: 15px;
        font-size: 0.8em;
        margin: 3px;
        font-weight: bold;
    }
    .tag-hot { background: #ff0040; color: white; }
    .tag-cold { background: #0066ff; color: white; }
    .tag-gold { background: #ffd700; color: black; }
    </style>
    """, unsafe_allow_html=True)
    
    st.markdown('<div class="main-header">🧬 TITAN V52 - NEURAL PREDICTOR</div>', unsafe_allow_html=True)
    st.markdown('<p style="text-align:center; color:#888;">AI-Powered 5D Lottery Prediction System</p>', unsafe_allow_html=True)

def render_metrics(ai_engine, dm):
    pattern = ai_engine.pattern_detector.analyze()
    accuracy = np.mean(dm.data['accuracy_history'][-20:]) if dm.data['accuracy_history'] else 0
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(f"""
        <div class="metric-box">
            <div style="color:#888; font-size:0.9em;">ĐỘ CHÍNH XÁC</div>
            <div style="font-size:1.8em; font-weight:bold; color:{'#00ff00' if accuracy > 0.4 else '#ff0040'}">
                {accuracy*100:.1f}%
            </div>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown(f"""
        <div class="metric-box">
            <div style="color:#888; font-size:0.9em;">SỐ KỲ</div>
            <div style="font-size:1.8em; font-weight:bold; color:#00ffff">
                {len(dm.data['results'])}
            </div>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        hot_str = ', '.join(pattern['hot_numbers'][:3]) if pattern['hot_numbers'] else '-'
        st.markdown(f"""
        <div class="metric-box">
            <div style="color:#888; font-size:0.9em;">SỐ NÓNG</div>
            <div style="font-size:1.5em; font-weight:bold; color:#ff0040">
                {hot_str}
            </div>
        </div>
        """, unsafe_allow_html=True)

def render_prediction_section(ai_engine, dm):
    st.markdown('<div style="margin-top:30px;"></div>', unsafe_allow_html=True)
    st.markdown('<h2 style="color:#00ffff; text-align:center;">🎯 DỰ ĐOÁN TIẾP THEO</h2>')
    
    if len(dm.data['results']) < 5:
        st.warning("Cần ít nhất 5 kỳ để dự đoán")
        return
    
    last_num = dm.data['results'][-1]
    prediction, details = ai_engine.predict(last_num)
    
    if prediction:
        st.markdown(f"""
        <div class="prediction-card">
            <div style="color:#888; margin-bottom:10px;">CẶP SỐ VIP (ĐỘ TIN CẬY: {details['confidence']:.1f}%)</div>
            <div class="big-number">{prediction[0]} - {prediction[1]}</div>
            <div style="margin-top:15px;">
                <span class="tag tag-gold">FREQUENCY: {', '.join(details['freq_pred'][:2])}</span>
                <span class="tag tag-gold">MARKOV: {', '.join(details['markov_pred'][:2])}</span>
                <span class="tag tag-gold">BAYESIAN: {', '.join(details['bayesian_pred'][:2])}</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Heatmap
        st.markdown('<h3 style="color:#ff00ff;">📊 HEATMAP TẦN SUẤT</h3>')
        
        scores = details['scores']
        max_score = max(scores.values()) if scores else 1
        
        heatmap_cols = st.columns(10)
        for i, col in enumerate(heatmap_cols):
            digit = str(i)
            score = scores.get(digit, 0)
            intensity = score / max_score if max_score > 0 else 0
            
            color = f'rgba(0, 255, 255, {0.2 + intensity * 0.8})'
            
            with col:
                st.markdown(f"""
                <div style="background:{color}; border-radius:10px; padding:15px 5px; text-align:center; 
                            border:1px solid #00ffff; margin:2px;">
                    <div style="font-size:1.5em; font-weight:bold; color:#fff;">{digit}</div>
                    <div style="font-size:0.7em; color:#aaa;">{score:.2f}</div>
                </div>
                """, unsafe_allow_html=True)

def render_history_section(dm):
    st.markdown('<h2 style="color:#00ffff; text-align:center;">📋 LỊCH SỬ DỰ ĐOÁN</h2>')
    
    if not dm.data['predictions']:
        st.info("Chưa có lịch sử dự đoán")
        return
    
    df = pd.DataFrame(dm.data['predictions'][-15:])
    df.columns = ['Dự đoán', 'Thực tế', 'Kết quả', 'STT']
    
    def color_result(val):
        return 'color: #00ff00; font-weight: bold' if val == True else 'color: #ff0040; font-weight: bold'
    
    st.dataframe(
        df.style.applymap(color_result, subset=['Kết quả']),
        use_container_width=True,
        hide_index=True
    )

def main():
    render_header()
    
    # Initialize
    dm = DataManager(DB_FILE)
    
    # Sidebar
    with st.sidebar:
        st.markdown('<h3 style="color:#00ffff;">⚙️ CHỨC NĂNG</h3>', unsafe_allow_html=True)
        
        menu = st.selectbox("Chọn chức năng", [
            "📥 Nhập kết quả",
            "🎯 Dự đoán",
            "📊 Phân tích",
            "📋 Lịch sử"
        ])
        
        st.markdown('<div style="margin-top:30px;"></div>', unsafe_allow_html=True)
        st.markdown('<h3 style="color:#ff00ff;">📊 THỐNG KÊ</h3>', unsafe_allow_html=True)
        st.write(f"Tổng kỳ: {len(dm.data['results'])}")
        st.write(f"Dự đoán: {len(dm.data['predictions'])}")
        
        if dm.data['accuracy_history']:
            acc = np.mean(dm.data['accuracy_history'][-20:]) * 100
            st.write(f"Độ chính xác (20 kỳ): {acc:.1f}%")
    
    # Main content
    if menu == "📥 Nhập kết quả":
        st.markdown('<h2 style="color:#00ffff;">📥 NHẬP KẾT QUẢ MỚI</h2>')
        
        input_num = st.text_input("Nhập số 5 chữ số:", max_chars=5, placeholder="Ví dụ: 12345")
        
        if st.button("Lưu kết quả", type="primary"):
            if len(input_num) == 5 and input_num.isdigit():
                dm.add_result(input_num)
                st.success(f"Đã lưu: {input_num}")
                st.rerun()
            else:
                st.error("Vui lòng nhập đúng 5 chữ số")
        
        # Bulk import
        st.markdown('<div style="margin-top:20px;"></div>', unsafe_allow_html=True)
        bulk_input = st.text_area("Nhập nhiều kết quả (mỗi số 1 dòng):", height=150)
        
        if st.button("Nhập hàng loạt"):
            lines = bulk_input.strip().split('\n')
            count = 0
            for line in lines:
                num = line.strip()
                if len(num) == 5 and num.isdigit():
                    dm.add_result(num)
                    count += 1
            st.success(f"Đã nhập {count} kết quả")
            st.rerun()
    
    elif menu == "🎯 Dự đoán":
        ai_engine = TitanAI(dm)
        render_metrics(ai_engine, dm)
        render_prediction_section(ai_engine, dm)
        
        # Verify prediction
        st.markdown('<div style="margin-top:30px;"></div>', unsafe_allow_html=True)
        st.markdown('<h3 style="color:#ff00ff;">✓ XÁC NHẬN KẾT QUẢ</h3>')
        
        actual_num = st.text_input("Nhập kết quả thực tế để học:", max_chars=5, key="verify")
        
        if st.button("Xác nhận và học"):
            if len(actual_num) == 5 and actual_num.isdigit():
                dm.add_result(actual_num)
                ai_engine_new = TitanAI(dm)
                
                if len(dm.data['predictions']) > 0:
                    last_pred = dm.data['predictions'][-1]['prediction']
                    is_correct, accuracy = ai_engine_new.learn(actual_num, last_pred)
                    
                    if is_correct:
                        st.success("✅ DỰ ĐOÁN ĐÚNG!")
                    else:
                        st.error("❌ Dự đoán sai - AI đang học...")
                    
                    st.info(f"Độ chính xác hiện tại: {accuracy*100:.1f}%")
                    st.rerun()
            else:
                st.error("Vui lòng nhập đúng 5 chữ số")
    
    elif menu == "📊 Phân tích":
        if len(dm.data['results']) < 5:
            st.warning("Cần ít nhất 5 kỳ để phân tích")
        else:
            ai_engine = TitanAI(dm)
            pattern = ai_engine.pattern_detector.analyze()
            
            st.markdown('<h2 style="color:#00ffff;">📊 PHÂN TÍCH CHUYÊN SÂU</h2>')
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown('<h3 style="color:#ff0040;">🔥 SỐ NÓNG</h3>')
                if pattern['hot_numbers']:
                    for num in pattern['hot_numbers']:
                        st.markdown(f'<div class="tag tag-hot" style="font-size:1.2em; padding:10px 20px;">{num}</div>', 
                                   unsafe_allow_html=True)
                
                st.markdown('<h3 style="color:#0066ff;">❄️ SỐ LẠNH</h3>')
                if pattern['cold_numbers']:
                    for num in pattern['cold_numbers'][:5]:
                        st.markdown(f'<div class="tag tag-cold" style="font-size:1.2em; padding:10px 20px;">{num}</div>', 
                                   unsafe_allow_html=True)
            
            with col2:
                st.markdown('<h3 style="color:#ffd700;">📈 CHU KỲ TRUNG BÌNH</h3>')
                cycles_df = pd.DataFrame([
                    {'Số': d, 'Chu kỳ': gap} 
                    for d, gap in pattern['cycles'].items()
                ]).sort_values('Chu kỳ')
                st.dataframe(cycles_df, use_container_width=True, hide_index=True)
            
            # Top pairs
            st.markdown('<div style="margin-top:30px;"></div>', unsafe_allow_html=True)
            st.markdown('<h3 style="color:#00ffff;">🎯 CẶP SỐ THƯỜNG ĐI CÙNG</h3>')
            
            top_pairs = pattern['pairs'].most_common(10)
            pairs_df = pd.DataFrame(top_pairs, columns=['Cặp số', 'Tần suất'])
            st.dataframe(pairs_df, use_container_width=True, hide_index=True)
    
    elif menu == "📋 Lịch sử":
        render_history_section(dm)

if __name__ == "__main__":
    # Initialize database with provided data
    if not os.path.exists(DB_FILE):
        dm = DataManager(DB_FILE)
        # Parse provided data
        raw_data = """46602 14606 97269 04675 98005 52064 60204 51253 89879 19626 34479 37882 76706 71199 41437 35732 69270 93401 03830 99143 22324 84058 41261 01837 20757 67623 38115 96989 50313 91658 21009 54530 51113 45990 45383 23348 68404 46966 23730 54173 13457 23273 11757 23713 89586 81262 16568 83337 55531 47512 92567 11574 40834 97148 00351 78384 98733 47336 62038 66409 92005 39956 33525 69645 40621 89859 20634 21030 02242 88621 45480 49297 98857 00931 73718 84901 33590 46670 60591 01846 23319 08367 89604 83057 97748 14930 31135 42501 81581 83847 31068 59397 79914 69161 52837 90177 04197 88288 10358 52863 73555 88003 41657 74513 47547 41852 43271 28910 76299 74258 64014 27019 57364 51937 79698 43314 22618 48063 76777 84412 93209 12780 95341 31092 67840 55533 62270 28491 16600 24318 93004 72530 70896 89962 78773 01811 83017 30445 32810 88461 99033 26323 33310 14489 44000 15512 41289 48859 39675 77471 52302 11211 41399 87159 17756 01823 10544 51038 05574 43850 93510 76370 20375 52909 73875 20966 57189 08514 52443 74478 02114 63318 01913 99597 74667 86844 14602 99034 21457 27650 71886 82019 33440 30703 97559 38798 95305 94210 25431 48363 57740 94925 34749 54759 01104 56509 05118 50388 22667 26314 63961 97212 93824 68430 23962 21109 34114 28664 28460 12020 13162 83240 02236 53900 15886 85826 08022 07431 53746 18761 94453 43060 12274 35461 07173 29349 43188 13971 27964 70377 43015 43254 79321 93170 66890 38385 29233 21009 82188 56942 29537 87558 34979 87136 26404 71990 07298 00443 02917 28485 69200 57769 59597 13385 76881 87203 16695 87558 34979 87136 26404 71990 07298 00443 02917 28485 69200 57769 59597 13385 76881 87203 16695 37485 00325 94144 56726 20115 77579 38010 29580 52771 15670 21391"""
        
        numbers = raw_data.split()
        for num in numbers:
            if len(num) == 5:
                dm.add_result(num)
    
    main()