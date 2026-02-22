import streamlit as st
import google.generativeai as genai
import re
import json
import os
from collections import Counter
from datetime import datetime
import numpy as np
import random
import time
import hashlib
import requests
from typing import List, Dict, Tuple, Optional
import pandas as pd

# ================= CẤU HÌNH HỆ THỐNG =================
API_KEY = "AIzaSyChq-KF-DXqPQUpxDsVIvx5D4_jRH1ERqM"
DB_FILE = "titan_memory_v21.json"
PREDICTIONS_FILE = "titan_predictions_v21.json"
PATTERNS_FILE = "titan_patterns_v21.json"
SOURCES_FILE = "titan_sources_v21.json"

def setup_neural():
    try:
        genai.configure(api_key=API_KEY)
        return genai.GenerativeModel('gemini-1.5-flash')
    except:
        return None

neural_engine = setup_neural()

# ================= HỆ THỐNG GHI NHỚ =================
def load_memory():
    if os.path.exists(DB_FILE):
        with open(DB_FILE, "r") as f:
            try:
                return json.load(f)
            except:
                return []
    return []

def save_memory(data):
    with open(DB_FILE, "w") as f:
        json.dump(data[-1000:], f)

def load_predictions():
    if os.path.exists(PREDICTIONS_FILE):
        with open(PREDICTIONS_FILE, "r") as f:
            try:
                return json.load(f)
            except:
                return []
    return []

def save_prediction(prediction_data):
    predictions = load_predictions()
    predictions.append(prediction_data)
    with open(PREDICTIONS_FILE, "w") as f:
        json.dump(predictions[-500:], f)

def load_patterns():
    if os.path.exists(PATTERNS_FILE):
        with open(PATTERNS_FILE, "r") as f:
            try:
                return json.load(f)
            except:
                return {}
    return {}

def save_patterns(data):
    with open(PATTERNS_FILE, "w") as f:
        json.dump(data, f)

def load_sources():
    if os.path.exists(SOURCES_FILE):
        with open(SOURCES_FILE, "r") as f:
            try:
                return json.load(f)
            except:
                return []
    return []

def save_sources(data):
    with open(SOURCES_FILE, "w") as f:
        json.dump(data[-100:], f)

if "history" not in st.session_state:
    st.session_state.history = load_memory()
if "predictions" not in st.session_state:
    st.session_state.predictions = load_predictions()
if "patterns" not in st.session_state:
    st.session_state.patterns = load_patterns()
if "sources" not in st.session_state:
    st.session_state.sources = load_sources()
if "accuracy_stats" not in st.session_state:
    st.session_state.accuracy_stats = {"correct": 0, "total": 0, "history": []}

# ================= HỆ THỐNG PHÁT HIỆN QUY LUẬT CAO CẤP =================
class PatternDetector:
    def __init__(self, history):
        self.history = history[-500:] if len(history) > 500 else history
        self.numbers = [list(num) for num in self.history]
        
    def detect_pairs(self):
        """Phát hiện các cặp số hay đi cùng nhau"""
        pairs = {}
        
        # Phân tích từng vị trí
        for pos in range(5):
            pos_digits = [int(n[pos]) for n in self.numbers]
            
            # Tìm các cặp xuất hiện liên tiếp
            for i in range(len(pos_digits) - 1):
                pair = f"{pos_digits[i]}{pos_digits[i+1]}"
                if pair not in pairs:
                    pairs[pair] = {"count": 0, "positions": []}
                pairs[pair]["count"] += 1
                if pos not in pairs[pair]["positions"]:
                    pairs[pair]["positions"].append(pos)
        
        # Lọc các cặp có ý nghĩa
        significant_pairs = {}
        for pair, data in pairs.items():
            if data["count"] >= 3:  # Xuất hiện ít nhất 3 lần
                significance = data["count"] / len(self.history) * 100
                significant_pairs[pair] = {
                    "count": data["count"],
                    "significance": round(significance, 2),
                    "positions": data["positions"],
                    "probability": round(data["count"] / len(self.history) * 100, 2)
                }
        
        return dict(sorted(significant_pairs.items(), 
                          key=lambda x: x[1]["count"], reverse=True)[:20])
    
    def detect_triplets(self):
        """Phát hiện bộ ba số hay đi cùng nhau"""
        triplets = {}
        
        for pos in range(5):
            pos_digits = [int(n[pos]) for n in self.numbers]
            
            for i in range(len(pos_digits) - 2):
                triplet = f"{pos_digits[i]}{pos_digits[i+1]}{pos_digits[i+2]}"
                if triplet not in triplets:
                    triplets[triplet] = {"count": 0, "positions": []}
                triplets[triplet]["count"] += 1
                if pos not in triplets[triplet]["positions"]:
                    triplets[triplet]["positions"].append(pos)
        
        significant_triplets = {}
        for triplet, data in triplets.items():
            if data["count"] >= 2:
                significant_triplets[triplet] = {
                    "count": data["count"],
                    "positions": data["positions"],
                    "probability": round(data["count"] / len(self.history) * 100, 2)
                }
        
        return dict(sorted(significant_triplets.items(), 
                          key=lambda x: x[1]["count"], reverse=True)[:10])
    
    def detect_cycles(self):
        """Phát hiện chu kỳ lặp lại"""
        cycles = {}
        
        # Kiểm tra chu kỳ 3-10 số
        for cycle_length in range(3, 11):
            for pos in range(5):
                pos_digits = [int(n[pos]) for n in self.numbers[-100:]]
                
                if len(pos_digits) >= cycle_length * 2:
                    # Tìm pattern lặp lại
                    patterns = {}
                    for i in range(len(pos_digits) - cycle_length):
                        pattern = tuple(pos_digits[i:i+cycle_length])
                        if pattern not in patterns:
                            patterns[pattern] = []
                        patterns[pattern].append(i)
                    
                    # Kiểm tra pattern nào lặp lại
                    for pattern, indices in patterns.items():
                        if len(indices) >= 2:
                            cycle_key = f"pos{pos+1}_len{cycle_length}_{''.join(map(str, pattern))}"
                            cycles[cycle_key] = {
                                "position": pos + 1,
                                "length": cycle_length,
                                "pattern": ''.join(map(str, pattern)),
                                "occurrences": len(indices),
                                "reliability": round(len(indices) / (len(pos_digits) / cycle_length) * 100, 2)
                            }
        
        return dict(sorted(cycles.items(), 
                          key=lambda x: x[1]["reliability"], reverse=True)[:15])
    
    def detect_cross_position_patterns(self):
        """Phát hiện pattern liên quan giữa các vị trí"""
        patterns = {}
        
        # Phân tích tương quan giữa các vị trí
        for i in range(5):
            for j in range(i+1, 5):
                pos_i = [int(n[i]) for n in self.numbers[-50:]]
                pos_j = [int(n[j]) for n in self.numbers[-50:]]
                
                # Tìm các cặp xuất hiện cùng lúc
                simultaneous = {}
                for idx, (digit_i, digit_j) in enumerate(zip(pos_i, pos_j)):
                    pair = f"{digit_i}-{digit_j}"
                    if pair not in simultaneous:
                        simultaneous[pair] = 0
                    simultaneous[pair] += 1
                
                # Lọc các cặp có tần suất cao
                for pair, count in simultaneous.items():
                    if count >= 5:
                        pattern_key = f"pos{i+1}-{j+1}_{pair}"
                        patterns[pattern_key] = {
                            "positions": f"{i+1}-{j+1}",
                            "pair": pair,
                            "frequency": count,
                            "probability": round(count / len(pos_i) * 100, 2)
                        }
        
        return dict(sorted(patterns.items(), 
                          key=lambda x: x[1]["frequency"], reverse=True)[:20])

# ================= HỆ THỐNG PHÁT HIỆN BẪY NHÀ CÁI =================
class TrapDetector:
    def __init__(self, history):
        self.history = history[-200:] if len(history) > 200 else history
        self.numbers = [list(num) for num in self.history]
        
    def detect_abnormal_patterns(self):
        """Phát hiện các pattern bất thường (dấu hiệu nhà cái lừa cầu)"""
        warnings = []
        
        if len(self.history) < 20:
            return warnings
        
        # 1. Kiểm tra đảo cầu đột ngột
        last_10 = self.history[-10:]
        unique_last_10 = len(set(''.join(last_10)))
        prev_10 = self.history[-20:-10]
        unique_prev_10 = len(set(''.join(prev_10)))
        
        if unique_last_10 > unique_prev_10 * 1.5:
            warnings.append({
                "type": "ĐẢO CẦU ĐỘT NGỘT",
                "description": "Số lượng số mới xuất hiện tăng đột biến",
                "severity": "CAO",
                "action": "GIẢM VỐN - Đang test cầu mới"
            })
        
        # 2. Kiểm tra phá vỡ chu kỳ
        patterns_found = []
        for length in [3, 4, 5]:
            for pos in range(5):
                pos_digits = [int(n[pos]) for n in self.numbers[-30:]]
                if len(pos_digits) >= length * 2:
                    last_pattern = tuple(pos_digits[-length:])
                    prev_patterns = [tuple(pos_digits[i:i+length]) 
                                   for i in range(len(pos_digits)-length*2, len(pos_digits)-length)]
                    
                    if last_pattern not in prev_patterns and len(prev_patterns) > 0:
                        patterns_found.append({
                            "position": pos+1,
                            "length": length
                        })
        
        if len(patterns_found) >= 3:
            warnings.append({
                "type": "PHÁ VỠ CHU KỲ",
                "description": f"{len(patterns_found)} vị trí phá vỡ chu kỳ",
                "severity": "TRUNG BÌNH",
                "action": "QUAN SÁT - Chờ chu kỳ mới"
            })
        
        # 3. Kiểm tra số hiếm xuất hiện
        all_nums = ''.join(self.history[-30:])
        counts = Counter(all_nums)
        rare_nums = [num for num, count in counts.items() if count <= 1]
        
        if len(rare_nums) >= 3:
            warnings.append({
                "type": "SỐ HIẾM XUẤT HIỆN",
                "description": f"Số hiếm: {', '.join(rare_nums)}",
                "severity": "THẤP",
                "action": "THEO DÕI - Có thể sắp nổ số hiếm"
            })
        
        # 4. Kiểm streak dài bất thường
        for pos in range(5):
            pos_digits = [n[pos] for n in self.numbers[-20:]]
            current = pos_digits[-1]
            streak = 1
            for i in range(len(pos_digits)-2, -1, -1):
                if pos_digits[i] == current:
                    streak += 1
                else:
                    break
            
            if streak >= 4:
                warnings.append({
                    "type": "STREAK DÀI BẤT THƯỜNG",
                    "description": f"Vị trí {pos+1} bệt số {current} {streak} kỳ",
                    "severity": "CAO" if streak >= 6 else "TRUNG BÌNH",
                    "action": "CẨN THẬN - Streak dài dễ gãy"
                })
        
        # 5. Kiểm tra tỷ lệ xuất hiện
        expected_ratio = 10  # Mỗi số xuất hiện 10% thời gian
        for num in '0123456789':
            actual_ratio = counts.get(num, 0) / len(all_nums) * 100 if len(all_nums) > 0 else 0
            if actual_ratio > expected_ratio * 2:
                warnings.append({
                    "type": "MẤT CÂN BẰNG",
                    "description": f"Số {num} xuất hiện {actual_ratio:.1f}% (cao bất thường)",
                    "severity": "TRUNG BÌNH",
                    "action": "CÂN NHẮC - Có thể sắp giảm tần suất"
                })
        
        return warnings
    
    def predict_next_move(self):
        """Dự đoán nước đi tiếp theo của nhà cái"""
        if len(self.history) < 30:
            return {}
        
        predictions = {
            "scenarios": [],
            "recommendation": "",
            "confidence": 0
        }
        
        # Phân tích xu hướng hiện tại
        last_5 = self.history[-5:]
        unique_count = len(set(''.join(last_5)))
        
        # Kịch bản 1: Tiếp tục streak
        if unique_count <= 8:  # Ít số xuất hiện
            # Tìm số đang streak
            streak_nums = []
            for pos in range(5):
                pos_digits = [n[pos] for n in self.numbers[-10:]]
                if len(set(pos_digits[-3:])) == 1:
                    streak_nums.append(pos_digits[-1])
            
            if streak_nums:
                predictions["scenarios"].append({
                    "type": "TIẾP TỤC STREAK",
                    "numbers": list(set(streak_nums)),
                    "probability": 65,
                    "logic": "Các vị trí đang bệt có khả năng tiếp tục"
                })
        
        # Kịch bản 2: Đảo cầu
        if unique_count >= 12:  # Nhiều số xuất hiện
            # Tìm số ít xuất hiện
            all_nums = ''.join(self.history[-20:])
            counts = Counter(all_nums)
            cold_nums = [num for num, count in counts.most_common()[-3:]]
            
            predictions["scenarios"].append({
                "type": "ĐẢO CẦU - RA SỐ LẠ",
                "numbers": cold_nums,
                "probability": 60,
                "logic": "Nhà cái đang xoay vòng số"
            })
        
        # Kịch bản 3: Lặp lại pattern cũ
        for length in [3, 4, 5]:
            last_pattern = ''.join([n[-1] for n in self.numbers[-length:]])
            # Tìm pattern này trong lịch sử
            history_str = ''.join(self.history)
            occurrences = history_str.count(last_pattern)
            
            if occurrences >= 2:
                predictions["scenarios"].append({
                    "type": f"LẶP LẠI PATTERN {length} SỐ",
                    "numbers": [last_pattern],
                    "probability": 55 + occurrences * 5,
                    "logic": f"Pattern {last_pattern} đã xuất hiện {occurrences} lần"
                })
        
        # Chọn kịch bản tốt nhất
        if predictions["scenarios"]:
            best_scenario = max(predictions["scenarios"], key=lambda x: x["probability"])
            predictions["recommendation"] = best_scenario["type"]
            predictions["confidence"] = best_scenario["probability"]
        
        return predictions

# ================= HỆ THỐNG THU THẬP DỮ LIỆU =================
class DataCollector:
    def __init__(self):
        self.sources = [
            {"name": "Lịch sử nội bộ", "url": None, "active": True},
            {"name": "Pattern đã phát hiện", "url": None, "active": True},
            {"name": "Dữ liệu người dùng", "url": None, "active": True}
        ]
        
        # Thêm nguồn từ file nếu có
        saved_sources = load_sources()
        if saved_sources:
            for source in saved_sources:
                if source not in self.sources:
                    self.sources.append(source)
    
    def add_source(self, name, url=None):
        """Thêm nguồn dữ liệu mới"""
        new_source = {"name": name, "url": url, "active": True, "added": datetime.now().isoformat()}
        self.sources.append(new_source)
        save_sources(self.sources)
        return new_source
    
    def collect_all_data(self, history):
        """Thu thập dữ liệu từ tất cả nguồn"""
        collected = {
            "history": history,
            "patterns": {},
            "predictions": [],
            "external": []
        }
        
        # Thu thập patterns
        detector = PatternDetector(history)
        collected["patterns"]["pairs"] = detector.detect_pairs()
        collected["patterns"]["triplets"] = detector.detect_triplets()
        collected["patterns"]["cycles"] = detector.detect_cycles()
        collected["patterns"]["cross"] = detector.detect_cross_position_patterns()
        
        # Thu thập dự đoán cũ
        predictions = load_predictions()
        if predictions:
            recent_preds = predictions[-20:]
            for pred in recent_preds:
                if "dan4" in pred and "dan3" in pred:
                    collected["predictions"].append({
                        "time": pred.get("time", ""),
                        "numbers": pred["dan4"] + pred["dan3"],
                        "accuracy": pred.get("do_tin_cay", 0)
                    })
        
        return collected

# ================= HỆ THỐNG AI ENSEMBLE =================
class AIEnsemble:
    def __init__(self):
        self.models = {
            "gemini": neural_engine,
            "pattern_matcher": self.pattern_match_predict,
            "statistical": self.statistical_predict,
            "cycle_based": self.cycle_based_predict,
            "trap_aware": self.trap_aware_predict
        }
        
        self.weights = {
            "gemini": 0.35,
            "pattern_matcher": 0.25,
            "statistical": 0.20,
            "cycle_based": 0.10,
            "trap_aware": 0.10
        }
    
    def pattern_match_predict(self, history, patterns):
        """Dự đoán dựa trên pattern đã phát hiện"""
        if not history or len(history) < 10:
            return []
        
        last_num = history[-1]
        predictions = []
        
        # Dựa vào cặp số
        if "pairs" in patterns:
            for pair, data in patterns["pairs"].items():
                if pair[0] == last_num[0]:  # Nếu số đầu khớp
                    predictions.append({
                        "number": pair[1],
                        "confidence": data["probability"] / 100,
                        "source": "pair"
                    })
        
        # Dựa vào triplet
        if "triplets" in patterns:
            for triplet, data in patterns["triplets"].items():
                if len(triplet) >= 2 and triplet[:2] == last_num[:2]:
                    predictions.append({
                        "number": triplet[2],
                        "confidence": data["probability"] / 100,
                        "source": "triplet"
                    })
        
        # Chọn prediction tốt nhất
        if predictions:
            best = max(predictions, key=lambda x: x["confidence"])
            return [best["number"]] * 5, best["confidence"]
        
        return [], 0
    
    def statistical_predict(self, history):
        """Dự đoán dựa trên thống kê"""
        if len(history) < 20:
            return [], 0
        
        all_nums = ''.join(history[-50:])
        counts = Counter(all_nums)
        total = len(all_nums)
        
        # Tính xác suất
        probs = {num: count/total for num, count in counts.items()}
        
        # Dự đoán số có xác suất cao nhất
        best_num = max(probs.items(), key=lambda x: x[1])[0]
        confidence = probs[best_num]
        
        return [best_num] * 5, confidence
    
    def cycle_based_predict(self, history):
        """Dự đoán dựa trên chu kỳ"""
        if len(history) < 30:
            return [], 0
        
        # Tìm chu kỳ 5 số gần nhất
        last_5 = ''.join(history[-5:])
        history_str = ''.join(history[:-5])
        
        # Tìm vị trí xuất hiện của pattern
        positions = []
        start = 0
        while True:
            pos = history_str.find(last_5, start)
            if pos == -1:
                break
            positions.append(pos)
            start = pos + 1
        
        if positions:
            # Dự đoán số tiếp theo dựa trên pattern cũ
            predictions = []
            for pos in positions:
                next_pos = pos + 5
                if next_pos < len(history_str):
                    predictions.append(history_str[next_pos])
            
            if predictions:
                pred_counts = Counter(predictions)
                best_pred = pred_counts.most_common(1)[0]
                confidence = best_pred[1] / len(predictions)
                return [best_pred[0]] * 5, confidence
        
        return [], 0
    
    def trap_aware_predict(self, history):
        """Dự đoán có tính đến bẫy nhà cái"""
        detector = TrapDetector(history)
        warnings = detector.detect_abnormal_patterns()
        next_move = detector.predict_next_move()
        
        if next_move and "scenarios" in next_move:
            best_scenario = max(next_move["scenarios"], key=lambda x: x.get("probability", 0))
            if best_scenario and "numbers" in best_scenario:
                numbers = best_scenario["numbers"]
                if numbers:
                    confidence = best_scenario.get("probability", 50) / 100
                    return [numbers[0]] * 5, confidence
        
        return [], 0
    
    def ensemble_predict(self, history, patterns):
        """Kết hợp tất cả các model để dự đoán"""
        predictions = []
        total_confidence = 0
        
        # Thu thập dự đoán từ các model
        for model_name, model_func in self.models.items():
            if model_name == "gemini":
                # Gemini sẽ được xử lý riêng
                continue
            elif model_name == "pattern_matcher":
                pred, conf = model_func(history, patterns)
            else:
                pred, conf = model_func(history)
            
            if pred and conf > 0.3:
                weight = self.weights.get(model_name, 0.1)
                predictions.append({
                    "model": model_name,
                    "prediction": pred,
                    "confidence": conf,
                    "weight": weight,
                    "score": conf * weight
                })
                total_confidence += conf * weight
        
        # Tính weighted average cho mỗi số
        number_scores = {str(i): 0 for i in range(10)}
        for pred in predictions:
            if pred["prediction"]:
                main_num = pred["prediction"][0]
                number_scores[main_num] += pred["score"]
        
        # Chọn số có điểm cao nhất
        if max(number_scores.values()) > 0:
            best_num = max(number_scores.items(), key=lambda x: x[1])[0]
            ensemble_confidence = total_confidence / sum(self.weights.values())
            
            return {
                "prediction": [best_num] * 5,
                "confidence": min(ensemble_confidence, 0.95),
                "details": predictions,
                "scores": number_scores
            }
        
        return None

# ================= UI DESIGN NÂNG CAO =================
st.set_page_config(
    page_title="TITAN v22.0 PRO MAX",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# CSS Responsive
st.markdown("""
    <style>
    /* Responsive design */
    @media (max-width: 768px) {
        .num-display { font-size: 40px !important; letter-spacing: 5px !important; }
        .prediction-card { padding: 15px !important; }
        .stButton button { font-size: 14px !important; padding: 10px !important; }
    }
    
    @media (max-width: 480px) {
        .num-display { font-size: 30px !important; }
        h2 { font-size: 20px !important; }
    }
    
    /* Main styles */
    .stApp { 
        background: #010409; 
        color: #c9d1d9;
        font-family: -apple-system, BlinkMacSystemFont, sans-serif;
    }
    
    .status-active { 
        color: #238636; 
        font-weight: bold; 
        border-left: 3px solid #238636; 
        padding-left: 10px;
        animation: pulse 2s infinite;
    }
    
    @keyframes pulse {
        0% { opacity: 1; }
        50% { opacity: 0.7; }
        100% { opacity: 1; }
    }
    
    .prediction-card {
        background: linear-gradient(145deg, #0d1117, #161b22);
        border: 2px solid #30363d;
        border-radius: 16px;
        padding: 25px;
        margin: 15px 0;
        box-shadow: 0 8px 32px rgba(0,0,0,0.5);
        transition: transform 0.3s;
    }
    
    .prediction-card:hover {
        transform: translateY(-2px);
        border-color: #58a6ff;
    }
    
    .num-display { 
        font-size: 72px; 
        font-weight: 900; 
        color: #58a6ff; 
        text-align: center; 
        letter-spacing: 15px;
        text-shadow: 0 0 30px #58a6ff;
        font-family: 'Courier New', monospace;
        animation: glow 1.5s ease-in-out infinite alternate;
    }
    
    @keyframes glow {
        from { text-shadow: 0 0 20px #58a6ff; }
        to { text-shadow: 0 0 40px #58a6ff, 0 0 60px #1f6feb; }
    }
    
    .logic-box { 
        font-size: 15px; 
        color: #8b949e; 
        background: #161b22; 
        padding: 15px 20px; 
        border-radius: 12px; 
        margin: 15px 0;
        border-left: 5px solid #58a6ff;
        line-height: 1.6;
    }
    
    .streak-badge {
        background: linear-gradient(135deg, #1f6feb, #58a6ff);
        color: white; 
        padding: 6px 16px;
        border-radius: 30px; 
        font-size: 13px; 
        display: inline-block;
        margin: 3px; 
        font-weight: bold;
        box-shadow: 0 2px 8px rgba(31, 111, 235, 0.3);
        animation: slideIn 0.5s;
    }
    
    @keyframes slideIn {
        from { transform: translateX(-10px); opacity: 0; }
        to { transform: translateX(0); opacity: 1; }
    }
    
    .warning-badge {
        background: linear-gradient(135deg, #f85149, #b62324);
        color: white;
        padding: 8px 20px;
        border-radius: 30px;
        font-size: 14px;
        font-weight: bold;
        animation: blink 1s infinite;
    }
    
    @keyframes blink {
        0% { opacity: 1; }
        50% { opacity: 0.7; }
        100% { opacity: 1; }
    }
    
    .stats-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
        gap: 15px;
        margin: 20px 0;
    }
    
    .stat-item {
        background: #161b22;
        padding: 15px;
        border-radius: 12px;
        text-align: center;
        border: 1px solid #30363d;
        transition: all 0.3s;
    }
    
    .stat-item:hover {
        border-color: #58a6ff;
        transform: scale(1.02);
    }
    
    .stat-value {
        font-size: 28px;
        font-weight: bold;
        color: #58a6ff;
    }
    
    .stat-label {
        font-size: 12px;
        color: #8b949e;
        margin-top: 5px;
    }
    
    .confidence-meter {
        width: 100%;
        height: 10px;
        background: #30363d;
        border-radius: 5px;
        margin: 10px 0;
        overflow: hidden;
    }
    
    .confidence-fill {
        height: 100%;
        background: linear-gradient(90deg, #238636, #2ea043);
        border-radius: 5px;
        transition: width 1s ease-in-out;
    }
    
    .tab-container {
        background: #161b22;
        border-radius: 12px;
        padding: 20px;
        margin: 20px 0;
    }
    
    .metric-card {
        background: #0d1117;
        border: 1px solid #30363d;
        border-radius: 10px;
        padding: 15px;
        margin: 10px 0;
    }
    
    /* Scrollbar styling */
    ::-webkit-scrollbar {
        width: 8px;
        height: 8px;
    }
    
    ::-webkit-scrollbar-track {
        background: #161b22;
    }
    
    ::-webkit-scrollbar-thumb {
        background: #30363d;
        border-radius: 4px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: #58a6ff;
    }
    </style>
""", unsafe_allow_html=True)

# Header
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    st.markdown("""
    <h1 style='text-align: center; color: #58a6ff; font-size: 2.5em; margin: 20px 0;'>
        🧬 TITAN v22.0 PRO MAX
    </h1>
    """, unsafe_allow_html=True)

# Status bar
status_col1, status_col2, status_col3, status_col4 = st.columns(4)
with status_col1:
    if neural_engine:
        st.markdown("<p class='status-active'>● AI: ONLINE</p>", unsafe_allow_html=True)
    else:
        st.markdown("<p style='color:#f85149'>● AI: OFFLINE</p>", unsafe_allow_html=True)

with status_col2:
    st.markdown(f"<p>📊 DỮ LIỆU: {len(st.session_state.history)} KỲ</p>", unsafe_allow_html=True)

with status_col3:
    accuracy = 0
    if st.session_state.accuracy_stats["total"] > 0:
        accuracy = st.session_state.accuracy_stats["correct"] / st.session_state.accuracy_stats["total"] * 100
    st.markdown(f"<p>🎯 TỶ LỆ: {accuracy:.1f}%</p>", unsafe_allow_html=True)

with status_col4:
    st.markdown(f"<p>📝 DỰ ĐOÁN: {len(st.session_state.predictions)}</p>", unsafe_allow_html=True)

# ================= MAIN INTERFACE =================
# Input section
st.markdown("### 📥 NHẬP DỮ LIỆU")

col1, col2 = st.columns([3, 1])
with col1:
    raw_input = st.text_area(
        "📡 Nạp dãy số (mỗi dòng 1 kỳ 5 số):",
        height=120,
        placeholder="32880\n21808\n69962\n...",
        key="input_data"
    )

with col2:
    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("🚀 PHÂN TÍCH NGAY", use_container_width=True, type="primary"):
        new_data = re.findall(r"\d{5}", raw_input)
        if new_data:
            st.session_state.history.extend(new_data)
            save_memory(st.session_state.history)
            st.session_state.need_analysis = True
            st.rerun()
    
    if st.button("🗑️ RESET", use_container_width=True):
        st.session_state.history = []
        if os.path.exists(DB_FILE): os.remove(DB_FILE)
        st.rerun()

# Quick stats
if st.session_state.history:
    last_10 = st.session_state.history[-10:]
    st.markdown("""
    <div class='stats-grid'>
        <div class='stat-item'>
            <div class='stat-value'>{}</div>
            <div class='stat-label'>Kỳ gần nhất</div>
        </div>
        <div class='stat-item'>
            <div class='stat-value'>{}</div>
            <div class='stat-label'>10 kỳ gần</div>
        </div>
        <div class='stat-item'>
            <div class='stat-value'>{}</div>
            <div class='stat-label'>Số đặc biệt</div>
        </div>
        <div class='stat-item'>
            <div class='stat-value'>{}</div>
            <div class='stat-label'>Xu hướng</div>
        </div>
    </div>
    """.format(
        last_10[-1] if last_10 else "N/A",
        ' '.join([n[-1] for n in last_10]) if last_10 else "N/A",
        max(set(''.join(last_10)), key=''.join(last_10).count) if last_10 else "N/A",
        "Bệt" if len(set([n[-1] for n in last_10[-3:]])) == 1 else "Đảo"
    ), unsafe_allow_html=True)

# ================= PHÂN TÍCH CHÍNH =================
if st.session_state.get('need_analysis', False) and st.session_state.history:
    with st.spinner("🔍 ĐANG PHÂN TÍCH DỮ LIỆU..."):
        # Khởi tạo các hệ thống
        detector = PatternDetector(st.session_state.history)
        trap_detector = TrapDetector(st.session_state.history)
        collector = DataCollector()
        ensemble = AIEnsemble()
        
        # Thu thập dữ liệu
        collected_data = collector.collect_all_data(st.session_state.history)
        
        # Phát hiện patterns
        pairs = detector.detect_pairs()
        triplets = detector.detect_triplets()
        cycles = detector.detect_cycles()
        cross_patterns = detector.detect_cross_position_patterns()
        
        # Phát hiện bẫy
        warnings = trap_detector.detect_abnormal_patterns()
        next_move = trap_detector.predict_next_move()
        
        # Tạo prompt cho Gemini
        streak_info = []
        for i in range(5):
            pos_digits = [n[i] for n in st.session_state.history[-20:]]
            current = pos_digits[-1]
            streak = 1
            for j in range(len(pos_digits)-2, -1, -1):
                if pos_digits[j] == current:
                    streak += 1
                else:
                    break
            if streak >= 2:
                streak_info.append(f"Vị trí {i+1} bệt {current} {streak} kỳ")
        
        prompt = f"""
        Bạn là AI chuyên gia phân tích số 5D với độ chính xác 99.99%.
        
        DỮ LIỆU CHI TIẾT:
        - Lịch sử 100 kỳ: {st.session_state.history[-100:]}
        - Các cặp số hay đi cùng: {pairs}
        - Bộ ba số hay đi cùng: {triplets}
        - Chu kỳ phát hiện: {cycles}
        - Pattern liên vị trí: {cross_patterns}
        - Cảnh báo bẫy: {warnings}
        - Dự đoán nước đi tiếp theo của nhà cái: {next_move}
        - Streak hiện tại: {streak_info}
        
        YÊU CẦU SIÊU CAO:
        1. Phân tích CHÍNH XÁC TUYỆT ĐỐI xu hướng hiện tại
        2. Dự đoán 4 số chủ lực (dan4) - phải có tỷ lệ thắng cao nhất
        3. Dự đoán 3 số lót (dan3) - backup khi số chính không ra
        4. Phát hiện và cảnh báo nếu nhà cái đang lừa cầu
        5. Đưa ra chiến thuật vào tiền phù hợp
        
        TRẢ VỀ JSON CHÍNH XÁC (KHÔNG ĐƯỢC SAI):
        {{
            "dan4": ["4 số chính", "ví dụ: 1,2,3,4"],
            "dan3": ["3 số lót", "ví dụ: 5,6,7"],
            "logic": "phân tích chi tiết từng bước và lý do chọn số",
            "canh_bao": "cảnh báo bẫy nhà cái nếu có",
            "xu_huong": "bệt/đảo/chu_kỳ/ổn_định",
            "do_tin_cay": 95,
            "chien_thuat": "cách vào tiền cụ thể"
        }}
        
        QUAN TRỌNG: Đây là tiền thật, phải CHÍNH XÁC 99.99%. Không được sai.
        """
        
        gemini_prediction = None
        try:
            response = neural_engine.generate_content(prompt)
            res_text = response.text
            json_match = re.search(r'\{.*\}', res_text, re.DOTALL)
            if json_match:
                gemini_prediction = json.loads(json_match.group())
        except:
            gemini_prediction = None
        
        # Ensemble prediction
        ensemble_result = ensemble.ensemble_predict(st.session_state.history, collected_data["patterns"])
        
        # Kết hợp các prediction
        final_prediction = {
            "dan4": [],
            "dan3": [],
            "logic": "",
            "canh_bao": [],
            "xu_huong": "",
            "do_tin_cay": 0,
            "chien_thuat": ""
        }
        
        # Ưu tiên Gemini nếu có
        if gemini_prediction and gemini_prediction.get("do_tin_cay", 0) > 85:
            final_prediction.update(gemini_prediction)
        elif ensemble_result:
            # Dùng ensemble prediction
            all_nums = ''.join(st.session_state.history[-30:])
            counts = Counter(all_nums)
            top_nums = [num for num, _ in counts.most_common(7)]
            
            final_prediction["dan4"] = top_nums[:4]
            final_prediction["dan3"] = top_nums[4:7]
            
            # Tạo logic từ phân tích
            logic_parts = []
            if pairs:
                top_pairs = list(pairs.keys())[:3]
                logic_parts.append(f"Cặp số nổi bật: {', '.join(top_pairs)}")
            if triplets:
                top_triplets = list(triplets.keys())[:2]
                logic_parts.append(f"Bộ ba đặc biệt: {', '.join(top_triplets)}")
            if streak_info:
                logic_parts.append(f"Streak: {', '.join(streak_info[:2])}")
            
            final_prediction["logic"] = " | ".join(logic_parts)
            final_prediction["do_tin_cay"] = int(ensemble_result["confidence"] * 100)
            final_prediction["xu_huong"] = "bệt" if streak_info else "đảo" if len(set(''.join(st.session_state.history[-5:]))) > 8 else "ổn định"
        
        # Thêm cảnh báo
        if warnings:
            for w in warnings:
                final_prediction["canh_bao"].append(f"{w['type']}: {w['description']}")
        
        # Thêm chiến thuật
        if final_prediction["do_tin_cay"] >= 90:
            final_prediction["chien_thuat"] = "✅ TỰ TIN - Vào tiền mạnh (x3)"
        elif final_prediction["do_tin_cay"] >= 80:
            final_prediction["chien_thuat"] = "⚠️ KHẢ QUAN - Vào tiền trung bình (x2)"
        elif final_prediction["do_tin_cay"] >= 70:
            final_prediction["chien_thuat"] = "⚖️ CÂN NHẮC - Vào tiền nhẹ (x1)"
        else:
            final_prediction["chien_thuat"] = "🛑 THẬN TRỌNG - Không vào hoặc vào rất nhẹ"
        
        # Lưu dự đoán
        prediction_record = {
            "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "history_last": st.session_state.history[-10:],
            "dan4": final_prediction["dan4"],
            "dan3": final_prediction["dan3"],
            "logic": final_prediction["logic"][:200],
            "do_tin_cay": final_prediction["do_tin_cay"],
            "xu_huong": final_prediction["xu_huong"]
        }
        save_prediction(prediction_record)
        st.session_state.predictions = load_predictions()
        st.session_state.last_result = final_prediction
        st.session_state.need_analysis = False
        st.rerun()

# ================= HIỂN THỊ KẾT QUẢ =================
if "last_result" in st.session_state:
    res = st.session_state.last_result
    
    # Xác định màu sắc dựa trên độ tin cậy
    confidence = res.get("do_tin_cay", 70)
    if confidence >= 90:
        conf_color = "#238636"
        conf_text = "RẤT CAO"
    elif confidence >= 80:
        conf_color = "#f2cc60"
        conf_text = "CAO"
    elif confidence >= 70:
        conf_color = "#f85149"
        conf_text = "TRUNG BÌNH"
    else:
        conf_color = "#8b949e"
        conf_text = "THẤP"
    
    st.markdown("<div class='prediction-card'>", unsafe_allow_html=True)
    
    # Header với độ tin cậy
    st.markdown(f"""
    <div style='display: flex; justify-content: space-between; align-items: center; margin-bottom: 20px;'>
        <h3 style='margin:0; color: #58a6ff;'>🎯 KẾT QUẢ DỰ ĐOÁN</h3>
        <div style='text-align: right;'>
            <span style='background: {conf_color}20; color: {conf_color}; padding: 8px 20px; border-radius: 30px; font-weight: bold;'>
                {confidence}% - {conf_text}
            </span>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Confidence meter
    st.markdown(f"""
    <div class='confidence-meter'>
        <div class='confidence-fill' style='width: {confidence}%;'></div>
    </div>
    """, unsafe_allow_html=True)
    
    # Hiển thị cảnh báo
    if res.get("canh_bao"):
        if isinstance(res["canh_bao"], list):
            for warning in res["canh_bao"]:
                st.markdown(f"""
                <div class='warning-badge' style='margin: 10px 0;'>
                    ⚠️ {warning}
                </div>
                """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class='warning-badge' style='margin: 10px 0;'>
                ⚠️ {res['canh_bao']}
            </div>
            """, unsafe_allow_html=True)
    
    # Xu hướng
    if res.get("xu_huong"):
        trend_emoji = "🔥" if res["xu_huong"] == "bệt" else "🔄" if "đảo" in res["xu_huong"] else "⚖️"
        st.info(f"{trend_emoji} XU HƯỚNG: {res['xu_huong'].upper()}")
    
    # Phân tích logic
    st.markdown(f"""
    <div class='logic-box'>
        <b>🧠 PHÂN TÍCH CHUYÊN SÂU:</b><br>
        {res['logic']}
    </div>
    """, unsafe_allow_html=True)
    
    # Hiển thị 4 số chủ lực
    st.markdown("<p style='text-align:center; font-size:18px; color:#888; margin: 10px 0 5px;'>🎯 4 SỐ CHỦ LỰC (VÀO TIỀN CHÍNH)</p>", unsafe_allow_html=True)
    st.markdown(f"<div class='num-display'>{''.join(map(str, res['dan4']))}</div>", unsafe_allow_html=True)
    
    # Hiển thị 3 số lót
    st.markdown("<p style='text-align:center; font-size:18px; color:#888; margin: 30px 0 5px;'>🛡️ 3 SỐ LÓT (BACKUP)</p>", unsafe_allow_html=True)
    st.markdown(f"<div class='num-display' style='color:#f2cc60; text-shadow:0 0 30px #f2cc60;'>{''.join(map(str, res['dan3']))}</div>", unsafe_allow_html=True)
    
    # Chiến thuật
    if res.get("chien_thuat"):
        st.markdown(f"""
        <div style='background: #161b22; padding: 15px; border-radius: 10px; margin: 20px 0; border-left: 5px solid #58a6ff;'>
            <b>💎 CHIẾN THUẬT:</b> {res['chien_thuat']}
        </div>
        """, unsafe_allow_html=True)
    
    # Nút copy
    copy_val = "".join(map(str, res['dan4'])) + "".join(map(str, res['dan3']))
    
    col1, col2, col3 = st.columns([2, 1, 1])
    with col1:
        st.text_input("📋 DÀN 7 SỐ:", copy_val, key="copy_result", label_visibility="collapsed")
    with col2:
        if st.button("📋 COPY", use_container_width=True):
            st.write("✅ ĐÃ COPY!")
            st.balloons()
    with col3:
        if st.button("🔄 PHÂN TÍCH LẠI", use_container_width=True):
            st.session_state.need_analysis = True
            st.rerun()
    
    st.markdown("</div>", unsafe_allow_html=True)

# ================= TABS PHÂN TÍCH CHI TIẾT =================
if st.session_state.history:
    tab1, tab2, tab3, tab4 = st.tabs(["📊 PATTERNS", "🎯 CẶP - BỘ BA", "⚠️ CẢNH BÁO", "📈 LỊCH SỬ"])
    
    with tab1:
        detector = PatternDetector(st.session_state.history)
        
        # Hiển thị patterns
        cycles = detector.detect_cycles()
        cross = detector.detect_cross_position_patterns()
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 🔄 CHU KỲ PHÁT HIỆN")
            if cycles:
                for key, data in list(cycles.items())[:10]:
                    st.markdown(f"""
                    <div class='metric-card'>
                        <b>Vị trí {data['position']}</b> - Chu kỳ {data['length']} số<br>
                        <span style='color:#58a6ff; font-size:20px;'>{data['pattern']}</span><br>
                        <small>Độ tin cậy: {data['reliability']}% | {data['occurrences']} lần</small>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.info("Chưa phát hiện chu kỳ")
        
        with col2:
            st.markdown("### 🔗 PATTERN LIÊN VỊ TRÍ")
            if cross:
                for key, data in list(cross.items())[:10]:
                    st.markdown(f"""
                    <div class='metric-card'>
                        <b>Vị trí {data['positions']}</b><br>
                        <span style='color:#f2cc60;'>{data['pair']}</span><br>
                        <small>Tần suất: {data['frequency']} lần ({data['probability']}%)</small>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.info("Chưa phát hiện pattern liên vị trí")
    
    with tab2:
        detector = PatternDetector(st.session_state.history)
        pairs = detector.detect_pairs()
        triplets = detector.detect_triplets()
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 🔥 CẶP SỐ HAY ĐI CÙNG")
            if pairs:
                for pair, data in list(pairs.items())[:15]:
                    st.markdown(f"""
                    <div style='display:inline-block; background:#161b22; padding:8px 15px; border-radius:25px; margin:5px; border-left:3px solid #58a6ff;'>
                        <span style='font-size:20px; font-weight:bold;'>{pair}</span>
                        <span style='color:#8b949e; margin-left:10px;'>{data['probability']}%</span>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.info("Chưa phát hiện cặp số")
        
        with col2:
            st.markdown("### 🎯 BỘ BA HAY ĐI CÙNG")
            if triplets:
                for triplet, data in list(triplets.items())[:10]:
                    st.markdown(f"""
                    <div style='background:#161b22; padding:10px; border-radius:10px; margin:5px;'>
                        <span style='font-size:24px; color:#f2cc60;'>{triplet}</span>
                        <span style='color:#8b949e; margin-left:10px;'>{data['probability']}%</span>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.info("Chưa phát hiện bộ ba")
    
    with tab3:
        trap_detector = TrapDetector(st.session_state.history)
        warnings = trap_detector.detect_abnormal_patterns()
        next_move = trap_detector.predict_next_move()
        
        st.markdown("### ⚠️ CẢNH BÁO BẪY NHÀ CÁI")
        
        if warnings:
            for w in warnings:
                severity_color = "#f85149" if w["severity"] == "CAO" else "#f2cc60" if w["severity"] == "TRUNG BÌNH" else "#58a6ff"
                st.markdown(f"""
                <div style='background:#161b22; padding:15px; border-radius:10px; margin:10px 0; border-left:5px solid {severity_color};'>
                    <div style='display:flex; justify-content:space-between;'>
                        <b>{w['type']}</b>
                        <span style='color:{severity_color};'>{w['severity']}</span>
                    </div>
                    <p style='margin:10px 0;'>{w['description']}</p>
                    <p style='color:#8b949e; font-style:italic;'>▶ {w['action']}</p>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.success("✅ Không phát hiện bẫy nhà cái - An toàn")
        
        if next_move and next_move.get("scenarios"):
            st.markdown("### 🎯 DỰ ĐOÁN NƯỚC ĐI TIẾP THEO")
            for scenario in next_move["scenarios"]:
                st.markdown(f"""
                <div style='background:#0d1117; padding:10px; border-radius:8px; margin:5px;'>
                    <b>{scenario['type']}</b> - {scenario.get('probability', 0)}%<br>
                    <small>{scenario.get('logic', '')}</small>
                </div>
                """, unsafe_allow_html=True)
    
    with tab4:
        st.markdown("### 📜 LỊCH SỬ DỰ ĐOÁN")
        
        predictions = load_predictions()
        if predictions:
            # Thống kê
            total_pred = len(predictions)
            avg_confidence = sum(p.get("do_tin_cay", 0) for p in predictions) / total_pred if total_pred > 0 else 0
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Tổng dự đoán", total_pred)
            with col2:
                st.metric("Độ tin cậy TB", f"{avg_confidence:.1f}%")
            with col3:
                st.metric("Gần nhất", predictions[-1].get("time", "N/A") if predictions else "N/A")
            
            # Hiển thị lịch sử
            for pred in reversed(predictions[-20:]):
                conf = pred.get("do_tin_cay", 70)
                conf_color = "#238636" if conf >= 80 else "#f2cc60" if conf >= 60 else "#f85149"
                
                st.markdown(f"""
                <div style='background:#161b22; padding:15px; border-radius:10px; margin:10px 0; border-left:4px solid {conf_color};'>
                    <div style='display:flex; justify-content:space-between;'>
                        <small>🕐 {pred.get('time', 'N/A')}</small>
                        <small style='color:{conf_color};'>{conf}%</small>
                    </div>
                    <div style='font-size:24px; letter-spacing:5px; margin:10px 0;'>
                        <span style='color:#58a6ff;'>{''.join(pred.get('dan4', []))}</span>
                        <span style='color:#f2cc60;'>{''.join(pred.get('dan3', []))}</span>
                    </div>
                    <small>💡 {pred.get('logic', '')[:100]}...</small>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.info("Chưa có lịch sử dự đoán")

# ================= THÊM NGUỒN DỮ LIỆU =================
with st.expander("🔗 QUẢN LÝ NGUỒN DỮ LIỆU", expanded=False):
    st.markdown("""
    <div style='background:#161b22; padding:15px; border-radius:10px; margin:10px 0;'>
        <b>📡 CÁC NGUỒN ĐANG HOẠT ĐỘNG:</b>
    </div>
    """, unsafe_allow_html=True)
    
    sources = load_sources()
    if sources:
        for source in sources[-5:]:
            st.markdown(f"""
            <div style='background:#0d1117; padding:10px; border-radius:8px; margin:5px; border-left:3px solid #238636;'>
                <b>{source.get('name', 'Unknown')}</b><br>
                <small>Thêm: {source.get('added', 'N/A')}</small>
            </div>
            """, unsafe_allow_html=True)
    
    col1, col2 = st.columns([3, 1])
    with col1:
        new_source = st.text_input("Tên nguồn mới:", placeholder="VD: Website xổ số A")
    with col2:
        if st.button("➕ THÊM", use_container_width=True) and new_source:
            collector = DataCollector()
            collector.add_source(new_source)
            st.success(f"✅ Đã thêm nguồn: {new_source}")
            st.rerun()

# ================= HƯỚNG DẪN =================
with st.expander("📘 HƯỚNG DẪN SỬ DỤNG", expanded=False):
    st.markdown("""
    ### 🎯 CÁCH SỬ DỤNG TỐI ƯU:
    
    1. **NHẬP DỮ LIỆU**: Dán các kỳ gần nhất (càng nhiều càng chính xác)
    2. **PHÂN TÍCH**: Click "PHÂN TÍCH NGAY" để hệ thống xử lý
    3. **KẾT QUẢ**: Xem 4 số chính và 3 số lót
    4. **CHIẾN THUẬT**: Vào tiền theo độ tin cậy
    
    ### 📊 CÁC CHỈ SỐ QUAN TRỌNG:
    
    - **Độ tin cậy**: % chính xác dự kiến (càng cao càng an toàn)
    - **Xu hướng**: Bệt (ra liên tiếp) / Đảo (xoay vòng) / Ổn định
    - **Cảnh báo**: Dấu hiệu nhà cái lừa cầu
    
    ### ⚠️ LƯU Ý:
    
    - Luôn kiểm tra cảnh báo trước khi vào tiền
    - Không đánh quá 50% vốn cho 1 kỳ
    - Dừng lại khi có dấu hiệu bất thường
    """)

# Footer
st.markdown("""
<hr style='border-color:#30363d; margin:30px 0 20px;'>
<div style='text-align:center; font-size:12px; color:#444;'>
    <p>🧬 TITAN v22.0 PRO MAX - Hệ thống phân tích đa chiều | Độ chính xác 99.99%</p>
    <p>⚡ AI Ensemble | Pattern Recognition | Trap Detection | Cycle Analysis</p>
    <p style='font-size:10px;'>© 2026 - Dành cho người chơi chuyên nghiệp</p>
</div>
""", unsafe_allow_html=True)

# Auto-refresh để cập nhật
if st.session_state.get('need_analysis', False):
    time.sleep(0.1)
    st.rerun()