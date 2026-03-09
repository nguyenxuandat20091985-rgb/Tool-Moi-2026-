# ==============================================================================
# TITAN AI v5.0 - AI Algorithms & Pattern Detection
# Production-Ready with Full Error Handling
# ==============================================================================

import numpy as np
import pandas as pd
from collections import Counter, defaultdict
import random
import math
import re  # ← CRITICAL: Must have this import
from typing import Dict, List, Tuple, Optional
from config import Config

# Validate configuration on module load
Config.validate()


class HousePatternDetector:
    """Professional house manipulation pattern detector."""
    
    def __init__(self):
        """Initialize pattern detector."""
        self.detected_patterns: Dict = {}
        self.risk_level: int = 0
    
    def detect_all_patterns(self, data: List[str]) -> Dict:
        """Run all pattern detection algorithms."""
        try:
            patterns = {}
            patterns['bet_cau'] = self._detect_bet_cau(data)
            patterns['dao_cau'] = self._detect_dao_cau(data)
            patterns['xoay_cau'] = self._detect_xoay_cau(data)
            patterns['nhip_bay'] = self._detect_nhip_bay(data)
            patterns['tong_control'] = self._detect_sum_control(data)
            
            self.risk_level = self._calculate_house_control_risk(patterns)
            self.detected_patterns = patterns
            
            return patterns
            
        except Exception as e:
            # Return safe defaults on error
            return {
                'bet_cau': {'detected': False, 'patterns': [], 'risk': 0, 'max_streak': 0},
                'dao_cau': {'detected': False, 'patterns': [], 'risk': 0},
                'xoay_cau': {'detected': False, 'patterns': [], 'risk': 0},
                'nhip_bay': {'detected': False, 'patterns': [], 'risk': 0},
                'tong_control': {'detected': False, 'patterns': [], 'risk': 0}
            }
    
    def _detect_bet_cau(self, data: List[str]) -> Dict:
        """Detect streaks (bệt cầu)."""
        if len(data) < 20:
            return {'detected': False, 'patterns': [], 'risk': 0, 'max_streak': 0}
        
        recent = data[:50] if len(data) >= 50 else data
        patterns = []
        risk = 0
        max_streak = 0
        
        for pos in range(5):
            seq = [n[pos] if len(n) > pos else '0' for n in recent]
            i = 0
            while i < len(seq) - 2:
                if seq[i] == seq[i+1] == seq[i+2]:
                    d = seq[i]
                    streak = 3
                    j = i + 3
                    while j < len(seq) and seq[j] == d:
                        streak += 1
                        j += 1
                    
                    if streak >= 3:
                        patterns.append({
                            'type': 'Bệt cầu',
                            'position': pos,
                            'digit': d,
                            'streak': streak,
                            'description': f'Vị {pos}: Số {d} bệt {streak} kỳ'
                        })
                        max_streak = max(max_streak, streak)
                        
                        if streak >= 5:
                            risk += 40
                        elif streak >= 4:
                            risk += 25
                        else:
                            risk += 15
                    i = j
                else:
                    i += 1
        
        return {
            'detected': len(patterns) > 0,
            'patterns': patterns,
            'risk': min(100, risk),
            'max_streak': max_streak
        }
    
    def _detect_dao_cau(self, data: List[str]) -> Dict:
        """Detect reversals (đảo cầu)."""
        if len(data) < 20:
            return {'detected': False, 'patterns': [], 'risk': 0}
        
        recent = data[:50] if len(data) >= 50 else data
        patterns = []
        risk = 0
        
        for i in range(len(recent) - 3):
            a, b = recent[i], recent[i+1]
            
            if len(a) >= 2 and len(b) >= 2:
                if a[0:2] == b[1::-1]:
                    patterns.append({
                        'type': 'Đảo cầu 2 số',
                        'position': i,
                        'pattern': f'{a[0:2]} → {b[0:2]}',
                        'description': f'Kỳ {i}: {a[0:2]} đảo thành {b[0:2]}'
                    })
                    risk += 10
        
        return {
            'detected': len(patterns) > 0,
            'patterns': patterns[:10],
            'risk': min(100, risk)
        }
    
    def _detect_xoay_cau(self, data: List[str]) -> Dict:
        """Detect rotations (xoay cầu)."""
        if len(data) < 30:
            return {'detected': False, 'patterns': [], 'risk': 0}
        
        recent = data[:60] if len(data) >= 60 else data
        patterns = []
        risk = 0
        
        for pos in range(5):
            seq = [n[pos] if len(n) > pos else '0' for n in recent]
            
            for cycle_len in [3, 4, 5]:
                if len(seq) >= cycle_len * 3:
                    cycle_matches = 0
                    for i in range(len(seq) - cycle_len * 3):
                        base = seq[i:i+cycle_len]
                        match = True
                        for j in range(1, 3):
                            if seq[i+j*cycle_len:i+(j+1)*cycle_len] != base:
                                match = False
                                break
                        if match:
                            cycle_matches += 1
                    
                    if cycle_matches >= 2:
                        patterns.append({
                            'type': 'Xoay cầu chu kỳ',
                            'position': pos,
                            'cycle_length': cycle_len,
                            'matches': cycle_matches,
                            'description': f'Vị {pos}: Chu kỳ {cycle_len} kỳ'
                        })
                        risk += 20
        
        return {
            'detected': len(patterns) > 0,
            'patterns': patterns[:10],
            'risk': min(100, risk)
        }
    
    def _detect_nhip_bay(self, data: List[str]) -> Dict:
        """Detect rhythm traps (bẫy nhịp)."""
        if len(data) < 20:
            return {'detected': False, 'patterns': [], 'risk': 0}
        
        recent = data[:50] if len(data) >= 50 else data
        patterns = []
        risk = 0
        
        for pos in range(5):
            seq = [n[pos] if len(n) > pos else '0' for n in recent]
            for i in range(len(seq) - 5):
                if seq[i] == seq[i+2] == seq[i+4] and seq[i] != seq[i+1]:
                    d = seq[i]
                    if i+5 < len(seq) and seq[i+5] != d:
                        patterns.append({
                            'type': 'Bẫy nhịp 2',
                            'position': pos,
                            'digit': d,
                            'description': f'Vị {pos}: Số {d} nhịp 2 bị gãy'
                        })
                        risk += 15
        
        return {
            'detected': len(patterns) > 0,
            'patterns': patterns[:10],
            'risk': min(100, risk)
        }
    
    def _detect_sum_control(self, data: List[str]) -> Dict:
        """Detect sum control."""
        if len(data) < 20:
            return {'detected': False, 'patterns': [], 'risk': 0}
        
        recent = data[:50] if len(data) >= 50 else data
        sums = []
        
        for n in recent:
            if len(n) == 5 and n.isdigit():
                s = sum(int(d) for d in n)
                sums.append(s)
        
        if len(sums) < 10:
            return {'detected': False, 'patterns': [], 'risk': 0}
        
        patterns = []
        risk = 0
        
        sum_std = float(np.std(sums))
        if sum_std < 2.5:
            patterns.append({
                'type': 'Kiểm soát tổng',
                'std_dev': round(sum_std, 2),
                'description': f'Độ lệch chuẩn: {sum_std:.2f}'
            })
            risk += 30
        
        return {
            'detected': len(patterns) > 0,
            'patterns': patterns,
            'risk': min(100, risk)
        }
    
    def _calculate_house_control_risk(self, patterns: Dict) -> int:
        """Calculate overall house control risk."""
        total_risk = 0.0
        
        if patterns.get('bet_cau', {}).get('detected'):
            total_risk += patterns['bet_cau']['risk'] * Config.PATTERN_CONFIG['bet_cau_weight']
        if patterns.get('dao_cau', {}).get('detected'):
            total_risk += patterns['dao_cau']['risk'] * Config.PATTERN_CONFIG['dao_cau_weight']
        if patterns.get('xoay_cau', {}).get('detected'):
            total_risk += patterns['xoay_cau']['risk'] * Config.PATTERN_CONFIG['xoay_cau_weight']
        if patterns.get('nhip_bay', {}).get('detected'):
            total_risk += patterns['nhip_bay']['risk'] * Config.PATTERN_CONFIG['nhip_bay_weight']
        if patterns.get('tong_control', {}).get('detected'):
            total_risk += patterns['tong_control']['risk'] * Config.PATTERN_CONFIG['tong_control_weight']
        
        return min(100, int(total_risk))
    
    def get_house_control_level(self) -> Tuple[str, str]:
        """Get house control level description."""
        if self.risk_level >= Config.HOUSE_CONTROL['high']:
            return 'RẤT CAO', '🚫 Nhà cái điều khiển mạnh - NÊN DỪNG'
        elif self.risk_level >= Config.HOUSE_CONTROL['medium']:
            return 'CAO', '⚠️ Có dấu hiệu điều khiển - CẨN THẬN'
        elif self.risk_level >= Config.HOUSE_CONTROL['low']:
            return 'TRUNG BÌNH', '⚠️ Một số pattern bất thường'
        else:
            return 'THẤP', '✅ Nhịp số tự nhiên'


class TitanAI:
    """Main AI prediction engine - Production Ready."""
    
    def __init__(self):
        """Initialize AI engine."""
        self.weights = Config.ALGORITHM_WEIGHTS.copy()
        self.accuracy_history: List[Dict] = []
        self.pattern_detector = HousePatternDetector()
    
    def analyze(self, history: List[str], max_simulations: Optional[int] = None) -> Dict:
        """Main analysis method with full error handling."""
        try:
            if max_simulations is None:
                max_simulations = Config.DEFAULT_SIMULATIONS
            
            if not history or len(history) < Config.MIN_HISTORY_LENGTH:
                return self._fallback(f"Cần ít nhất {Config.MIN_HISTORY_LENGTH} kỳ")
            
            clean_data = self._clean_history(history)
            if len(clean_data) < Config.MIN_HISTORY_LENGTH:
                return self._fallback("Dữ liệu không hợp lệ")
            
            # Detect house patterns
            house_patterns = self.pattern_detector.detect_all_patterns(clean_data)
            house_control_level, house_warning = self.pattern_detector.get_house_control_level()
            
            # Run all analysis algorithms
            results = {}
            results['frequency'] = self._analyze_frequency(clean_data)
            results['gap'] = self._analyze_gap(clean_data)
            results['markov'] = self._analyze_markov(clean_data)
            results['monte_carlo'] = self._analyze_monte_carlo(clean_data, max_simulations)
            results['pattern'] = self._analyze_pattern_advanced(clean_data)
            results['hot_cold'] = self._analyze_hot_cold(clean_data)
            
            # Adjust weights based on house patterns
            if house_patterns['bet_cau']['detected'] and house_patterns['bet_cau']['risk'] >= 40:
                self.weights['frequency'] = 15
                self.weights['pattern'] = 25
            
            ensemble = self._ensemble_vote(results, house_patterns)
            stats_df = self._build_stats_df(clean_data, results)
            risk = self._calculate_risk(clean_data, house_patterns)
            
            return {
                'main_3': ensemble['main_3'],
                'support_4': ensemble['support_4'],
                'stats_df': stats_df,
                'risk': risk,
                'confidence': ensemble['confidence'],
                'logic': self._build_logic(results, ensemble, house_patterns),
                'house_patterns': house_patterns,
                'house_control_level': house_control_level,
                'house_warning': house_warning,
                'success': True
            }
            
        except Exception as e:
            return self._fallback(f"Lỗi phân tích: {str(e)[:50]}")
    
    def _clean_history(self, history: List[str]) -> List[str]:
        """Clean and validate history data."""
        cleaned = []
        for item in history:
            if not isinstance(item, str):
                item = str(item)
            s = item.strip()
            match = re.search(r'\d{5}', s)
            if match:
                cleaned.append(match.group())
        return cleaned
    
    def _analyze_frequency(self, data: List[str]) -> Dict:
        """Frequency analysis with recency weighting."""
        weighted = Counter()
        n = len(data)
        
        for idx, num in enumerate(data):
            weight = 3.0 - 2.0 * (idx / max(n, 1))
            for d in num:
                if d.isdigit():
                    weighted[d] += weight
        
        scores = {d: weighted.get(d, 0) for d in '0123456789'}
        top_3 = [d for d, _ in sorted(scores.items(), key=lambda x: -x[1])[:3]]
        
        while len(top_3) < 3:
            for d in '0123456789':
                if d not in top_3:
                    top_3.append(d)
                    break
        
        return {'scores': scores, 'top_3': top_3}
    
    def _analyze_gap(self, data: List[str]) -> Dict:
        """Gap analysis."""
        last_seen = {d: -1 for d in '0123456789'}
        
        for idx, num in enumerate(data):
            for d in num:
                if last_seen[d] == -1:
                    last_seen[d] = idx
        
        scores = {}
        for d in '0123456789':
            gap = last_seen[d] if last_seen[d] >= 0 else len(data)
            scores[d] = gap * 2.5 if gap <= 15 else max(0, 37.5 - (gap - 15) * 0.5)
        
        top_3 = [d for d, _ in sorted(scores.items(), key=lambda x: -x[1])[:3]]
        return {'scores': scores, 'top_3': top_3}
    
    def _analyze_markov(self, data: List[str]) -> Dict:
        """Markov chain analysis."""
        if len(data) < 20:
            return {'scores': {d: 10.0 for d in '0123456789'}, 'top_3': ['1', '5', '9']}
        
        transitions = defaultdict(Counter)
        
        for i in range(len(data) - 1):
            curr = data[i]
            next_num = data[i + 1]
            for pos in range(5):
                if pos < len(curr) and pos < len(next_num):
                    transitions[curr[pos]][next_num[pos]] += 1
        
        last_num = data[0] if data else '00000'
        next_prob = Counter()
        
        for pos, last_d in enumerate(last_num[:5]):
            if last_d in transitions and transitions[last_d]:
                total = sum(transitions[last_d].values())
                if total > 0:
                    for next_d, count in transitions[last_d].items():
                        pos_weight = 1.0 + 0.25 * (2 - abs(pos - 2))
                        next_prob[next_d] += (count / total) * pos_weight
        
        scores = {d: next_prob.get(d, 0) * 10 for d in '0123456789'}
        top_3 = [d for d, _ in sorted(scores.items(), key=lambda x: -x[1])[:3]]
        
        while len(top_3) < 3:
            for d in '0123456789':
                if d not in top_3:
                    top_3.append(d)
                    break
        
        return {'scores': scores, 'top_3': top_3}
    
    def _analyze_monte_carlo(self, data: List[str], n_simulations: int) -> Dict:
        """Monte Carlo simulation."""
        if len(data) < 20:
            return {'scores': {d: 10.0 for d in '0123456789'}, 'top_3': ['2', '4', '6']}
        
        recent = data[:80] if len(data) >= 80 else data
        pool = []
        
        for idx, num in enumerate(recent):
            weight = max(1, 4 - idx // 20)
            for d in num:
                if d.isdigit():
                    pool.extend([d] * weight)
        
        if not pool:
            return {'scores': {d: 10.0 for d in '0123456789'}, 'top_3': ['0', '1', '2']}
        
        sim_count = Counter()
        for _ in range(min(n_simulations, 5000)):
            sample = random.choices(pool, k=3)
            for d in sample:
                sim_count[d] += 1
        
        total = sum(sim_count.values()) or 1
        scores = {d: sim_count.get(d, 0) / total * 100 for d in '0123456789'}
        top_3 = [d for d, _ in sorted(scores.items(), key=lambda x: -x[1])[:3]]
        
        return {'scores': scores, 'top_3': top_3}
    
    def _analyze_pattern_advanced(self, data: List[str]) -> Dict:
        """Advanced pattern detection."""
        if len(data) < 25:
            return {'scores': {d: 10.0 for d in '0123456789'}, 'top_3': ['3', '5', '7'], 'patterns': [], 'avoid': []}
        
        recent = data[:50] if len(data) >= 50 else data
        candidates = Counter()
        patterns_found = []
        avoid = []
        
        # Bệt cầu detection
        for pos in range(5):
            seq = [n[pos] if len(n) > pos else '0' for n in recent]
            i = 0
            while i < len(seq) - 2:
                if seq[i] == seq[i+1] == seq[i+2]:
                    d = seq[i]
                    streak_len = 3
                    j = i + 3
                    while j < len(seq) and seq[j] == d:
                        streak_len += 1
                        j += 1
                    
                    if streak_len >= 3:
                        patterns_found.append(f'Bệt vị {pos}: {d} ({streak_len} kỳ)')
                        if streak_len >= 5:
                            avoid.append(d)
                        elif streak_len >= 3:
                            candidates[d] += 5
                    i = j
                else:
                    i += 1
        
        # Nhịp 2 detection
        for pos in range(5):
            seq = [n[pos] if len(n) > pos else '0' for n in recent]
            for i in range(len(seq) - 4):
                if seq[i] == seq[i+2] == seq[i+4] and seq[i] != seq[i+1]:
                    d = seq[i]
                    patterns_found.append(f'Nhịp-2 vị {pos}: {d}')
                    candidates[d] += 4
        
        if not candidates:
            all_digits = ''.join(recent)
            freq = Counter(all_digits)
            for d, c in freq.most_common(3):
                candidates[d] += 3
        
        scores = {d: candidates.get(d, 0) * 2 for d in '0123456789'}
        top_3 = [d for d, _ in sorted(scores.items(), key=lambda x: -x[1])[:3]]
        
        while len(top_3) < 3:
            for d in '0123456789':
                if d not in top_3:
                    top_3.append(d)
                    break
        
        return {'scores': scores, 'top_3': top_3, 'patterns': patterns_found[:10], 'avoid': list(set(avoid))}
    
    def _analyze_hot_cold(self, data: List[str]) -> Dict:
        """Hot/cold number analysis."""
        recent = data[:15] if len(data) >= 15 else data
        older = data[15:45] if len(data) >= 45 else []
        
        recent_count = Counter(''.join(recent))
        older_count = Counter(''.join(older)) if older else Counter()
        
        scores = {}
        for d in '0123456789':
            r = recent_count.get(d, 0)
            o = older_count.get(d, 0)
            
            if r >= 4:
                scores[d] = 25 + r * 3
            elif r == 0 and o >= 3:
                scores[d] = 20 + o * 2
            elif r >= 2:
                scores[d] = 15 + r * 2
            else:
                scores[d] = 10 + r
        
        top_3 = [d for d, _ in sorted(scores.items(), key=lambda x: -x[1])[:3]]
        return {'scores': scores, 'top_3': top_3}
    
    def _ensemble_vote(self, results: Dict, house_patterns: Dict) -> Dict:
        """Ensemble voting with house pattern awareness."""
        votes = Counter()
        avoid_votes = []
        
        # Avoid streaking numbers during high house control
        if house_patterns['bet_cau']['detected'] and house_patterns['bet_cau']['risk'] >= 40:
            for p in house_patterns['bet_cau']['patterns']:
                if p.get('streak', 0) >= 4:
                    avoid_votes.append(p['digit'])
        
        for algo_name, result in results.items():
            weight = self.weights.get(algo_name, 10)
            for d in result.get('top_3', []):
                votes[d] += weight
            if result.get('avoid'):
                avoid_votes.extend(result['avoid'])
        
        avoid_set = set(avoid_votes)
        main_3 = [d for d, _ in votes.most_common(3) if d not in avoid_set]
        
        while len(main_3) < 3:
            for d in '0123456789':
                if d not in main_3 and d not in avoid_set:
                    main_3.append(d)
                    break
        
        remaining = [d for d, _ in votes.most_common(10) if d not in main_3 and d not in avoid_set]
        support_4 = remaining[:4]
        
        while len(support_4) < 4:
            for d in '0123456789':
                if d not in main_3 and d not in support_4 and d not in avoid_set:
                    support_4.append(d)
                    break
        
        if votes:
            top_votes = [c for _, c in votes.most_common(3)]
            confidence = min(95, 55 + sum(top_votes) / 3)
        else:
            confidence = 50
        
        return {
            'main_3': main_3,
            'support_4': support_4,
            'confidence': int(confidence),
            'avoid': list(avoid_set)
        }
    
    def _build_stats_df(self, data: List[str], results: Dict) -> pd.DataFrame:
        """Build statistics DataFrame."""
        rows = []
        for d in '0123456789':
            row = {'Digit': d}
            row['Frequency'] = float(results['frequency']['scores'].get(d, 0))
            row['Gap'] = float(results['gap']['scores'].get(d, 0))
            row['Markov'] = float(results['markov']['scores'].get(d, 0))
            row['Monte_Carlo'] = float(results['monte_carlo']['scores'].get(d, 0))
            row['Pattern'] = float(results['pattern']['scores'].get(d, 0))
            row['Hot_Cold'] = float(results['hot_cold']['scores'].get(d, 0))
            
            ai_score = (row['Frequency'] * 0.25 + row['Gap'] * 0.20 + 
                       row['Markov'] * 0.20 + row['Monte_Carlo'] * 0.15 + 
                       row['Pattern'] * 0.12 + row['Hot_Cold'] * 0.08)
            row['AI_Score'] = round(ai_score, 1)
            rows.append(row)
        
        df = pd.DataFrame(rows)
        return df.sort_values('AI_Score', ascending=False).reset_index(drop=True)
    
    def _calculate_risk(self, data: List[str], house_patterns: Dict) -> Dict:
        """Calculate risk level."""
        base_risk = 0
        reasons = []
        
        # House control risk
        house_risk = self.pattern_detector.risk_level
        if house_risk >= 50:
            base_risk += house_risk * 0.5
            reasons.append(f'Nhà cái điều khiển: {house_risk}%')
        
        # Standard risk
        if len(data) < 20:
            return {'score': 30, 'level': 'MEDIUM', 'reason': 'Dữ liệu ít'}
        
        all_digits = ''.join(data[:50])
        counts = Counter(all_digits)
        total = len(all_digits)
        
        if total > 0:
            entropy = sum(- (c/total) * math.log2(c/total) for c in counts.values() if c > 0)
            if entropy < Config.RISK_THRESHOLDS['entropy_min']:
                base_risk += 25
                reasons.append('Kết quả quá đều')
            elif entropy > Config.RISK_THRESHOLDS['entropy_max']:
                base_risk += 15
                reasons.append('Biến động mạnh')
        
        base_risk = min(100, int(base_risk))
        level = 'HIGH' if base_risk >= 50 else 'MEDIUM' if base_risk >= 25 else 'OK'
        
        return {'score': base_risk, 'level': level, 'reason': '; '.join(reasons) if reasons else 'Ổn định'}
    
    def _build_logic(self, results: Dict, ensemble: Dict, house_patterns: Dict) -> str:
        """Build logic explanation."""
        parts = []
        
        freq_top = [d for d, _ in sorted(results['frequency']['scores'].items(), key=lambda x: -x[1])[:2]]
        if freq_top:
            parts.append(f"Tần suất: {','.join(freq_top)}")
        
        if house_patterns['bet_cau']['detected']:
            parts.append(f"Bệt: {house_patterns['bet_cau']['max_streak']} kỳ")
        
        if ensemble['confidence'] >= 75:
            parts.append('Đồng thuận cao')
        
        if ensemble.get('avoid'):
            parts.append(f"Tránh: {','.join(ensemble['avoid'][:2])}")
        
        return ' | '.join(parts) if parts else 'Phân tích AI'
    
    def _fallback(self, msg: str = "Chưa đủ dữ liệu") -> Dict:
        """Return fallback prediction."""
        return {
            'main_3': ['?', '?', '?'],
            'support_4': ['0', '0', '0', '0'],
            'stats_df': pd.DataFrame({'Digit': list('0123456789'), 'AI_Score': [0.0]*10}),
            'risk': {'score': 0, 'level': 'LOW', 'reason': msg},
            'confidence': 0,
            'logic': msg,
            'house_patterns': {},
            'house_control_level': 'N/A',
            'house_warning': '',
            'success': False
        }