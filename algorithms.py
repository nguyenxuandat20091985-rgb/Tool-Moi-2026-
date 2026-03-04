# ==============================================================================
# TITAN v36.0 - MULTI-LAYER AI PREDICTION ENGINE
# Self-Learning Architecture with Adaptive Weights
# ==============================================================================

import numpy as np
from collections import Counter, defaultdict
from datetime import datetime
import math

class PredictionEngine:
    """
    Multi-Layer AI Engine for 5D Bet Prediction
    Layers: Frequency | Pattern | Markov | Neural-Lite | Ensemble
    """
    
    def __init__(self):
        # Algorithm weights (auto-adjusted by self-learning)
        self.weights = {
            'frequency': 35,    # Statistical frequency analysis
            'pattern': 25,       # Pattern recognition
            'markov': 20,        # Markov chain prediction
            'neural': 15,        # Lightweight neural scoring
            'ensemble': 5        # Consensus boosting
        }
        
        # Learning parameters
        self.learning_rate = 0.1
        self.win_history = []  # Track last 20 predictions
        self.max_history = 20
        
        # Pattern memory for self-learning
        self.pattern_memory = defaultdict(list)
        
        # Risk thresholds
        self.risk_thresholds = {
            'streak': 5,
            'entropy_min': 2.8,
            'sum_std_max': 2.5
        }
    
    def predict(self, history):
        """
        Main prediction: Run all AI layers and ensemble results.
        Returns dict with main_3, support_4, confidence, logic.
        """
        if len(history) < 20:
            return self._fallback_prediction("Cần ít nhất 20 kỳ dữ liệu")
        
        # Run each AI layer
        results = {}
        results['frequency'] = self._layer_frequency(history)
        results['pattern'] = self._layer_pattern(history)
        results['markov'] = self._layer_markov(history)
        results['neural'] = self._layer_neural(history)
        
        # Ensemble voting with adaptive weights
        ensemble_result = self._ensemble_vote(results)
        
        # Calculate confidence based on layer agreement
        confidence = self._calculate_confidence(results, ensemble_result)
        
        # Build explanation logic
        logic = self._build_logic(results, ensemble_result)
        
        return {
            'main_3': ensemble_result['main_3'],
            'support_4': ensemble_result['support_4'],
            'confidence': confidence,
            'logic': logic,
            'avoid': ensemble_result.get('avoid', []),
            'layer_scores': {k: v.get('score', 0) for k, v in results.items()},
            'algorithm': 'Multi-Layer AI (v36.0)'
        }
    
    def calculate_risk(self, history):
        """
        Multi-factor risk assessment.
        Returns: (risk_score: 0-100, risk_level: LOW/MEDIUM/HIGH, reasons: list)
        """
        if len(history) < 20:
            return 0, "LOW", []
        
        recent = history[-50:] if len(history) >= 50 else history
        all_digits = ''.join(recent)
        freq = Counter(all_digits)
        reasons = []
        risk = 0
        
        # Factor 1: Over-represented numbers (nhà cái điều khiển)
        total_slots = len(all_digits)
        if total_slots > 0:
            for num, count in freq.most_common(3):
                rate = count / total_slots
                if rate > 0.25:
                    risk += 20
                    reasons.append(f"⚠️ Số '{num}' xuất hiện {rate*100:.0f}% (bất thường)")
        
        # Factor 2: Abnormal streaks (cầu bệt giả)
        for pos in range(5):
            seq = [n[pos] if len(n) > pos else '0' for n in recent]
            max_streak = self._find_max_streak(seq)
            if max_streak >= self.risk_thresholds['streak']:
                risk += 30
                reasons.append(f"🚫 Cầu bệt {max_streak} kỳ vị trí {pos} (NHÀ CÁI BẪY)")
        
        # Factor 3: Low entropy (kết quả không ngẫu nhiên)
        entropy = self._calculate_entropy(freq, len(all_digits))
        if entropy < self.risk_thresholds['entropy_min']:
            risk += 25
            reasons.append(f"⚠️ Entropy thấp ({entropy:.2f}) - Kết quả giả")
        
        # Factor 4: Overly stable sums (kiểm soát tổng)
        totals = [sum(int(d) for d in n) for n in recent if len(n) == 5]
        if len(totals) > 10:
            std_dev = np.std(totals)
            if std_dev < self.risk_thresholds['sum_std_max']:
                risk += 15
                reasons.append(f"⚠️ Tổng quá ổn định (σ={std_dev:.2f})")
        
        # Factor 5: Recent prediction accuracy (self-learning feedback)
        if self.win_history:
            recent_win_rate = sum(self.win_history[-10:]) / min(len(self.win_history), 10)
            if recent_win_rate < 0.2:  # Less than 20% win rate recently
                risk += 10
                reasons.append(f"⚠️ AI đang học lại (win rate thấp: {recent_win_rate*100:.0f}%)")
        
        risk = min(100, risk)
        level = "HIGH" if risk >= 70 else "MEDIUM" if risk >= 40 else "LOW"
        
        return risk, level, reasons
    
    def update_weights(self, won: bool, winning_method: str = None):
        """
        SELF-LEARNING: Adjust algorithm weights based on prediction outcome.
        Reinforcement learning style update.
        """
        # Record outcome
        self.win_history.append(1 if won else 0)
        if len(self.win_history) > self.max_history:
            self.win_history.pop(0)
        
        # Calculate adjustment factor
        adjustment = self.learning_rate if won else -self.learning_rate * 0.5
        
        # Boost the method that contributed to win (or reduce if lost)
        if winning_method and winning_method in self.weights:
            self.weights[winning_method] = min(50, self.weights[winning_method] + adjustment * 10)
        
        # Normalize weights to sum to 100
        total = sum(self.weights.values())
        if total > 0:
            for key in self.weights:
                self.weights[key] = round(self.weights[key] * 100 / total)
        
        # Store pattern memory for future learning
        if winning_method:
            self.pattern_memory[winning_method].append({
                'won': won,
                'timestamp': datetime.now().isoformat()
            })
            # Keep memory manageable
            if len(self.pattern_memory[winning_method]) > 50:
                self.pattern_memory[winning_method].pop(0)
    
    def get_ai_status(self):
        """Return AI engine status for display."""
        recent_win_rate = sum(self.win_history) / len(self.win_history) * 100 if self.win_history else 0
        
        return {
            'weights': self.weights,
            'recent_win_rate': round(recent_win_rate, 1),
            'predictions_tracked': len(self.win_history),
            'pattern_memory_size': sum(len(v) for v in self.pattern_memory.values())
        }
    
    # ==========================================================================
    # AI LAYER 1: FREQUENCY ANALYSIS (Statistical Foundation)
    # ==========================================================================
    
    def _layer_frequency(self, history):
        """
        Layer 1: Exponential-weighted frequency analysis.
        Recent data weighted heavier than old data.
        """
        recent = history[-100:] if len(history) >= 100 else history
        
        weighted_freq = defaultdict(float)
        
        for idx, num in enumerate(recent):
            # Exponential decay: weight = 5.0 (most recent) → 1.0 (oldest)
            weight = 5.0 - 4.0 * (idx / max(len(recent), 1))
            for digit in num:
                if digit.isdigit():
                    weighted_freq[digit] += weight
        
        # Get top candidates
        sorted_items = sorted(weighted_freq.items(), key=lambda x: x[1], reverse=True)
        top_3 = [str(x[0]) for x in sorted_items[:3]]
        
        # Fill if needed
        while len(top_3) < 3:
            for i in range(10):
                if str(i) not in top_3:
                    top_3.append(str(i))
                    break
        
        return {
            'top_3': top_3,
            'scores': {k: round(v, 2) for k, v in sorted_items[:10]},
            'score': sum(weighted_freq.get(n, 0) for n in top_3) / 3,
            'method': 'Exponential Frequency'
        }
    
    # ==========================================================================
    # AI LAYER 2: PATTERN RECOGNITION (Rule-Based AI)
    # ==========================================================================
    
    def _layer_pattern(self, history):
        """
        Layer 2: Pattern detection with rule-based reasoning.
        Detects: streaks, rhythms, reversals, triangles.
        """
        recent = history[-30:] if len(history) >= 30 else history
        
        patterns = {
            'bet': [], 'nhip2': [], 'nhip3': [], 'dao': [],
            'detected': [], 'likely': [], 'avoid': []
        }
        
        # Detect streaks (cầu bệt)
        for pos in range(5):
            seq = [n[pos] if len(n) > pos else '0' for n in recent]
            i = 0
            while i < len(seq) - 2:
                if seq[i] == seq[i+1] == seq[i+2]:
                    digit = seq[i]
                    streak_len = 3
                    j = i + 3
                    while j < len(seq) and seq[j] == digit:
                        streak_len += 1
                        j += 1
                    
                    if digit not in patterns['bet']:
                        patterns['bet'].append(digit)
                        patterns['detected'].append(f'Bệt {streak_len} kỳ vị {pos}: {digit}')
                        
                        # If streak >= 4, likely to break → avoid
                        if streak_len >= 4:
                            if digit not in patterns['avoid']:
                                patterns['avoid'].append(digit)
                        else:
                            if digit not in patterns['likely']:
                                patterns['likely'].append(digit)
                    i = j
                else:
                    i += 1
        
        # Detect rhythm-2 (X _ X _ X)
        for pos in range(5):
            seq = [n[pos] if len(n) > pos else '0' for n in recent]
            for i in range(len(seq) - 4):
                if seq[i] == seq[i+2] == seq[i+4] and seq[i] != seq[i+1]:
                    digit = seq[i]
                    if digit not in patterns['nhip2']:
                        patterns['nhip2'].append(digit)
                        patterns['detected'].append(f'Nhịp-2 vị {pos}: {digit}')
                        if digit not in patterns['likely']:
                            patterns['likely'].append(digit)
        
        # Detect rhythm-3 (X _ _ X _ _ X)
        for pos in range(5):
            seq = [n[pos] if len(n) > pos else '0' for n in recent]
            for i in range(len(seq) - 6):
                if seq[i] == seq[i+3] == seq[i+6]:
                    digit = seq[i]
                    if digit not in patterns['nhip3']:
                        patterns['nhip3'].append(digit)
                        patterns['detected'].append(f'Nhịp-3 vị {pos}: {digit}')
                        if digit not in patterns['likely']:
                            patterns['likely'].append(digit)
        
        # Build candidates from patterns
        candidates = Counter()
        for digit in patterns['likely']:
            candidates[digit] += 3
        for digit in patterns['avoid']:
            candidates[digit] -= 2  # Penalize avoided numbers
        
        top_3 = [str(x[0]) for x in candidates.most_common(3)]
        while len(top_3) < 3:
            for i in range(10):
                if str(i) not in top_3:
                    top_3.append(str(i))
                    break
        
        return {
            'top_3': top_3,
            'patterns_found': patterns['detected'],
            'likely': patterns['likely'],
            'avoid': patterns['avoid'],
            'score': len(patterns['detected']) * 2,
            'method': 'Rule-Based Pattern AI'
        }
    
    # ==========================================================================
    # AI LAYER 3: MARKOV CHAIN (Probabilistic Modeling)
    # ==========================================================================
    
    def _layer_markov(self, history):
        """
        Layer 3: Markov Chain prediction.
        Models: last_digit → next_digits transition probabilities.
        """
        if len(history) < 30:
            return {'top_3': ['0', '1', '2'], 'score': 0, 'method': 'Markov (insufficient data)'}
        
        # Build transition matrix: last_digit → {next_digit: count}
        transition = defaultdict(lambda: defaultdict(int))
        
        for idx in range(len(history) - 1):
            curr = history[idx]
            next_num = history[idx + 1]
            
            if len(curr) >= 5 and len(next_num) >= 5:
                last_digit = curr[-1]
                for digit in next_num:
                    transition[last_digit][digit] += 1
        
        # Calculate most likely next digits based on recent last digits
        recent_last_digits = [h[-1] for h in history[-10:] if len(h) >= 5]
        
        next_prob = defaultdict(float)
        for last_d in recent_last_digits:
            if last_d in transition:
                total = sum(transition[last_d].values())
                if total > 0:
                    for next_d, count in transition[last_d].items():
                        next_prob[next_d] += count / total
        
        # Get top candidates
        sorted_items = sorted(next_prob.items(), key=lambda x: x[1], reverse=True)
        top_3 = [str(x[0]) for x in sorted_items[:3]]
        
        while len(top_3) < 3:
            for i in range(10):
                if str(i) not in top_3:
                    top_3.append(str(i))
                    break
        
        return {
            'top_3': top_3,
            'transition_strength': sum(next_prob.get(n, 0) for n in top_3),
            'score': sum(next_prob.values()) / max(1, len(next_prob)),
            'method': 'Markov Chain'
        }
    
    # ==========================================================================
    # AI LAYER 4: NEURAL-LITE (Lightweight Scoring Network)
    # ==========================================================================
    
    def _layer_neural(self, history):
        """
        Layer 4: Lightweight neural-inspired scoring.
        Not a full NN, but uses weighted feature scoring.
        """
        recent = history[-50:] if len(history) >= 50 else history
        
        # Features for each digit (0-9)
        digit_scores = defaultdict(float)
        
        for digit in map(str, range(10)):
            score = 0
            
            # Feature 1: Recent appearance (last 10)
            recent_count = sum(1 for n in recent[-10:] if digit in n)
            score += recent_count * 2.0
            
            # Feature 2: Position diversity
            positions = set()
            for n in recent:
                for pos, d in enumerate(n[:5]):
                    if d == digit:
                        positions.add(pos)
            score += len(positions) * 1.5
            
            # Feature 3: Gap analysis (time since last appearance)
            gaps = []
            last_seen = -1
            for idx, n in enumerate(recent):
                if digit in n:
                    if last_seen >= 0:
                        gaps.append(idx - last_seen)
                    last_seen = idx
            if gaps:
                avg_gap = sum(gaps) / len(gaps)
                # Prefer digits with moderate gaps (not too hot, not too cold)
                if 2 <= avg_gap <= 5:
                    score += 3.0
                elif avg_gap > 8:
                    score += 1.5  # Due number
            
            # Feature 4: Pair co-occurrence
            for n in recent:
                if digit in n:
                    other_digits = [d for d in n if d != digit]
                    for od in other_digits:
                        # Boost if appears with other frequent digits
                        if sum(1 for x in recent if od in x) > 3:
                            score += 0.3
            
            digit_scores[digit] = score
        
        # Get top 3
        sorted_items = sorted(digit_scores.items(), key=lambda x: x[1], reverse=True)
        top_3 = [str(x[0]) for x in sorted_items[:3]]
        
        while len(top_3) < 3:
            for i in range(10):
                if str(i) not in top_3:
                    top_3.append(str(i))
                    break
        
        return {
            'top_3': top_3,
            'feature_scores': {k: round(v, 2) for k, v in sorted_items[:10]},
            'score': sum(digit_scores.get(n, 0) for n in top_3) / 3,
            'method': 'Neural-Lite Scoring'
        }
    
    # ==========================================================================
    # ENSEMBLE & UTILITIES
    # ==========================================================================
    
    def _ensemble_vote(self, results):
        """
        Ensemble voting: Combine all layer results with adaptive weights.
        """
        all_votes = Counter()
        avoid_votes = []
        
        for layer_name, layer_result in results.items():
            weight = self.weights.get(layer_name, 10)
            
            # Add votes for top candidates
            for num in layer_result.get('top_3', []):
                all_votes[num] += weight
            
            # Collect avoid numbers
            for num in layer_result.get('avoid', []):
                avoid_votes.append(num)
        
        # Get top 3 (excluding avoided numbers)
        avoid_set = set(avoid_votes)
        final_3 = []
        
        for num, count in all_votes.most_common():
            if num not in final_3 and num not in avoid_set:
                final_3.append(num)
            if len(final_3) == 3:
                break
        
        # Fill if needed
        while len(final_3) < 3:
            for i in range(10):
                if str(i) not in final_3 and str(i) not in avoid_set:
                    final_3.append(str(i))
                    break
        
        # Support 4: Next best candidates
        remaining = [n for n, c in all_votes.most_common(10) if n not in final_3 and n not in avoid_set]
        support_4 = remaining[:4]
        while len(support_4) < 4:
            for i in range(10):
                if str(i) not in final_3 and str(i) not in support_4 and str(i) not in avoid_set:
                    support_4.append(str(i))
                    break
        
        return {
            'main_3': final_3,
            'support_4': support_4,
            'avoid': list(avoid_set),
            'vote_counts': dict(all_votes.most_common(10))
        }
    
    def _calculate_confidence(self, results, ensemble_result):
        """Calculate confidence based on layer agreement."""
        if not results:
            return 50
        
        # Check agreement among layers
        all_top = []
        for layer_result in results.values():
            all_top.extend(layer_result.get('top_3', []))
        
        # Count how many times ensemble picks appear in layer picks
        agreement = sum(1 for num in ensemble_result['main_3'] if all_top.count(num) >= 2)
        
        # Base confidence + agreement bonus
        base_conf = 60
        agreement_bonus = agreement * 10
        confidence = min(95, base_conf + agreement_bonus)
        
        # Reduce if avoid numbers are in top votes
        if ensemble_result.get('avoid'):
            confidence = max(40, confidence - 15)
        
        return confidence
    
    def _build_logic(self, results, ensemble_result):
        """Build human-readable explanation."""
        parts = []
        
        if results.get('frequency', {}).get('top_3'):
            parts.append(f"Tần suất: {','.join(results['frequency']['top_3'][:3])}")
        
        if results.get('pattern', {}).get('patterns_found'):
            count = len(results['pattern']['patterns_found'])
            parts.append(f"{count} pattern phát hiện")
        
        if results.get('neural', {}).get('feature_scores'):
            top_features = list(results['neural']['feature_scores'].items())[:2]
            if top_features:
                parts.append(f"Score cao: {','.join([f[0] for f in top_features])}")
        
        if ensemble_result.get('avoid'):
            parts.append(f"⚠️ Tránh: {','.join(ensemble_result['avoid'])}")
        
        return ' | '.join(parts) if parts else 'Phân tích đa tầng AI'
    
    def _find_max_streak(self, seq):
        """Find maximum consecutive streak in sequence."""
        if not seq:
            return 0
        max_streak = 1
        current = 1
        for i in range(1, len(seq)):
            if seq[i] == seq[i-1]:
                current += 1
                max_streak = max(max_streak, current)
            else:
                current = 1
        return max_streak
    
    def _calculate_entropy(self, freq, total):
        """Calculate Shannon entropy."""
        if total == 0:
            return 0
        entropy = 0
        for count in freq.values():
            if count > 0:
                p = count / total
                entropy -= p * math.log2(p)
        return entropy
    
    def _fallback_prediction(self, error_msg):
        """Return safe fallback prediction."""
        return {
            'main_3': ['?', '?', '?'],
            'support_4': ['0', '0', '0', '0'],
            'confidence': 0,
            'logic': error_msg,
            'avoid': [],
            'layer_scores': {},
            'algorithm': 'Fallback'
        }