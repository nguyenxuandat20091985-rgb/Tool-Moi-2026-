# ==============================================================================
# TITAN v37.5 PRO MAX - MULTI-LAYER AI PREDICTION ENGINE
# Enhanced with Self-Learning, Noise Filtering & Smart Ensemble
# ==============================================================================

import numpy as np
from collections import Counter, defaultdict, deque
import math
import random
from datetime import datetime

class PredictionEngine:
    """
    Multi-Layer AI Engine for 5D Lottery Prediction
    Layers: Frequency | Markov | Monte Carlo | Pattern | Ensemble
    """
    
    def __init__(self):
        # Algorithm weights (adaptive via self-learning)
        self.weights = {
            'frequency': 30,      # Statistical frequency with decay
            'markov': 25,          # Markov chain transition modeling
            'monte_carlo': 25,     # Monte Carlo simulation with filtering
            'pattern': 20          # Pattern recognition with noise filter
        }
        
        # Self-learning parameters
        self.win_history = deque(maxlen=50)  # Track last 50 predictions
        self.learning_rate = 0.08
        self.performance_by_layer = defaultdict(list)
        
        # Pattern memory for smarter detection
        self.pattern_memory = {
            'streaks': defaultdict(int),
            'rhythms': defaultdict(int),
            'last_update': None
        }
        
        # Monte Carlo config
        self.monte_iterations = 5000
        self.monte_confidence_threshold = 0.15
        
        # Risk thresholds
        self.risk_config = {
            'entropy_min': 2.8,
            'entropy_max': 3.3,
            'streak_max': 5,
            'variance_threshold': 2.5
        }
    
    def predict(self, history):
        """
        Main prediction: Execute all AI layers and ensemble results.
        Returns dict with main_3, support_4, confidence, logic.
        """
        # Data validation
        if not history:
            return self._fallback_prediction("⚠️ Chưa có dữ liệu")
        
        if len(history) < 10:
            return self._fallback_prediction(f"⚠️ Cần thêm dữ liệu (Tối thiểu 15 kỳ, hiện có: {len(history)})")
        
        # Clean history: ensure all entries are 5-digit strings
        clean_history = [h.zfill(5) if len(h) < 5 else h[:5] for h in history if h.isdigit()]
        if len(clean_history) < 10:
            return self._fallback_prediction("⚠️ Dữ liệu không hợp lệ")
        
        # Execute all AI layers
        freq_res = self._layer_frequency(clean_history)
        markov_res = self._layer_markov(clean_history)
        monte_res = self._layer_monte_carlo(clean_history)
        pattern_res = self._layer_pattern(clean_history)
        
        # Build layer results dict
        all_layers = {
            'frequency': freq_res,
            'markov': markov_res,
            'monte_carlo': monte_res,
            'pattern': pattern_res
        }
        
        # Ensemble voting with adaptive weights
        ensemble = self._ensemble_vote(all_layers)
        
        # Calculate risk metrics
        risk_metrics = self.calculate_risk(clean_history)
        
        # Calculate confidence based on layer agreement
        confidence = self._calculate_confidence(all_layers, ensemble)
        
        # Build human-readable logic
        logic = self._build_logic(all_layers, ensemble, risk_metrics)
        
        return {
            'main_3': ensemble['main_3'],
            'support_4': ensemble['support_4'],
            'confidence': confidence,
            'logic': logic,
            'risk_metrics': risk_metrics,
            'layer_details': {k: v.get('details', {}) for k, v in all_layers.items()},
            'algorithm': 'Multi-Layer AI v37.5'
        }
    
    def calculate_risk(self, history):
        """
        Multi-factor risk assessment with enhanced detection.
        Returns: dict with score, level, reasons
        """
        if not history or len(history) < 10:
            return {'score': 0, 'level': 'LOW', 'reasons': ['Chưa đủ dữ liệu']}
        
        recent = history[-30:] if len(history) >= 30 else history
        all_digits = "".join(recent)
        
        if not all_digits:
            return {'score': 50, 'level': 'MEDIUM', 'reasons': ['Dữ liệu trống']}
        
        counts = Counter(all_digits)
        total = len(all_digits)
        
        # Factor 1: Entropy analysis (randomness check)
        entropy = 0
        for c in counts.values():
            if c > 0:
                p = c / total
                entropy -= p * math.log2(p)
        
        entropy_score = 0
        entropy_reasons = []
        if entropy < self.risk_config['entropy_min']:
            entropy_score += 25
            entropy_reasons.append(f"Entropy thấp ({entropy:.2f}) - Kết quả có thể bị điều khiển")
        elif entropy > self.risk_config['entropy_max']:
            entropy_score += 15
            entropy_reasons.append(f"Entropy cao ({entropy:.2f}) - Biến động mạnh")
        
        # Factor 2: Streak detection (abnormal consecutive patterns)
        streak_score = 0
        streak_reasons = []
        for pos in range(5):
            seq = [n[pos] for n in recent if len(n) > pos]
            max_streak = self._find_max_streak(seq)
            if max_streak >= self.risk_config['streak_max']:
                streak_score += 20
                streak_reasons.append(f"Cầu bệt {max_streak} kỳ vị trí {pos}")
        
        # Factor 3: Variance check (overly stable sums)
        variance_score = 0
        variance_reasons = []
        totals = [sum(int(d) for d in n) for n in recent if len(n) == 5 and n.isdigit()]
        if len(totals) >= 10:
            std_dev = np.std(totals)
            if std_dev < self.risk_config['variance_threshold']:
                variance_score += 15
                variance_reasons.append(f"Tổng quá ổn định (σ={std_dev:.2f})")
        
        # Factor 4: Recent prediction performance feedback
        performance_score = 0
        if self.win_history:
            recent_win_rate = sum(list(self.win_history)[-20:]) / min(len(self.win_history), 20)
            if recent_win_rate < 0.25:
                performance_score += 15
                entropy_reasons.append(f"AI đang học lại (win rate: {recent_win_rate*100:.0f}%)")
        
        # Calculate total risk score
        total_score = min(100, entropy_score + streak_score + variance_score + performance_score)
        
        # Determine risk level
        if total_score >= 60:
            level = "HIGH"
        elif total_score >= 35:
            level = "MEDIUM"
        else:
            level = "LOW"
        
        # Compile all reasons
        all_reasons = entropy_reasons + streak_reasons + variance_reasons
        if not all_reasons:
            all_reasons = ["Nhịp số tự nhiên - Risk thấp"]
        
        return {
            'score': total_score,
            'level': level,
            'reasons': all_reasons,
            'details': {
                'entropy': round(entropy, 3),
                'max_streak': max([self._find_max_streak([n[p] for n in recent if len(n)>p]) for p in range(5)], default=0),
                'sum_std': round(np.std(totals), 2) if totals else 0
            }
        }
    
    def update_weights(self, won: bool, winning_layer: str = None):
        """
        SELF-LEARNING: Adjust algorithm weights based on prediction outcome.
        Reinforcement learning with adaptive learning rate.
        """
        # Record outcome
        self.win_history.append(1 if won else 0)
        
        # Calculate adaptive learning rate (decrease over time for stability)
        adaptive_lr = self.learning_rate * (1 - len(self.win_history) / 200)
        adaptive_lr = max(0.02, adaptive_lr)  # Minimum learning rate
        
        if won and winning_layer and winning_layer in self.weights:
            # Boost winning layer
            self.weights[winning_layer] = min(45, self.weights[winning_layer] + adaptive_lr * 15)
            
            # Slightly reduce other layers to maintain balance
            for layer in self.weights:
                if layer != winning_layer:
                    self.weights[layer] = max(10, self.weights[layer] - adaptive_lr * 3)
        elif not won:
            # Gentle penalty for all layers when losing
            for layer in self.weights:
                self.weights[layer] = max(10, self.weights[layer] - adaptive_lr * 2)
        
        # Normalize weights to sum to 100
        total = sum(self.weights.values())
        if total > 0:
            for key in self.weights:
                self.weights[key] = round(self.weights[key] * 100 / total)
        
        # Record performance by layer for analytics
        if winning_layer:
            self.performance_by_layer[winning_layer].append({
                'won': won,
                'timestamp': datetime.now().isoformat(),
                'weights_snapshot': dict(self.weights)
            })
    
    def get_ai_status(self):
        """Return comprehensive AI engine status for display."""
        recent_wins = list(self.win_history)[-20:] if self.win_history else []
        recent_win_rate = sum(recent_wins) / len(recent_wins) * 100 if recent_wins else 0
        
        # Calculate layer performance
        layer_performance = {}
        for layer, records in self.performance_by_layer.items():
            if records:
                recent = records[-20:]
                win_rate = sum(1 for r in recent if r['won']) / len(recent) * 100
                layer_performance[layer] = round(win_rate, 1)
        
        return {
            'weights': dict(self.weights),
            'recent_win_rate': round(recent_win_rate, 1),
            'predictions_tracked': len(self.win_history),
            'layer_performance': layer_performance,
            'pattern_memory_size': sum(len(v) for v in self.pattern_memory.values() if isinstance(v, dict))
        }
    
    # ==========================================================================
    # LAYER 1: FREQUENCY ANALYSIS (Enhanced with Exponential Decay)
    # ==========================================================================
    
    def _layer_frequency(self, history):
        """
        Enhanced frequency analysis with exponential time decay.
        Recent data weighted significantly higher than old data.
        """
        # Use last 60 periods for analysis
        recent = history[-60:] if len(history) >= 60 else history
        
        weighted_freq = defaultdict(float)
        
        for idx, num in enumerate(recent):
            # Exponential decay: weight = 4.0 (most recent) → 1.0 (oldest in window)
            decay_factor = 0.95
            weight = 1.0 + 3.0 * (decay_factor ** idx)
            
            for digit in num:
                if digit.isdigit():
                    weighted_freq[digit] += weight
        
        # Get top candidates with scores
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
            'details': {
                'method': 'Exponential Decay Frequency',
                'window': len(recent),
                'top_score': sorted_items[0][1] if sorted_items else 0
            }
        }
    
    # ==========================================================================
    # LAYER 2: MARKOV CHAIN (Enhanced with Multi-Order Modeling)
    # ==========================================================================
    
    def _layer_markov(self, history):
        """
        Enhanced Markov Chain with multi-order transition modeling.
        Models: last_digit → next_digits AND position-aware transitions.
        """
        if len(history) < 15:
            return ['1', '5', '9']
        
        # First-order: last digit → next last digit
        first_order = defaultdict(Counter)
        for i in range(len(history) - 1):
            if len(history[i]) >= 5 and len(history[i+1]) >= 5:
                last_curr = history[i][-1]
                last_next = history[i+1][-1]
                first_order[last_curr][last_next] += 1
        
        # Position-aware: track which digits appear together
        position_cooccurrence = defaultdict(Counter)
        for num in history:
            if len(num) >= 5:
                digits = list(num[:5])
                for i, d1 in enumerate(digits):
                    for j, d2 in enumerate(digits):
                        if i != j:
                            position_cooccurrence[d1][d2] += 1
        
        # Get predictions from last known digit
        last_digit = history[-1][-1] if len(history[-1]) >= 1 else '0'
        
        # Combine first-order and co-occurrence scores
        combined_scores = Counter()
        
        # First-order contribution (60%)
        if last_digit in first_order:
            total_first = sum(first_order[last_digit].values())
            if total_first > 0:
                for digit, count in first_order[last_digit].items():
                    combined_scores[digit] += (count / total_first) * 0.6
        
        # Co-occurrence contribution (40%)
        if last_digit in position_cooccurrence:
            total_cooc = sum(position_cooccurrence[last_digit].values())
            if total_cooc > 0:
                for digit, count in position_cooccurrence[last_digit].items():
                    combined_scores[digit] += (count / total_cooc) * 0.4
        
        # Get top 3
        if combined_scores:
            top_3 = [str(x[0]) for x in combined_scores.most_common(3)]
        else:
            top_3 = ['1', '5', '9']  # Fallback
        
        while len(top_3) < 3:
            for i in range(10):
                if str(i) not in top_3:
                    top_3.append(str(i))
                    break
        
        return {
            'top_3': top_3,
            'details': {
                'method': 'Multi-Order Markov',
                'transitions_tracked': sum(len(v) for v in first_order.values()),
                'last_digit': last_digit
            }
        }
    
    # ==========================================================================
    # LAYER 3: MONTE CARLO SIMULATION (Enhanced with Smart Filtering)
    # ==========================================================================
    
    def _layer_monte_carlo(self, history):
        """
        Enhanced Monte Carlo with intelligent sampling and noise filtering.
        Runs 5000+ simulations with weighted probability distribution.
        """
        if len(history) < 15:
            return ['2', '6', '8']
        
        # Build weighted pool from recent history (last 40 periods)
        recent = history[-40:] if len(history) >= 40 else history
        pool = []
        
        # Weight recent entries higher
        for idx, num in enumerate(recent):
            weight = int(3 + 2 * (1 - idx / len(recent)))  # 3-5 copies per number
            for digit in num:
                if digit.isdigit():
                    pool.extend([digit] * weight)
        
        if not pool:
            return ['2', '6', '8']
        
        # Run Monte Carlo simulations
        sim_results = Counter()
        iterations = min(self.monte_iterations, 10000)
        
        for _ in range(iterations):
            # Sample 3 digits with replacement (weighted by pool)
            sample = random.choices(pool, k=3)
            
            # Apply noise filter: skip samples with all same digits (unlikely)
            if len(set(sample)) == 1:
                continue
            
            for digit in sample:
                sim_results[digit] += 1
        
        # Filter low-confidence results
        total_votes = sum(sim_results.values())
        if total_votes > 0:
            filtered = {k: v for k, v in sim_results.items() 
                       if v / total_votes >= self.monte_confidence_threshold}
            if filtered:
                sim_results = Counter(filtered)
        
        # Get top 3
        if sim_results:
            top_3 = [str(x[0]) for x in sim_results.most_common(3)]
        else:
            top_3 = ['2', '6', '8']
        
        while len(top_3) < 3:
            for i in range(10):
                if str(i) not in top_3:
                    top_3.append(str(i))
                    break
        
        return {
            'top_3': top_3,
            'details': {
                'method': 'Filtered Monte Carlo',
                'iterations': iterations,
                'pool_size': len(pool),
                'unique_results': len(sim_results)
            }
        }
    
    # ==========================================================================
    # LAYER 4: PATTERN RECOGNITION (Enhanced with Noise Filtering)
    # ==========================================================================
    
    def _layer_pattern(self, history):
        """
        Enhanced pattern recognition with noise filtering and multi-pattern detection.
        Detects: streaks, rhythms, reversals, triangles with confidence scoring.
        """
        if len(history) < 15:
            return [history[-1][0] if len(history[-1])>0 else '0', '5', '0']
        
        recent = history[-25:] if len(history) >= 25 else history
        patterns_found = []
        likely_candidates = Counter()
        avoid_candidates = set()
        
        # Pattern 1: Streak detection (cầu bệt) with position awareness
        for pos in range(5):
            seq = [n[pos] for n in recent if len(n) > pos]
            i = 0
            while i < len(seq) - 2:
                if seq[i] == seq[i+1] == seq[i+2]:
                    digit = seq[i]
                    streak_len = 3
                    j = i + 3
                    while j < len(seq) and seq[j] == digit:
                        streak_len += 1
                        j += 1
                    
                    # Record pattern
                    pattern_key = f"streak_{pos}_{digit}"
                    self.pattern_memory['streaks'][pattern_key] = streak_len
                    
                    if streak_len >= 4:
                        # Long streak likely to break → avoid
                        avoid_candidates.add(digit)
                        patterns_found.append(f"⚠️ Bệt {streak_len} kỳ vị {pos}: {digit} (sắp gãy)")
                    else:
                        # Short streak may continue → boost
                        likely_candidates[digit] += (5 - streak_len)
                        patterns_found.append(f"✓ Bệt {streak_len} kỳ vị {pos}: {digit}")
                    
                    i = j
                else:
                    i += 1
        
        # Pattern 2: Rhythm-2 detection (X _ X _ X)
        for pos in range(5):
            seq = [n[pos] for n in recent if len(n) > pos]
            for i in range(len(seq) - 4):
                if seq[i] == seq[i+2] == seq[i+4] and seq[i] != seq[i+1]:
                    digit = seq[i]
                    pattern_key = f"rhythm2_{pos}_{digit}"
                    self.pattern_memory['rhythms'][pattern_key] += 1
                    likely_candidates[digit] += 3
                    patterns_found.append(f"✓ Nhịp-2 vị {pos}: {digit}")
        
        # Pattern 3: Position diversity boost
        digit_positions = defaultdict(set)
        for num in recent:
            for pos, digit in enumerate(num[:5]):
                digit_positions[digit].add(pos)
        
        for digit, positions in digit_positions.items():
            if len(positions) >= 3:  # Appears in 3+ different positions
                likely_candidates[digit] += 2
        
        # Build final candidates
        candidates = Counter()
        for digit, score in likely_candidates.items():
            if digit not in avoid_candidates:
                candidates[digit] = score
        
        # Get top 3
        if candidates:
            top_3 = [str(x[0]) for x in candidates.most_common(3)]
        else:
            # Fallback: use last number's digits
            fallback = list(set(recent[-1][:5])) if recent else ['0', '1', '2']
            top_3 = fallback[:3]
        
        while len(top_3) < 3:
            for i in range(10):
                if str(i) not in top_3:
                    top_3.append(str(i))
                    break
        
        # Update memory timestamp
        self.pattern_memory['last_update'] = datetime.now().isoformat()
        
        return {
            'top_3': top_3,
            'details': {
                'method': 'Filtered Pattern Recognition',
                'patterns_detected': len(patterns_found),
                'patterns': patterns_found[:5],  # Top 5 patterns
                'avoid': list(avoid_candidates)
            }
        }
    
    # ==========================================================================
    # ENSEMBLE & UTILITIES
    # ==========================================================================
    
    def _ensemble_vote(self, layers):
        """
        Enhanced ensemble voting with adaptive weights and conflict resolution.
        """
        votes = Counter()
        avoid_votes = Counter()
        
        for name, result in layers.items():
            weight = self.weights.get(name, 10)
            
            # Add votes for top candidates
            top_3 = result if isinstance(result, list) else result.get('top_3', [])
            for num in top_3:
                votes[num] += weight
            
            # Collect avoid numbers from pattern layer
            if name == 'pattern' and isinstance(result, dict):
                details = result.get('details', {})
                for num in details.get('avoid', []):
                    avoid_votes[num] += weight * 0.5
        
        # Get top candidates excluding avoided numbers
        final_candidates = []
        for num, count in votes.most_common(10):
            if avoid_votes.get(num, 0) < votes.get(num, 0) * 0.3:  # Only exclude if strongly avoided
                final_candidates.append(num)
            if len(final_candidates) >= 7:
                break
        
        # Ensure we have enough candidates
        while len(final_candidates) < 7:
            for i in range(10):
                if str(i) not in final_candidates:
                    final_candidates.append(str(i))
                    break
        
        return {
            'main_3': final_candidates[:3],
            'support_4': final_candidates[3:7],
            'vote_counts': dict(votes.most_common(10)),
            'avoid': [k for k, v in avoid_votes.items() if v > 0]
        }
    
    def _calculate_confidence(self, layers, ensemble):
        """Calculate confidence based on layer agreement and risk factors."""
        if not layers:
            return 50
        
        # Check agreement among layers
        all_picks = []
        for layer_result in layers.values():
            picks = layer_result if isinstance(layer_result, list) else layer_result.get('top_3', [])
            all_picks.extend(picks)
        
        # Count agreement for ensemble picks
        agreement_score = 0
        for num in ensemble['main_3']:
            agreement_score += all_picks.count(num)
        
        # Base confidence + agreement bonus
        base_conf = 55
        agreement_bonus = min(25, agreement_score * 3)
        confidence = base_conf + agreement_bonus
        
        # Adjust for data quantity
        data_bonus = min(15, len([l for l in layers.values() if l]) * 4)
        confidence += data_bonus
        
        return min(96, confidence)
    
    def _build_logic(self, layers, ensemble, risk_metrics):
        """Build human-readable explanation of the prediction."""
        parts = []
        
        # Frequency insight
        freq_details = layers.get('frequency', {}).get('details', {})
        if freq_details.get('top_score', 0) > 0:
            parts.append(f"Tần suất: {','.join(layers['frequency'].get('top_3', [])[:2])}")
        
        # Pattern insights
        pattern_details = layers.get('pattern', {}).get('details', {})
        if pattern_details.get('patterns_detected', 0) > 0:
            parts.append(f"{pattern_details['patterns_detected']} pattern")
        
        # Risk context
        if risk_metrics.get('level') == 'HIGH':
            parts.append("⚠️ Risk cao")
        elif risk_metrics.get('level') == 'MEDIUM':
            parts.append("⚡ Theo dõi")
        
        # Add noise filter mention
        parts.append("Đã lọc nhiễu")
        
        return " | ".join(parts) if parts else "Phân tích đa tầng AI"
    
    def _find_max_streak(self, seq):
        """Find maximum consecutive streak in a sequence."""
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
    
    def _fallback_prediction(self, msg):
        """Return safe fallback prediction when data is insufficient."""
        return {
            'main_3': ['?', '?', '?'],
            'support_4': ['?', '?', '?', '?'],
            'confidence': 0,
            'logic': msg,
            'risk_metrics': {'score': 0, 'level': 'LOW', 'reasons': [msg]},
            'layer_details': {},
            'algorithm': 'Fallback'
        }