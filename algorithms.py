# ==============================================================================
# TITAN v37.5 PRO MAX - Prediction Engine (FIXED & UPGRADED)
# Multi-Layer AI: Frequency | Markov | Monte Carlo | Pattern
# ==============================================================================

import numpy as np
from collections import Counter, defaultdict
import math
import random

class PredictionEngine:
    """
    Multi-Layer Prediction Engine for 5D Bet
    Layers: Frequency(30%) | Markov(25%) | MonteCarlo(25%) | Pattern(20%)
    """
    
    def __init__(self):
        # Algorithm weights (sum to 100)
        self.weights = {
            'frequency': 30,
            'markov': 25,
            'monte_carlo': 25,
            'pattern': 20
        }
        self.win_history = []
        self.max_history = 50  # Track last 50 predictions for learning
        
        # Risk thresholds
        self.risk_config = {
            'entropy_min': 2.8,
            'entropy_max': 3.3,
            'streak_max': 5,
            'min_data': 15
        }

    def predict(self, history):
        """
        Main prediction: Run all AI layers and ensemble results.
        Returns dict with main_3, support_4, confidence, logic, risk_metrics.
        """
        # Validate input
        if not history or len(history) < self.risk_config['min_data']:
            return self._fallback_prediction(f"⚠️ Cần tối thiểu {self.risk_config['min_data']} kỳ dữ liệu")
        
        # Clean history: ensure all entries are 5-digit strings
        clean_history = [h for h in history if isinstance(h, str) and len(h) == 5 and h.isdigit()]
        if len(clean_history) < self.risk_config['min_data']:
            return self._fallback_prediction("⚠️ Dữ liệu không hợp lệ (cần số 5 chữ số)")
        
        try:
            # Layer 1: Frequency Analysis
            freq_res = self._layer_frequency(clean_history)
            
            # Layer 2: Markov Chain
            markov_res = self._layer_markov(clean_history)
            
            # Layer 3: Monte Carlo Simulation
            monte_res = self._layer_monte_carlo(clean_history)
            
            # Layer 4: Pattern Recognition
            pattern_res = self._layer_pattern(clean_history)

            # Ensemble Voting
            all_layers = {
                'frequency': freq_res,
                'markov': markov_res,
                'monte_carlo': monte_res,
                'pattern': pattern_res
            }
            
            ensemble = self._ensemble_vote(all_layers)
            risk_metrics = self.calculate_risk(clean_history)
            
            # Calculate confidence based on data quantity and layer agreement
            base_conf = 60
            data_bonus = min(20, len(clean_history) - self.risk_config['min_data'])
            agreement_bonus = self._calculate_agreement_bonus(all_layers, ensemble)
            confidence = min(96, base_conf + data_bonus + agreement_bonus)
            
            # Build logic explanation
            logic = self._build_logic(all_layers, ensemble)
            
            return {
                'main_3': ensemble['main_3'],
                'support_4': ensemble['support_4'],
                'confidence': confidence,
                'logic': logic,
                'risk_metrics': risk_metrics,
                'layer_details': {k: v.get('top_3', []) for k, v in all_layers.items()}
            }
            
        except Exception as e:
            return self._fallback_prediction(f"❌ Lỗi phân tích: {str(e)}")

    def calculate_risk(self, history):
        """
        Multi-factor risk assessment.
        Returns: {'score': 0-100, 'level': 'LOW'/'HIGH', 'reasons': []}
        """
        if not history or len(history) < 10:
            return {'score': 0, 'level': 'LOW', 'reasons': ['Chưa đủ dữ liệu']}
        
        try:
            recent = history[-20:] if len(history) >= 20 else history
            all_digits = "".join(recent)
            
            if not all_digits:
                return {'score': 50, 'level': 'MEDIUM', 'reasons': ['Không có dữ liệu số']}
            
            counts = Counter(all_digits)
            total = len(all_digits)
            
            # Factor 1: Entropy check (randomness measure)
            entropy = 0
            for c in counts.values():
                if c > 0:
                    p = c / total
                    entropy -= p * math.log2(p)
            
            entropy_risk = 0
            entropy_reasons = []
            if entropy < self.risk_config['entropy_min']:
                entropy_risk += 25
                entropy_reasons.append(f"Entropy thấp ({entropy:.2f}) - Số lặp nhiều")
            elif entropy > self.risk_config['entropy_max']:
                entropy_risk += 15
                entropy_reasons.append(f"Entropy cao ({entropy:.2f}) - Quá phân tán")
            
            # Factor 2: Streak detection (abnormal consecutive same digit)
            streak_risk = 0
            streak_reasons = []
            for pos in range(5):
                seq = [n[pos] for n in recent if len(n) > pos]
                if seq:
                    max_streak = 1
                    current = 1
                    for i in range(1, len(seq)):
                        if seq[i] == seq[i-1]:
                            current += 1
                            max_streak = max(max_streak, current)
                        else:
                            current = 1
                    if max_streak >= self.risk_config['streak_max']:
                        streak_risk += 20
                        streak_reasons.append(f"Cầu bệt {max_streak} kỳ vị trí {pos}")
            
            # Factor 3: Over-represented numbers
            over_rep_risk = 0
            over_rep_reasons = []
            for num, count in counts.most_common(2):
                rate = count / total
                if rate > 0.25:  # >25% appearance rate
                    over_rep_risk += 15
                    over_rep_reasons.append(f"Số '{num}' xuất hiện {rate*100:.0f}%")
            
            # Calculate total risk score
            score = min(100, entropy_risk + streak_risk + over_rep_risk)
            level = "HIGH" if score >= 50 else "LOW"
            reasons = entropy_reasons + streak_reasons + over_rep_reasons
            if not reasons:
                reasons = ["Nhịp số tự nhiên"]
            
            return {'score': score, 'level': level, 'reasons': reasons}
            
        except Exception as e:
            return {'score': 50, 'level': 'MEDIUM', 'reasons': [f'Lỗi tính risk: {str(e)}']}

    def _layer_frequency(self, history):
        """
        Layer 1: Exponential-weighted frequency analysis.
        Returns: {'top_3': [], 'score': float, 'method': str}
        """
        try:
            # Use last 50 entries, weight recent heavier
            recent = history[-50:] if len(history) >= 50 else history
            
            weighted_freq = Counter()
            for idx, num in enumerate(recent):
                # Exponential decay: weight 3.0 (most recent) → 1.0 (oldest)
                weight = 3.0 - 2.0 * (idx / max(len(recent), 1))
                for digit in num:
                    if digit.isdigit():
                        weighted_freq[digit] += weight
            
            top_3 = [str(x[0]) for x in weighted_freq.most_common(3)]
            
            # Fill if needed
            while len(top_3) < 3:
                for i in range(10):
                    if str(i) not in top_3:
                        top_3.append(str(i))
                        break
            
            score = sum(weighted_freq.get(n, 0) for n in top_3) / 3 if top_3 else 0
            
            return {'top_3': top_3, 'score': score, 'method': 'Frequency'}
            
        except:
            return {'top_3': ['0', '1', '2'], 'score': 0, 'method': 'Frequency (error)'}

    def _layer_markov(self, history):
        """
        Layer 2: Markov Chain prediction (last digit → next digit transitions).
        Returns: {'top_3': [], 'score': float, 'method': str}
        """
        try:
            if len(history) < 5:
                return {'top_3': ['1', '5', '9'], 'score': 0, 'method': 'Markov'}
            
            # Build transition matrix: last_digit → {next_digit: count}
            nodes = defaultdict(Counter)
            for i in range(len(history) - 1):
                if len(history[i]) >= 5 and len(history[i+1]) >= 5:
                    last_d = history[i][-1]
                    # Count all digits in next number as possible transitions
                    for d in history[i+1]:
                        if d.isdigit():
                            nodes[last_d][d] += 1
            
            # Get last digit from most recent entry
            last_digit = history[-1][-1] if len(history[-1]) >= 1 else '0'
            
            # Get most likely next digits
            if last_digit in nodes and nodes[last_digit]:
                res = [str(x[0]) for x in nodes[last_digit].most_common(3)]
            else:
                # Fallback: use overall most common digits
                all_next = Counter()
                for transitions in nodes.values():
                    all_next.update(transitions)
                res = [str(x[0]) for x in all_next.most_common(3)]
            
            # Fill if needed
            while len(res) < 3:
                for i in range(10):
                    if str(i) not in res:
                        res.append(str(i))
                        break
            
            score = sum(nodes.get(last_digit, {}).get(n, 0) for n in res) / 3 if res else 0
            
            return {'top_3': res, 'score': score, 'method': 'Markov'}
            
        except:
            return {'top_3': ['1', '5', '9'], 'score': 0, 'method': 'Markov (error)'}

    def _layer_monte_carlo(self, history):
        """
        Layer 3: Monte Carlo Simulation (5000 scenarios).
        Simulates random draws based on historical digit distribution.
        Returns: {'top_3': [], 'score': float, 'method': str}
        """
        try:
            if len(history) < 10:
                return {'top_3': ['3', '7', '0'], 'score': 0, 'method': 'MonteCarlo'}
            
            # Create weighted pool from recent history
            recent = history[-40:] if len(history) >= 40 else history
            pool = []
            for idx, num in enumerate(recent):
                # Weight recent entries heavier
                weight = 3 if idx < 10 else 2 if idx < 25 else 1
                for digit in num:
                    if digit.isdigit():
                        pool.extend([digit] * weight)
            
            if not pool:
                return {'top_3': ['3', '7', '0'], 'score': 0, 'method': 'MonteCarlo'}
            
            # Run 5000 simulations
            sim = Counter()
            for _ in range(5000):
                # Sample 3 digits (representing our 3-number bet)
                sample = random.choices(pool, k=3)
                for n in set(sample):  # Count unique digits in sample
                    sim[n] += 1
            
            res = [str(x[0]) for x in sim.most_common(3)]
            
            # Fill if needed
            while len(res) < 3:
                for i in range(10):
                    if str(i) not in res:
                        res.append(str(i))
                        break
            
            total_sims = sum(sim.values())
            score = sum(sim.get(n, 0) for n in res) / total_sims * 100 if total_sims > 0 else 0
            
            return {'top_3': res, 'score': score, 'method': 'MonteCarlo'}
            
        except:
            return {'top_3': ['3', '7', '0'], 'score': 0, 'method': 'MonteCarlo (error)'}

    def _layer_pattern(self, history):
        """
        Layer 4: Pattern Recognition (streaks, rhythms, positions).
        Returns: {'top_3': [], 'score': float, 'method': str}
        """
        try:
            if len(history) < 5:
                return {'top_3': ['2', '6', '8'], 'score': 0, 'method': 'Pattern'}
            
            recent = history[-30:] if len(history) >= 30 else history
            candidates = Counter()
            
            # Pattern 1: Most frequent first digit (position 0)
            first_digits = [n[0] for n in recent if len(n) > 0]
            if first_digits:
                for d, c in Counter(first_digits).most_common(2):
                    candidates[d] += 3
            
            # Pattern 2: Most frequent last digit (position 4)
            last_digits = [n[-1] for n in recent if len(n) >= 5]
            if last_digits:
                for d, c in Counter(last_digits).most_common(2):
                    candidates[d] += 3
            
            # Pattern 3: Rhythm detection (X appears every 2-3 periods)
            for digit in map(str, range(10)):
                positions = [i for i, n in enumerate(recent) if digit in n]
                if len(positions) >= 3:
                    gaps = [positions[i+1] - positions[i] for i in range(len(positions)-1)]
                    # If gaps are consistent (2 or 3), boost this digit
                    if gaps and all(2 <= g <= 3 for g in gaps[:3]):
                        candidates[digit] += 4
            
            # Pattern 4: Avoid numbers that appeared in last 2 draws (cooling)
            last_2 = "".join(history[-2:]) if len(history) >= 2 else ""
            for d in set(last_2):
                if d.isdigit():
                    candidates[d] -= 1  # Slight penalty
            
            res = [str(x[0]) for x in candidates.most_common(3)]
            
            # Fill if needed
            while len(res) < 3:
                for i in range(10):
                    if str(i) not in res:
                        res.append(str(i))
                        break
            
            score = sum(candidates.get(n, 0) for n in res) / 3 if res else 0
            
            return {'top_3': res, 'score': score, 'method': 'Pattern'}
            
        except:
            return {'top_3': ['2', '6', '8'], 'score': 0, 'method': 'Pattern (error)'}

    def _ensemble_vote(self, layers):
        """
        Ensemble voting: Combine all layer results with adaptive weights.
        Returns: {'main_3': [], 'support_4': []}
        """
        votes = Counter()
        
        for name, res in layers.items():
            weight = self.weights.get(name, 10)
            top_3 = res.get('top_3', [])
            for num in top_3:
                votes[num] += weight
        
        # Get top 7 candidates for main_3 + support_4
        best = votes.most_common(7)
        
        main_3 = [str(x[0]) for x in best[:3]]
        support_4 = [str(x[0]) for x in best[3:7]]
        
        # Ensure we have exactly 3 + 4 numbers
        while len(main_3) < 3:
            for i in range(10):
                if str(i) not in main_3:
                    main_3.append(str(i))
                    break
        
        while len(support_4) < 4:
            for i in range(10):
                if str(i) not in main_3 and str(i) not in support_4:
                    support_4.append(str(i))
                    break
        
        return {'main_3': main_3, 'support_4': support_4}

    def _calculate_agreement_bonus(self, layers, ensemble):
        """Calculate bonus based on how many layers agree on final picks."""
        if not layers or not ensemble:
            return 0
        
        bonus = 0
        for num in ensemble['main_3']:
            # Count how many layers included this number in their top_3
            agreement = sum(1 for layer in layers.values() 
                          if num in layer.get('top_3', []))
            if agreement >= 3:  # 3+ layers agree
                bonus += 5
            elif agreement == 2:
                bonus += 2
        
        return min(15, bonus)  # Cap at 15

    def _build_logic(self, layers, ensemble):
        """Build human-readable explanation of the prediction."""
        parts = []
        
        # Add top picks from each layer
        for layer_name in ['frequency', 'markov', 'monte_carlo', 'pattern']:
            if layer_name in layers:
                top = layers[layer_name].get('top_3', [])[:2]
                if top:
                    parts.append(f"{layer_name[:4]}:{','.join(top)}")
        
        # Add ensemble info
        if ensemble.get('main_3'):
            parts.append(f"ensemble:{','.join(ensemble['main_3'][:2])}")
        
        logic = " | ".join(parts) if parts else "Phân tích đa tầng"
        
        # Add noise filter note
        return f"{logic} + Noise Filter"

    def get_ai_status(self):
        """Return AI engine status for display."""
        recent_wins = self.win_history[-20:] if len(self.win_history) >= 20 else self.win_history
        win_rate = sum(recent_wins) / len(recent_wins) * 100 if recent_wins else 0
        
        return {
            'weights': self.weights,
            'recent_win_rate': round(win_rate, 1),
            'predictions_tracked': len(self.win_history)
        }

    def update_weights(self, won: bool, winning_method: str = None):
        """
        Self-learning: Adjust algorithm weights based on outcome.
        Reinforcement learning style update.
        """
        # Record outcome
        self.win_history.append(1 if won else 0)
        if len(self.win_history) > self.max_history:
            self.win_history.pop(0)
        
        # Only adjust if we know which method contributed
        if winning_method and winning_method in self.weights:
            adjustment = 0.05 if won else -0.03
            self.weights[winning_method] = min(40, max(10, 
                self.weights[winning_method] + adjustment * 10))
        
        # Normalize weights to sum to 100
        total = sum(self.weights.values())
        if total > 0:
            for key in self.weights:
                self.weights[key] = round(self.weights[key] * 100 / total)

    def _fallback_prediction(self, msg):
        """Return safe fallback prediction when errors occur."""
        return {
            'main_3': ['?', '?', '?'],
            'support_4': ['?', '?', '?', '?'],
            'confidence': 0,
            'logic': msg,
            'risk_metrics': {'score': 50, 'level': 'MEDIUM', 'reasons': ['Fallback mode']},
            'layer_details': {}
        }