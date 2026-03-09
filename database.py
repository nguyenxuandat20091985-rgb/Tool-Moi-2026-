# ==============================================================================
# TITAN AI v5.0 - Database Management
# Professional Data Management with Validation
# ==============================================================================

import re
from typing import List, Dict, Tuple, Optional
from datetime import datetime

class DatabaseManager:
    """Professional lottery data management."""
    
    def __init__(self):
        """Initialize database manager."""
        self.data: List[str] = []
        self.test_log: List[Dict] = []
        self._last_cleanup = datetime.now()
    
    def clean_data(self, raw_text: str) -> List[str]:
        """
        Clean and extract 5-digit numbers from raw text.
        
        Args:
            raw_text: Raw input text
            
        Returns:
            List of validated 5-digit numbers
        """
        if not raw_text or not isinstance(raw_text, str):
            return []
        
        # Extract all 5-digit sequences
        numbers = re.findall(r'\d{5}', raw_text)
        
        # Validate each number
        validated = []
        for num in numbers:
            if len(num) == 5 and num.isdigit():
                validated.append(num)
        
        return validated
    
    def add_numbers(self, new_numbers: List[str], 
                   existing_data: Optional[List[str]] = None) -> Tuple[List[str], int]:
        """
        Add new numbers to database with deduplication.
        
        Args:
            new_numbers: List of new numbers to add
            existing_data: Existing database (uses self.data if None)
            
        Returns:
            Tuple of (updated_data, count_added)
        """
        if existing_data is None:
            existing_data = self.data
        
        # Create set for O(1) lookup
        existing_set = set(existing_data)
        unique_new = []
        
        for num in new_numbers:
            # Validate number format
            if not isinstance(num, str) or len(num) != 5 or not num.isdigit():
                continue
            
            if num not in existing_set:
                unique_new.append(num)
                existing_set.add(num)
        
        # Add new numbers to front (newest first)
        updated_data = unique_new + existing_data
        
        # Enforce maximum size limit
        if len(updated_data) > 500:
            updated_data = updated_data[:500]
        
        self.data = updated_data
        return updated_data, len(unique_new)
    
    def record_test(self, prediction: List[str], actual: str, won: bool,
                   confidence: int, house_risk: int = 0) -> bool:
        """
        Record a test prediction for accuracy tracking.
        
        Args:
            prediction: List of 3 predicted numbers
            actual: Actual 5-digit result
            won: Whether prediction won
            confidence: Prediction confidence (0-100)
            house_risk: House control risk level (0-100)
            
        Returns:
            True if recorded successfully
        """
        try:
            # Validate inputs
            if not isinstance(prediction, list) or len(prediction) != 3:
                return False
            
            if not isinstance(actual, str) or len(actual) != 5:
                return False
            
            if not isinstance(confidence, int) or confidence < 0 or confidence > 100:
                return False
            
            self.test_log.append({
                'timestamp': datetime.now().isoformat(),
                'prediction': prediction,
                'actual': actual,
                'won': won,
                'confidence': confidence,
                'house_risk': house_risk
            })
            
            # Keep last 100 tests only
            if len(self.test_log) > 100:
                self.test_log = self.test_log[-100:]
            
            return True
            
        except Exception:
            return False
    
    def get_accuracy_stats(self) -> Dict:
        """
        Calculate comprehensive accuracy statistics.
        
        Returns:
            Dictionary with accuracy metrics
        """
        if not self.test_log:
            return {
                'total': 0,
                'wins': 0,
                'losses': 0,
                'win_rate': 0.0,
                'avg_confidence': 0.0,
                'by_confidence': {},
                'by_house_risk': {}
            }
        
        total = len(self.test_log)
        wins = sum(1 for t in self.test_log if t.get('won', False))
        losses = total - wins
        win_rate = (wins / total * 100) if total > 0 else 0.0
        avg_conf = sum(t.get('confidence', 0) for t in self.test_log) / total
        
        # By confidence bracket
        by_confidence = {}
        for bracket in ['50-69', '70-84', '85+']:
            if bracket == '50-69':
                subset = [t for t in self.test_log if 50 <= t.get('confidence', 0) < 70]
            elif bracket == '70-84':
                subset = [t for t in self.test_log if 70 <= t.get('confidence', 0) < 85]
            else:
                subset = [t for t in self.test_log if t.get('confidence', 0) >= 85]
            
            if subset:
                w = sum(1 for t in subset if t.get('won', False))
                by_confidence[bracket] = {
                    'count': len(subset),
                    'wins': w,
                    'win_rate': round(w / len(subset) * 100, 1)
                }
        
        # By house risk level
        by_house_risk = {}
        for bracket in ['0-29', '30-49', '50+']:
            if bracket == '0-29':
                subset = [t for t in self.test_log if t.get('house_risk', 0) < 30]
            elif bracket == '30-49':
                subset = [t for t in self.test_log if 30 <= t.get('house_risk', 0) < 50]
            else:
                subset = [t for t in self.test_log if t.get('house_risk', 0) >= 50]
            
            if subset:
                w = sum(1 for t in subset if t.get('won', False))
                by_house_risk[bracket] = {
                    'count': len(subset),
                    'wins': w,
                    'win_rate': round(w / len(subset) * 100, 1)
                }
        
        return {
            'total': total,
            'wins': wins,
            'losses': losses,
            'win_rate': round(win_rate, 1),
            'avg_confidence': round(avg_conf, 1),
            'by_confidence': by_confidence,
            'by_house_risk': by_house_risk
        }
    
    def get_recent_tests(self, limit: int = 20) -> List[Dict]:
        """Get recent test results."""
        return self.test_log[-limit:] if self.test_log else []
    
    def clear(self) -> None:
        """Clear all data safely."""
        self.data = []
        self.test_log = []
        self._last_cleanup = datetime.now()
    
    def export_data(self) -> Dict:
        """Export all data for backup."""
        return {
            'data': self.data,
            'test_log': self.test_log,
            'exported_at': datetime.now().isoformat(),
            'version': '5.0'
        }
    
    def import_data(self, data_dict: Dict) -> bool:
        """Import data from backup."""
        try:
            if not isinstance(data_dict, dict):
                return False
            
            if 'data' in data_dict and isinstance(data_dict['data'], list):
                self.data = data_dict['data'][:500]
            
            if 'test_log' in data_dict and isinstance(data_dict['test_log'], list):
                self.test_log = data_dict['test_log'][-100:]
            
            return True
            
        except Exception:
            return False
    
    def get_statistics(self) -> Dict:
        """Get database statistics."""
        if not self.data:
            return {
                'total_numbers': 0,
                'total_tests': 0,
                'oldest_entry': None,
                'newest_entry': None
            }
        
        return {
            'total_numbers': len(self.data),
            'total_tests': len(self.test_log),
            'oldest_entry': self.data[-1] if self.data else None,
            'newest_entry': self.data[0] if self.data else None
        }