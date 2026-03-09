# ==============================================================================
# TITAN AI v5.0 - Configuration
# Professional Production Configuration
# ==============================================================================

class Config:
    """Application configuration - DO NOT MODIFY WITHOUT TESTING."""
    
    # App Settings
    APP_TITLE = "🎯 TITAN AI v5.0"
    APP_SUBTITLE = "House Pattern Detector - Professional Edition"
    PAGE_ICON = "🔍"
    
    # Data Settings
    MIN_HISTORY_LENGTH = 15
    MAX_HISTORY_LENGTH = 500
    DEFAULT_SIMULATIONS = 2000
    
    # Algorithm Weights (must sum to 100)
    ALGORITHM_WEIGHTS = {
        'frequency': 25,
        'gap': 20,
        'markov': 20,
        'monte_carlo': 15,
        'pattern': 12,
        'hot_cold': 8
    }
    
    # Risk Thresholds
    RISK_THRESHOLDS = {
        'entropy_min': 2.8,
        'entropy_max': 3.4,
        'max_streak': 5,
        'sum_std_max': 2.5
    }
    
    # House Control Thresholds
    HOUSE_CONTROL = {
        'low': 30,
        'medium': 50,
        'high': 70
    }
    
    # Pattern Detection Weights
    PATTERN_CONFIG = {
        'bet_cau_weight': 0.30,
        'dao_cau_weight': 0.20,
        'xoay_cau_weight': 0.20,
        'nhip_bay_weight': 0.15,
        'tong_control_weight': 0.15
    }
    
    # UI Theme Colors
    THEME = {
        'primary_color': '#1e3a8a',
        'secondary_color': '#7c3aed',
        'success_color': '#059669',
        'warning_color': '#d97706',
        'danger_color': '#dc2626'
    }
    
    # Validation
    @classmethod
    def validate(cls):
        """Validate configuration integrity."""
        weight_sum = sum(cls.ALGORITHM_WEIGHTS.values())
        if weight_sum != 100:
            raise ValueError(f"ALGORITHM_WEIGHTS must sum to 100, got {weight_sum}")
        
        pattern_sum = sum(cls.PATTERN_CONFIG.values())
        if abs(pattern_sum - 1.0) > 0.01:
            raise ValueError(f"PATTERN_CONFIG must sum to 1.0, got {pattern_sum}")
        
        return True