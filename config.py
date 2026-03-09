# ==============================================================================
# TITAN AI v5.0 - Configuration
# ==============================================================================

class Config:
    """Application configuration."""
    
    # App Settings
    APP_TITLE = "🎯 TITAN AI v5.0"
    APP_SUBTITLE = "House Pattern Detector"
    PAGE_ICON = "🔍"
    
    # Data Settings
    MIN_HISTORY_LENGTH = 15
    MAX_HISTORY_LENGTH = 500
    DEFAULT_SIMULATIONS = 2000
    
    # Algorithm Weights
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
    
    @classmethod
    def validate(cls):
        """Validate configuration."""
        return True