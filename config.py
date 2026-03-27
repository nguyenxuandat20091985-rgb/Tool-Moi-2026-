"""
TITAN V27 - Configuration Module
"""
import os
from pathlib import Path

# 📁 Paths
BASE_DIR = Path(__file__).parent
DB_FILE = BASE_DIR / "titan_v27_permanent.json"

# 🔑 API Keys - Lấy từ environment variables hoặc hardcoded (chỉ để test)
NVIDIA_API_KEY = os.getenv("NVIDIA_API_KEY", "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")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "AIzaSyD1-XMO6FsA9ZgAf2P6nIiXLPp8moTPMrc")

# 🎲 Luật chơi
LOTTERY_CONFIG = {
    "digit_range": range(10),
    "bet_count": 3,
    "draw_length": 5,
    "positions": ["Chục ngàn", "Ngàn", "Trăm", "Chục", "Đơn vị"]
}

# 📊 Bảng bộ số hay đi cùng
PAIR_RULES = [
    "178", "034", "458", "578", "019", "679", "235", "456", "124", "245", 
    "247", "248", "246", "340", "349", "348", "015", "236", "028", "026", 
    "047", "046", "056", "136", "138", "378"
]

# 🤖 AI Models
AI_MODELS = {
    "nvidia": {
        "base_url": "https://integrate.api.nvidia.com/v1",
        "model": "meta/llama-3.1-70b-instruct",
        "temperature": 0.2
    },
    "gemini": {
        "model": "gemini-1.5-flash",
        "temperature": 0.3
    }
}

# 🎨 UI Theme
THEME = {
    "bg_primary": "#05050a",
    "bg_secondary": "#0d1117",
    "accent": "#76b900",
    "accent_secondary": "#00d4ff",
    "danger": "#ff4444",
    "text_primary": "#ffffff",
    "text_secondary": "#888888"
}