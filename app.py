import streamlit as st
import re
from collections import Counter, defaultdict
from itertools import combinations
from openai import OpenAI
import google.generativeai as genai
import json
import math

# API Keys
NVIDIA_API_KEY = "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"
GEMINI_API_KEY = "AIzaSyD1-XMO6FsA9ZgAf2P6nIiXLPp8moTPMrc"

# Tuổi Sửu - Mệnh Kim: Lucky numbers
LUCKY_OX = [0, 2, 5, 6, 7, 8]
METAL_NUMS = [1, 6]
EARTH_NUMS = [2, 5, 8]

# Tách mã nguồn thành các module riêng biệt
def get_nums(text):
    return [n for n in re.findall(r"\d{5}", text) if n]

def analyze_position_bias(db, positions=5):
    # ...
    return bias_result

def detect_repeating_patterns(db):
    # ...
    return patterns

def detect_cold_hot(db, window=30):
    # ...
    return hot_info, cold_info

def calculate_zodiac_boost(digit, zodiac="ox"):
    # ...
    return boost

def predict_with_bias_detection(db):
    # ...
    return result

# Tạo giao diện người dùng
st.set_page_config(page_title="TITAN V27 AI", page_icon="", layout="centered")
st.markdown("""
<style>
    .main {padding: 0.5rem;}
    .big-num {font-size: 42px; font-weight: bold; color: #FFD700; text-align: center; font-family: monospace; letter-spacing: 6px; margin: 5px 0;}
    .box {background: linear-gradient(135deg, #2F4F4F, #1C3A3A); color: #FFD700; padding: 10px; border-radius: 10px; text-align: center; margin: 5px 0; border: 2px solid #FFD700;}
    .grid {display: grid; grid-template-columns: repeat(3, 1fr); gap: 5px; margin: 10px 0;}
    .item {background: linear-gradient(135deg, #FFD700, #FFA500); color: #000; padding: 12px; border-radius: 8px; text-align: center; font-family: monospace; font-size: 28px; font-weight: bold;}
    .item-3 {background: linear-gradient(135deg, #C0C0C0, #808080); color: #000;}
    button {width: 100%; background: linear-gradient(135deg, #FFD700, #FFA500); color: #000; font-size: 18px; font-weight: bold; padding: 12px; border: none; border-radius: 8px;}
    textarea {height: 80px; font-size: 14px;}
    .metric {display: flex; justify-content: space-around; margin: 5px 0; background: rgba(255,215,0,0.1); padding: 8px; border-radius: 8px;}
    .metric-val {font-size: 20px; font-weight: bold; color: #FFD700;}
    .metric-lbl {font-size: 11px; color: #C
