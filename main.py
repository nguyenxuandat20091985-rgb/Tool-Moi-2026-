import os
import google.generativeai as genai
from openai import OpenAI # NVIDIA NIM dùng chuẩn OpenAI API
from moviepy.editor import VideoFileClip, vfx

# --- CẤU HÌNH BIẾN MÔI TRƯỜNG (Lấy từ GitHub Secrets) ---
GEMINI_KEY = os.getenv("GEMINI_API_KEY")
NVIDIA_KEY = os.getenv("NVIDIA_API_KEY")

# Khởi tạo Gemini
genai.configure(api_key=GEMINI_KEY)
gemini_model = genai.GenerativeModel('gemini-1.5-flash')

# Khởi tạo NVIDIA (Dùng Llama 3.1 để tối ưu kịch bản)
nvidia_client = OpenAI(
    base_url="https://integrate.api.nvidia.com/v1",
    api_key=NVIDIA_KEY
)

def create_viral_script(product_name):
    """Dùng Gemini để tạo kịch bản Content Automation"""
    prompt = f"""
    Viết kịch bản TikTok 20 giây bán {product_name}. 
    Cấu trúc: 3 giây đầu gây sốc, 12 giây tính năng, 5 giây kêu gọi mua ở Bio.
    Ngôn ngữ: Tiếng Việt đời thường, lôi cuốn phụ nữ.
    """
    response = gemini_model.generate_content(prompt)
    return response.text

def process_video_reup(input_path, output_path):
    """Xào lại video cũ để lách bản quyền TikTok"""
    try:
        clip = VideoFileClip(input_path)
        
        # 1. Lật gương hình ảnh
        clip = clip.fx(vfx.mirror_x)
        
        # 2. Tăng tốc độ 5% (Bot không nhận ra re-up)
        clip = clip.fx(vfx.speedx, 1.05)
        
        # 3. Chỉnh màu sắc nhẹ để đổi mã MD5
        clip = clip.fx(vfx.colorx, 1.1)
        
        # Xuất video (nén nhẹ để upload nhanh)
        clip.write_videofile(output_path, codec="libx264", audio_codec="aac", fps=24)
        print(f"Thành công: {output_path}")
    except Exception as e:
        print(f"Lỗi xử lý video {input_path}: {e}")

if __name__ == "__main__":
    # Tạo thư mục nếu chưa có
    if not os.path.exists("output"): os.makedirs("output")
    
    # Ví dụ: Xử lý tất cả video trong thư mục input
    input_dir = "input"
    if os.path.exists(input_dir):
        for file in os.listdir(input_dir):
            if file.endswith(".mp4"):
                # 1. Tạo kịch bản mới (để anh lấy ý tưởng làm lồng tiếng/caption)
                script = create_viral_script(file.replace(".mp4", ""))
                print(f"--- Kịch bản cho {file} ---\n{script}\n")
                
                # 2. Transform video
                process_video_reup(os.path.join(input_dir, file), os.path.join("output", "AI_" + file))
