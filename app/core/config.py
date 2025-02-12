from pydantic_settings import BaseSettings  # Sử dụng BaseSettings từ pydantic-settings
from dotenv import load_dotenv

load_dotenv()

class Settings(BaseSettings):
    HOST: str  # Định nghĩa các biến cấu hình tại đây
    PORT: int  # Định nghĩa các biến cấu hình tại đây
    DATABASE_URL: str  # Định nghĩa các biến cấu hình tại đây
    LOG_LEVEL: str   # LOG_LEVEL mặc định là INFO
    LOG_JSON_FORMAT: bool   # LOG_JSON_FORMAT mặc định là False
    TOP_K_RECOMMENDATION: int   # TOP_K mặc định là 5
    PROFILE_EXPERIENCE_WEIGHT: float 
    PROFILE_EDUCATION_WEIGHT: float 
    PROFILE_CERTIFICATION_WEIGHT: float 
    PROFILE_MAJOR_WEIGHT: float


    class Config:
        env_file = ".env"  # Đọc các biến môi trường từ file .env

# Tạo một đối tượng `settings` để sử dụng trong ứng dụng
settings = Settings()
