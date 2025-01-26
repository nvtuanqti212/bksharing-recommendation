from pydantic_settings import BaseSettings  # Sử dụng BaseSettings từ pydantic-settings

class Settings(BaseSettings):
    DATABASE_URL: str  # Định nghĩa các biến cấu hình tại đây

    class Config:
        env_file = ".env"  # Đọc các biến môi trường từ file .env

# Tạo một đối tượng `settings` để sử dụng trong ứng dụng
settings = Settings()

