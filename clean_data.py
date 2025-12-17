import pandas as pd
import os
import warnings

# Tắt cảnh báo đỏ
warnings.simplefilter(action='ignore', category=FutureWarning)

# --- DANH SÁCH CÁC FILE CẦN XỬ LÝ ---
# Bạn kiểm tra xem tên file trong máy bạn có đúng y hệt như này không nhé
FILES_TO_CLEAN = [
    # (Tên file gốc, Loại dữ liệu)
    ("FBref_Premier-League_2024-2025_Full_Merged.xlsx", "player"),
    ("PL_2024-2025_Keepers_Full.xlsx", "player"), 
    ("PL_2024-2025_Teams_For.xlsx", "team"),
    ("PL_2024-2025_Teams_VS.xlsx", "team")
]

def clean_data(df, data_type="player"):
    """
    Hàm làm sạch dữ liệu chung cho cả Player và Team
    """
    # 1. Xóa cột rác (Link trận đấu, Rank...)
    cols_to_drop = ['Rk', 'Matches']
    # Tìm và xóa các cột bắt đầu bằng "Matches_" (ví dụ Matches_shooting)
    cols_to_drop.extend([c for c in df.columns if str(c).startswith('Matches')])
    
    df = df.drop(columns=[c for c in cols_to_drop if c in df.columns])

    # 2. Xử lý riêng cho Cầu thủ/Thủ môn (Vì đội bóng không có Tuổi/Quốc tịch)
    if data_type == "player":
        # Xử lý Quốc tịch: "eng ENG" -> "ENG"
        if 'Nation' in df.columns:
            df['Nation'] = df['Nation'].astype(str).str.split(' ').str.get(-1).replace('nan', 'Unknown')

        # Xử lý Vị trí: "DF,MF" -> "DF"
        if 'Pos' in df.columns:
            df['Pos'] = df['Pos'].astype(str).str.split(',').str.get(0)

        # Xử lý Tuổi: "24-150" -> 24
        if 'Age' in df.columns:
            df['Age'] = df['Age'].astype(str).str[:2]
            df['Age'] = pd.to_numeric(df['Age'], errors='coerce')

    # 3. Chuẩn hóa tên Đội bóng (Squad)
    # Đôi khi FBref ghi "Arsenal vs ..." hoặc có ký tự lạ
    if 'Squad' in df.columns:
        df['Squad'] = df['Squad'].astype(str).str.strip()

    # 4. Chuyển đổi toàn bộ số liệu về dạng số (Numeric)
    # Xác định các cột không phải số (Thông tin định danh)
    if data_type == "player":
        exclude_cols = ['Player', 'Nation', 'Pos', 'Squad', 'Born', 'Team']
    else: # team
        exclude_cols = ['Squad']
    
    # Lấy danh sách cột cần là số
    numeric_cols = [c for c in df.columns if c not in exclude_cols]

    # Ép kiểu về số, lỗi thì biến thành NaN, sau đó điền 0
    for col in numeric_cols:
        # Xóa dấu phẩy nếu có (ví dụ 1,000 phút -> 1000)
        if df[col].dtype == object:
             df[col] = df[col].astype(str).str.replace(',', '')
             
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

    return df

def main():
    print("🚀 BẮT ĐẦU QUÁ TRÌNH LÀM SẠCH DỮ LIỆU...\n")
    
    for filename, dtype in FILES_TO_CLEAN:
        if os.path.exists(filename):
            print(f"-> Đang xử lý file: {filename} (Loại: {dtype.upper()})...")
            
            # Đọc file
            try:
                df = pd.read_excel(filename)
                
                # Gọi hàm làm sạch
                df_clean = clean_data(df, data_type=dtype)
                
                # Lưu file mới với tên có tiền tố "Cleaned_"
                new_filename = f"Cleaned_{filename}"
                df_clean.to_excel(new_filename, index=False)
                
                print(f"   ✅ Xong! Đã lưu thành: {new_filename}")
                print(f"   📊 Kích thước: {len(df_clean)} dòng, {len(df_clean.columns)} cột.\n")
                
            except Exception as e:
                print(f"   ❌ Lỗi khi đọc file {filename}: {e}\n")
        else:
            print(f"⚠️ Không tìm thấy file gốc: {filename} (Bỏ qua)\n")

    print("🎉 HOÀN TẤT! Kiểm tra các file bắt đầu bằng 'Cleaned_' trong thư mục.")

if __name__ == "__main__":
    main()