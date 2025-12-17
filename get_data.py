import undetected_chromedriver as uc
import pandas as pd
import time
import sys
from functools import reduce

# Cấu hình hiển thị tiếng Việt
sys.stdout.reconfigure(encoding='utf-8')

# Tắt cảnh báo đỏ khó chịu
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

def get_data_by_category(driver, category, season, comp_id, comp_name):
    # Tạo URL
    if category == 'stats':
        url = f"https://fbref.com/en/comps/{comp_id}/{season}/{season}-{comp_name}-Stats"
    else:
        url = f"https://fbref.com/en/comps/{comp_id}/{season}/{category}/{season}-{comp_name}-Stats"
    
    print(f"\n--- Đang lấy dữ liệu: {category.upper()} ---")
    print(f"Link: {url}")
    
    driver.get(url)
    time.sleep(10)
    driver.execute_script("window.scrollTo(0, 600);")
    time.sleep(2)
    
    try:
        html = driver.page_source
        
        # Đọc TẤT CẢ các bảng trong trang (không lọc match="Player" nữa để tránh bỏ sót)
        dfs = pd.read_html(html)
        print(f"   -> Tìm thấy tổng cộng {len(dfs)} bảng.")
        
        target_df = None
        
        # --- LOGIC MỚI: TÌM BẢNG TO NHẤT ---
        # Chúng ta sẽ duyệt qua tất cả các bảng, tìm bảng nào có cột 'Player'
        # VÀ có số dòng nhiều nhất.
        max_rows = 0
        
        for i, table in enumerate(dfs):
            # Chuyển tên cột về chữ thường để kiểm tra
            cols = [str(c).lower() for c in table.columns.values]
            
            # Điều kiện: Phải có chữ "player" trong tên cột
            has_player = any("player" in c for c in cols)
            
            # In ra thông tin để debug (xem nó tìm thấy gì)
            print(f"      - Bảng {i}: {len(table)} dòng. Có cột Player? {'CÓ' if has_player else 'KHÔNG'}")
            
            if has_player:
                # Nếu bảng này to hơn bảng trước đó tìm được -> Chọn bảng này
                if len(table) > max_rows:
                    max_rows = len(table)
                    target_df = table
        
        if target_df is not None:
            print(f"   -> Đã chọn bảng có {len(target_df)} dòng.")
            
            # Xử lý tên cột (Flatten MultiIndex)
            if isinstance(target_df.columns, pd.MultiIndex):
                new_cols = []
                for col in target_df.columns.values:
                    if "Unnamed" in str(col[0]):
                        new_cols.append(str(col[1]))
                    else:
                        new_cols.append(f"{col[0]}_{col[1]}")
                target_df.columns = new_cols
            
            # Chuẩn hóa tên cột Player
            for col in target_df.columns:
                if "player" in str(col).lower():
                    target_df = target_df.rename(columns={col: 'Player'})
                    break
            
            # Lọc rác
            target_df = target_df[target_df['Player'] != 'Player']
            target_df = target_df[target_df['Player'] != 'Player']
            
            cols_to_drop = ['Rk', 'Matches']
            target_df = target_df.drop(columns=[c for c in cols_to_drop if c in target_df.columns])
            
            # Xử lý cột 90s
            if '90s' in target_df.columns:
                target_df = target_df.rename(columns={'90s': f'90s_{category}'})
            
            # --- QUAN TRỌNG: KIỂM TRA CỘT SQUAD ---
            # Nếu thiếu cột Squad, đây chắc chắn là bảng rác (Top Scorer...)
            if 'Squad' not in target_df.columns:
                 print(f"   -> ⚠️ Cảnh báo: Bảng này thiếu cột Squad! Có thể lấy nhầm bảng Top Scorer.")
                 # Nếu bảng Stats bị lỗi, trả về None để code chính bỏ qua nó,
                 # thay vì để nó làm hỏng cả quá trình Merge.
                 return None

            print(f"-> ✅ Đã lấy thành công {len(target_df)} dòng dữ liệu chuẩn.")
            return target_df
        else:
            print(f"-> ❌ Không tìm thấy bảng dữ liệu hợp lệ cho {category}")
            return None

    except Exception as e:
        print(f"-> ❌ Lỗi khi lấy {category}: {e}")
        return None

def main():
    # --- CẤU HÌNH ---
    SEASON = "2024-2025"
    LEAGUE_ID = "9"
    LEAGUE_NAME = "Premier-League"
    
    categories = [
        "stats", "shooting", "passing", "passing_types",
        "gca", "defense", "possession", "misc"
    ]
    
    print("Đang khởi động trình duyệt...")
    options = uc.ChromeOptions()
    options.headless = False
    
    # Lưu ý: Giữ nguyên version_main=142 nếu máy bạn vẫn dùng Chrome 142
    driver = uc.Chrome(options=options, version_main=142)
    
    all_dataframes = []
    
    try:
        for cat in categories:
            df = get_data_by_category(driver, cat, SEASON, LEAGUE_ID, LEAGUE_NAME)
            if df is not None:
                # Đổi tên các cột để tránh trùng lặp (ngoại trừ các cột định danh)
                cols_id = ['Player', 'Squad', 'Nation', 'Pos', 'Age', 'Born']
                
                # Logic đổi tên: Thêm tiền tố category vào trước các chỉ số (ví dụ: shooting_Gls)
                # Nhưng giữ nguyên tên cột định danh
                new_names = {}
                for c in df.columns:
                    if c not in cols_id and not c.startswith(f"90s_{cat}"): # Tránh đổi tên lại cột 90s đã xử lý
                        new_names[c] = f"{cat}_{c}"
                
                df = df.rename(columns=new_names)
                all_dataframes.append(df)
            
            time.sleep(5) # Nghỉ chút cho đỡ bị chặn
            
        print("\n--- ĐANG GỘP DỮ LIỆU... ---")
        if len(all_dataframes) > 0:
            # Hàm gộp thông minh
            final_df = reduce(lambda left, right: pd.merge(
                left, 
                right, 
                on=['Player', 'Squad', 'Nation', 'Pos', 'Age', 'Born'], 
                how='outer'
            ), all_dataframes)
            
            print(f"Tổng hợp thành công: {len(final_df)} cầu thủ.")
            
            # Sắp xếp lại cột: Đưa các cột định danh lên đầu
            cols_first = ['Player', 'Nation', 'Pos', 'Squad', 'Age', 'Born']
            cols_others = [c for c in final_df.columns if c not in cols_first]
            final_df = final_df[cols_first + cols_others]

            file_name = f"FBref_{LEAGUE_NAME}_{SEASON}_Full_Merged.xlsx"
            final_df.to_excel(file_name, index=False)
            print(f"\n✅ ĐÃ LƯU FILE THÀNH CÔNG: {file_name}")
            
        else:
            print("Không lấy được dữ liệu nào cả.")

    finally:
        try:
            driver.quit()
        except:
            pass

if __name__ == "__main__":
    main()