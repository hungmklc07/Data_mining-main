import undetected_chromedriver as uc
import pandas as pd
import time
import sys
from functools import reduce
import warnings

# --- CẤU HÌNH ---
SEASON = "2024-2025"  # Mùa giải mới nhất
LEAGUE_ID = "9"       # Ngoại hạng Anh
LEAGUE_NAME = "Premier-League"

# Tắt cảnh báo đỏ và hỗ trợ hiển thị tiếng Việt
warnings.simplefilter(action='ignore', category=FutureWarning)
sys.stdout.reconfigure(encoding='utf-8')

def setup_driver():
    print("Đang khởi động trình duyệt...")
    options = uc.ChromeOptions()
    options.headless = False # Bắt buộc False để qua mặt FBref
    
    # Version Chrome hiện tại của bạn (giữ nguyên 142 như đã sửa trước đó)
    driver = uc.Chrome(options=options, version_main=142)
    
    # Mẹo: Thu nhỏ cửa sổ xuống Taskbar để không che màn hình
    driver.minimize_window()
    return driver

def get_table_from_url(driver, url, table_type="player"):
    print(f"Link: {url}")
    driver.get(url)
    
    # Đợi tải trang (20s cho chắc ăn với bảng Team)
    time.sleep(15) 
    driver.execute_script("window.scrollTo(0, 600);")
    time.sleep(2)
    
    try:
        html = driver.page_source
        
        # Đọc tất cả bảng
        # Nếu là lấy đội bóng -> Tìm bảng có chữ "Squad"
        # Nếu là cầu thủ/thủ môn -> Tìm bảng có chữ "Player"
        match_str = "Squad" if table_type == "team" else "Player"
        dfs = pd.read_html(html, match=match_str)
        
        target_df = None
        max_rows = 0
        
        for table in dfs:
            # Logic lọc bảng
            if table_type == "team":
                # Bảng đội bóng phải có cột Squad nhưng KHÔNG có cột Player
                cols = [str(c).lower() for c in table.columns.values]
                if "squad" in str(cols) and "player" not in str(cols):
                    # Bảng team stats thường có 20 đội (dòng)
                    if len(table) >= 20: 
                        target_df = table
                        break
            else:
                # Logic cũ cho cầu thủ/thủ môn
                if len(table) > 20:
                    if len(table) > max_rows:
                        max_rows = len(table)
                        target_df = table

        if target_df is not None:
            # Xử lý header 2 tầng
            if isinstance(target_df.columns, pd.MultiIndex):
                new_cols = []
                for col in target_df.columns.values:
                    if "Unnamed" in str(col[0]):
                        new_cols.append(str(col[1]))
                    else:
                        new_cols.append(f"{col[0]}_{col[1]}")
                target_df.columns = new_cols
            
            # Xóa cột rác
            if table_type != "team":
                 target_df = target_df[target_df['Player'] != 'Player']
            
            cols_to_drop = ['Rk', 'Matches']
            target_df = target_df.drop(columns=[c for c in cols_to_drop if c in target_df.columns])
            
            print(f"-> ✅ Đã lấy được {len(target_df)} dòng dữ liệu.")
            return target_df
        else:
            print(f"-> ❌ Không tìm thấy bảng {table_type} hợp lệ.")
            return None

    except Exception as e:
        print(f"-> ❌ Lỗi: {e}")
        return None

def main():
    driver = setup_driver()
    
    try:
        # ==========================================
        # PHẦN 1: DỮ LIỆU ĐỘI BÓNG (TEAMS)
        # ==========================================
        print("\n=== 1. LẤY DỮ LIỆU ĐỘI BÓNG (TEAMS) ===")
        url_stats = f"https://fbref.com/en/comps/{LEAGUE_ID}/{SEASON}/{SEASON}-{LEAGUE_NAME}-Stats"
        
        # --- Lấy bảng FOR (Chỉ số của đội nhà) ---
        print("\n[Team For] Đang lấy...")
        # Bảng For thường là bảng đầu tiên tìm thấy
        df_team_for = get_table_from_url(driver, url_stats, table_type="team")
        
        if df_team_for is not None:
            file_for = f"PL_{SEASON}_Teams_For.xlsx"
            df_team_for.to_excel(file_for, index=False)
            print(f"-> Đã lưu: {file_for}")

        # --- Lấy bảng VS (Chỉ số đối đầu/bị thủng lưới) ---
        # Mẹo: FBref lưu bảng VS ngay trong trang stats, nhưng ta cần tìm kỹ hơn
        # Cách đơn giản nhất: Bảng VS là bảng thứ 2 có chữ Squad.
        # Nhưng để chính xác, ta dùng code selenium để tìm bảng có ID cụ thể nếu cần.
        # Ở đây ta dùng cách đơn giản: Load lại HTML và lấy bảng thứ 2.
        
        print("\n[Team VS] Đang lấy...")
        # Do df_team_for đã lấy bảng đầu tiên, ta cần logic lấy bảng thứ 2 (VS)
        # Ta gọi lại hàm get_table nhưng thêm logic xử lý riêng ở đây cho nhanh
        html = driver.page_source
        dfs = pd.read_html(html, match="Squad")
        # Thường bảng 0 là League Table, Bảng 1 là Stats For, Bảng 2 là Stats VS
        # Ta sẽ lọc ra các bảng có 20 dòng
        team_tables = [t for t in dfs if len(t) == 20]
        
        if len(team_tables) >= 2:
            # Bảng stats VS thường là bảng thứ 2 trong danh sách các bảng stats
            # (Sau bảng For)
            df_team_vs = team_tables[1] 
            
            # Xử lý header
            if isinstance(df_team_vs.columns, pd.MultiIndex):
                new_cols = []
                for col in df_team_vs.columns.values:
                    if "Unnamed" in str(col[0]):
                        new_cols.append(str(col[1]))
                    else:
                        new_cols.append(f"{col[0]}_{col[1]}")
                df_team_vs.columns = new_cols
                
            file_vs = f"PL_{SEASON}_Teams_VS.xlsx"
            df_team_vs.to_excel(file_vs, index=False)
            print(f"-> ✅ Đã lưu: {file_vs}")
        else:
            print("-> ❌ Không tìm thấy bảng Team VS.")

        # ==========================================
        # PHẦN 2: DỮ LIỆU THỦ MÔN (KEEPERS)
        # ==========================================
        print(f"\n=== 2. LẤY DỮ LIỆU THỦ MÔN (KEEPERS) ===")
        
        # 1. Thủ môn cơ bản
        url_gk = f"https://fbref.com/en/comps/{LEAGUE_ID}/{SEASON}/keepers/{SEASON}-{LEAGUE_NAME}-Stats"
        print("Đang lấy Keeper Basic...")
        df_gk1 = get_table_from_url(driver, url_gk, table_type="player")
        
        # 2. Thủ môn nâng cao (Advanced)
        url_gk_adv = f"https://fbref.com/en/comps/{LEAGUE_ID}/{SEASON}/keepersadv/{SEASON}-{LEAGUE_NAME}-Stats"
        print("Đang lấy Keeper Advanced...")
        df_gk2 = get_table_from_url(driver, url_gk_adv, table_type="player")
        
        if df_gk1 is not None and df_gk2 is not None:
            print("\nĐang gộp dữ liệu thủ môn...")
            # Đổi tên cột bảng 2 để tránh trùng
            cols_id = ['Player', 'Squad', 'Nation', 'Pos', 'Age', 'Born']
            new_names = {c: f"Adv_{c}" for c in df_gk2.columns if c not in cols_id}
            df_gk2 = df_gk2.rename(columns=new_names)
            
            # Gộp
            df_keeper_final = pd.merge(df_gk1, df_gk2, on=cols_id, how='outer')
            
            file_keeper = f"PL_{SEASON}_Keepers_Full.xlsx"
            df_keeper_final.to_excel(file_keeper, index=False)
            print(f"-> ✅ ĐÃ LƯU FILE THỦ MÔN: {file_keeper}")
        else:
            print("-> ❌ Lỗi khi lấy dữ liệu thủ môn (thiếu 1 trong 2 bảng).")

    finally:
        print("\nHoàn tất. Đang đóng trình duyệt...")
        try:
            driver.quit()
        except:
            pass

if __name__ == "__main__":
    main()