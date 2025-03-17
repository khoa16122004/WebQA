import os
import gdown
from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive

# Xác thực Google Drive API
gauth = GoogleAuth()
gauth.LocalWebserverAuth()
drive = GoogleDrive(gauth)

# Thay bằng ID thư mục trên Google Drive của bạn
folder_id = "19ApkbD5w0I5sV1IeQ9EofJRyAjKnA7tb"

# Lấy danh sách tất cả file trong thư mục
file_list = drive.ListFile({'q': f"'{folder_id}' in parents and trashed=false"}).GetList()

# Tạo thư mục để lưu file
download_dir = "Dataset"
os.makedirs(download_dir, exist_ok=True)

# Tải từng file bằng gdown
for file in file_list:
    file_id = file['id']
    file_name = file['title']
    save_path = os.path.join(download_dir, file_name)

    print(f"Đang tải: {file_name}...")
    gdown.download(f"https://drive.google.com/uc?id={file_id}", save_path, quiet=False)

print("✅ Tải xuống hoàn tất!")
