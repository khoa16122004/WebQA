import os
import py7zr

def merge_split_files(split_prefix, num_parts, output_7z):
    """
    Ghép các file .7z.001, .7z.002, ... thành một file .7z duy nhất.
    
    Args:
        split_prefix (str): Đường dẫn đến file .7z.001 (không bao gồm số)
        num_parts (int): Số lượng file (từ .001 đến .xxx)
        output_7z (str): Đường dẫn file đầu ra (.7z)
    """
    with open(output_7z, "wb") as output_file:
        for i in range(1, num_parts + 1):
            part_file = f"{split_prefix}.7z.{i:03d}"
            if os.path.exists(part_file):
                with open(part_file, "rb") as f:
                    output_file.write(f.read())
                print(f"Đã ghép {part_file}")
            else:
                print(f"Lỗi: Không tìm thấy {part_file}")
                return False
    return True

def extract_7z(archive_path, output_dir):

    os.makedirs(output_dir, exist_ok=True)
    with py7zr.SevenZipFile(archive_path, mode='r') as archive:
        archive.extractall(path=output_dir)
    print(f"Giải nén thành công vào: {output_dir}")

split_prefix = "WebQA_imgs_7z_chunks/imgs"
num_parts = 50 
output_7z = "merged_image.7z"
output_dir = "extracted_images"

# if merge_split_files(split_prefix, num_parts, output_7z):
# extract_7z(output_7z, output_dir)
extract_7z("WebQA_imgs_7z_chunks/imgs.7z.001", "test")
