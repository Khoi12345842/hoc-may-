import pandas as pd
import numpy as np
import os

def create_student_data_diverse(n_samples=5000, file_name="du_lieu_sinh_vien_diverse.csv"):
    """
    Tạo dữ liệu mẫu đa dạng cho việc dự đoán kết quả học tập của sinh viên
    
    Parameters:
    -----------
    n_samples : int
        Số lượng mẫu cần tạo
    file_name : str
        Tên file CSV đầu ra
    """
    np.random.seed(42)  # Để kết quả có thể tái tạo
    
    # Tạo các biến đầu vào với phân phối đa dạng hơn
    
    # 1. Giờ học mỗi tuần - tạo phân phối hỗn hợp (mixture distribution)
    gio_hoc_moi_tuan_type = np.random.choice(['ít', 'trung bình', 'nhiều'], size=n_samples, 
                                             p=[0.3, 0.4, 0.3])
    
    gio_hoc_moi_tuan = np.zeros(n_samples)
    for i, type in enumerate(gio_hoc_moi_tuan_type):
        if type == 'ít':
            gio_hoc_moi_tuan[i] = np.random.normal(10, 5, 1)[0]
        elif type == 'trung bình':
            gio_hoc_moi_tuan[i] = np.random.normal(25, 7, 1)[0]
        else:  # nhiều
            gio_hoc_moi_tuan[i] = np.random.normal(45, 10, 1)[0]
    
    gio_hoc_moi_tuan = np.clip(gio_hoc_moi_tuan, 0, 80).astype(int)  # Mở rộng phạm vi
    
    # 2. Điểm đầu vào - phân phối hình chữ U (sinh viên thường có điểm rất tốt hoặc rất kém)
    u_mix = np.random.beta(0.5, 0.5, n_samples)  # Phân phối hình chữ U
    diem_dau_vao = u_mix * 10  # Trải từ 0-10
    diem_dau_vao = np.clip(diem_dau_vao, 1.0, 10.0)  # Giới hạn tối thiểu 1.0
    
    # 3. Điểm trung bình trước - tương quan vừa phải với điểm đầu vào và thêm nhiễu
    diem_trung_binh_truoc = (0.6 * diem_dau_vao + 
                            0.4 * np.random.normal(5, 2.5, n_samples))
    diem_trung_binh_truoc = np.clip(diem_trung_binh_truoc, 1.0, 10.0)
    
    # 4. Tham gia ngoại khóa - phân bố theo điểm trung bình trước (sinh viên giỏi thường tham gia nhiều hơn)
    prob_ngoai_khoa = 0.3 + 0.5 * (diem_trung_binh_truoc / 10)  # Từ 0.3-0.8 theo điểm TB
    tham_gia_ngoai_khoa = np.random.binomial(1, prob_ngoai_khoa, n_samples)
    
    # 5. Giờ giải trí - phân phối hỗn hợp
    giai_tri_group = np.random.choice(['ít', 'trung bình', 'nhiều'], size=n_samples,
                                       p=[0.25, 0.5, 0.25])
    
    gio_giai_tri = np.zeros(n_samples)
    for i, group in enumerate(giai_tri_group):
        if group == 'ít':
            gio_giai_tri[i] = np.random.normal(5, 3, 1)[0]
        elif group == 'trung bình':
            gio_giai_tri[i] = np.random.normal(20, 5, 1)[0]
        else:  # nhiều
            gio_giai_tri[i] = np.random.normal(40, 8, 1)[0]
    
    gio_giai_tri = np.clip(gio_giai_tri, 0, 70).astype(int)  # Mở rộng phạm vi
    
    # 6. Làm thêm - xác suất tăng theo khoảng cách đến trường (chi phí sinh hoạt)
    lam_them = np.random.binomial(1, 0.4, n_samples)
    
    # 7. Khoảng cách đến trường - phân phối co đuôi dài hơn
    khoang_cach_den_truong = np.random.gamma(2, 8, n_samples).astype(int)
    khoang_cach_den_truong = np.clip(khoang_cach_den_truong, 1, 80)  # Mở rộng phạm vi
    
    # 8. Thêm biến mới: Chất lượng giấc ngủ (0-10)
    chat_luong_giac_ngu = np.random.gamma(5, 1, n_samples)
    chat_luong_giac_ngu = np.clip(chat_luong_giac_ngu, 0, 10)
    
    # 9. Thêm biến mới: Số lần vắng mặt
    so_lan_vang_mat = np.random.poisson(5, n_samples)
    so_lan_vang_mat = np.clip(so_lan_vang_mat, 0, 30)
    
    # Tính GPA với công thức phức tạp hơn và nhiễu lớn hơn
    # Tăng hệ số của các biến và thêm nhiễu lớn hơn để tạo ra phạm vi GPA rộng hơn
    gpa_base = (0.04 * gio_hoc_moi_tuan + 
               0.20 * diem_dau_vao + 
               0.25 * diem_trung_binh_truoc + 
               0.15 * tham_gia_ngoai_khoa - 
               0.015 * gio_giai_tri - 
               0.12 * lam_them - 
               0.01 * khoang_cach_den_truong +
               0.08 * chat_luong_giac_ngu -
               0.05 * (so_lan_vang_mat/10))
    
    # Thêm nhiễu lớn theo phân phối t-student (để có nhiều đuôi dài)
    degrees_of_freedom = 3
    noise = np.random.standard_t(df=degrees_of_freedom, size=n_samples) * 0.8
    
    # Tạo GPA với nhiều khoảng giá trị
    gpa = gpa_base + noise
    
    # Tạo một số lượng nhỏ sinh viên xuất sắc và rất kém
    exceptional_high = np.random.choice([True, False], size=n_samples, p=[0.05, 0.95])
    exceptional_low = np.random.choice([True, False], size=n_samples, p=[0.05, 0.95])
    
    gpa[exceptional_high] += np.random.uniform(1.0, 2.0, size=sum(exceptional_high))
    gpa[exceptional_low] -= np.random.uniform(1.0, 2.0, size=sum(exceptional_low))
    
    # Đảm bảo GPA nằm trong khoảng hợp lý 0-10, nhưng đa dạng hơn
    gpa = np.clip(gpa, 0.0, 10.0)
    
    # Tạo DataFrame với cả biến mới
    df = pd.DataFrame({
        'gio_hoc_moi_tuan': gio_hoc_moi_tuan,
        'diem_dau_vao': np.round(diem_dau_vao, 2),
        'diem_trung_binh_truoc': np.round(diem_trung_binh_truoc, 2),
        'tham_gia_ngoai_khoa': tham_gia_ngoai_khoa,
        'gio_giai_tri': gio_giai_tri,
        'lam_them': lam_them,
        'khoang_cach_den_truong': khoang_cach_den_truong,
        'chat_luong_giac_ngu': np.round(chat_luong_giac_ngu, 2),
        'so_lan_vang_mat': so_lan_vang_mat,
        'gpa': np.round(gpa, 2)
    })
    
    # Lưu DataFrame vào file CSV
    df.to_csv(file_name, index=False)
    
    print(f"Đã tạo file {file_name} với {n_samples} mẫu dữ liệu đa dạng")
    print(f"5 mẫu đầu tiên:")
    print(df.head())
    
    # Hiển thị thống kê mô tả
    print("\nThống kê mô tả:")
    print(df.describe())
    
    # Phân tích phân phối GPA
    gpa_ranges = [(0, 2), (2, 4), (4, 6), (6, 8), (8, 10)]
    print("\nPhân phối GPA:")
    for low, high in gpa_ranges:
        count = ((df['gpa'] >= low) & (df['gpa'] < high)).sum()
        percent = count / n_samples * 100
        print(f"GPA {low}-{high}: {count} sinh viên ({percent:.2f}%)")
    
    return df

# Tạo dữ liệu với 5000 mẫu đa dạng
create_student_data_diverse(500, "du_lieu_sinh_vien_diverse.csv")

print("\nBạn có thể sử dụng file 'du_lieu_sinh_vien_diverse.csv' để tải lên ứng dụng dự đoán kết quả học tập.")