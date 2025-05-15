import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

def create_perfect_student_data(n_samples=5000, file_name="du_lieu_sinh_vien_perfect.csv", r2_target=0.98):
    """
    Tạo dữ liệu mẫu cho mô hình dự đoán GPA với R² rất cao (gần 1.0)
    """
    np.random.seed(42)
    
    # Tạo các biến độc lập với phân phối chuẩn và độc lập với nhau
    gio_hoc_moi_tuan = np.random.normal(30, 10, n_samples)
    diem_dau_vao = np.random.normal(7.5, 1.0, n_samples)
    diem_trung_binh_truoc = np.random.normal(7.0, 1.0, n_samples)
    tham_gia_ngoai_khoa = np.random.binomial(1, 0.5, n_samples)
    gio_giai_tri = np.random.normal(15, 5, n_samples)
    lam_them = np.random.binomial(1, 0.3, n_samples)
    khoang_cach_den_truong = np.random.gamma(2, 5, n_samples)
    chat_luong_giac_ngu = np.random.normal(7, 1.5, n_samples)
    so_lan_vang_mat = np.random.poisson(3, n_samples)
    
    # Giới hạn giá trị trong khoảng hợp lý
    gio_hoc_moi_tuan = np.clip(gio_hoc_moi_tuan, 0, 70)
    diem_dau_vao = np.clip(diem_dau_vao, 1, 10)
    diem_trung_binh_truoc = np.clip(diem_trung_binh_truoc, 1, 10)
    gio_giai_tri = np.clip(gio_giai_tri, 0, 40)
    khoang_cach_den_truong = np.clip(khoang_cach_den_truong, 0.5, 50)
    chat_luong_giac_ngu = np.clip(chat_luong_giac_ngu, 0, 10)
    so_lan_vang_mat = np.clip(so_lan_vang_mat, 0, 20)
    
    # Định nghĩa hệ số cố định cho mô hình
    coef = {
        'gio_hoc_moi_tuan': 0.30,
        'diem_dau_vao': 0.25,
        'diem_trung_binh_truoc': 0.35,
        'tham_gia_ngoai_khoa': 0.15,
        'gio_giai_tri': -0.10,
        'lam_them': -0.05,
        'khoang_cach_den_truong': -0.03,
        'chat_luong_giac_ngu': 0.10,
        'so_lan_vang_mat': -0.08,
        'bias': 2.0  # Hệ số chặn
    }
    
    # Chuẩn hóa các biến để đảm bảo chúng có cùng phạm vi ảnh hưởng
    # Quan trọng để mô hình đánh giá đúng về mức độ ảnh hưởng
    gio_hoc_scaled = (gio_hoc_moi_tuan - np.mean(gio_hoc_moi_tuan)) / np.std(gio_hoc_moi_tuan)
    diem_dau_vao_scaled = (diem_dau_vao - np.mean(diem_dau_vao)) / np.std(diem_dau_vao)
    diem_truoc_scaled = (diem_trung_binh_truoc - np.mean(diem_trung_binh_truoc)) / np.std(diem_trung_binh_truoc)
    giai_tri_scaled = (gio_giai_tri - np.mean(gio_giai_tri)) / np.std(gio_giai_tri)
    khoang_cach_scaled = (khoang_cach_den_truong - np.mean(khoang_cach_den_truong)) / np.std(khoang_cach_den_truong)
    giac_ngu_scaled = (chat_luong_giac_ngu - np.mean(chat_luong_giac_ngu)) / np.std(chat_luong_giac_ngu)
    vang_mat_scaled = (so_lan_vang_mat - np.mean(so_lan_vang_mat)) / np.std(so_lan_vang_mat)
    
    # Tính GPA theo công thức tuyến tính hoàn hảo
    gpa_perfect = (coef['bias'] + 
                  coef['gio_hoc_moi_tuan'] * gio_hoc_scaled +
                  coef['diem_dau_vao'] * diem_dau_vao_scaled + 
                  coef['diem_trung_binh_truoc'] * diem_truoc_scaled +
                  coef['tham_gia_ngoai_khoa'] * tham_gia_ngoai_khoa +
                  coef['gio_giai_tri'] * giai_tri_scaled +
                  coef['lam_them'] * lam_them +
                  coef['khoang_cach_den_truong'] * khoang_cach_scaled +
                  coef['chat_luong_giac_ngu'] * giac_ngu_scaled +
                  coef['so_lan_vang_mat'] * vang_mat_scaled)
    
    # Chỉnh GPA để nằm trong khoảng 0-10
    gpa_min, gpa_max = np.min(gpa_perfect), np.max(gpa_perfect)
    gpa_rescaled = (gpa_perfect - gpa_min) / (gpa_max - gpa_min) * 10
    
    # Thêm nhiễu nhỏ có phân phối chuẩn để dữ liệu trông tự nhiên hơn
    # Điều chỉnh phương sai nhiễu để đạt R² mục tiêu
    noise_variance = 0.02  # Bắt đầu với giá trị nhỏ
    max_iterations = 10
    
    for i in range(max_iterations):
        noise = np.random.normal(0, np.sqrt(noise_variance), n_samples)
        gpa = gpa_rescaled + noise
        gpa = np.clip(gpa, 0, 10)  # Đảm bảo GPA nằm trong khoảng 0-10
        
        # Tạo DataFrame và kiểm tra R²
        X = pd.DataFrame({
            'gio_hoc_moi_tuan': gio_hoc_moi_tuan,
            'diem_dau_vao': diem_dau_vao,
            'diem_trung_binh_truoc': diem_trung_binh_truoc,
            'tham_gia_ngoai_khoa': tham_gia_ngoai_khoa,
            'gio_giai_tri': gio_giai_tri,
            'lam_them': lam_them,
            'khoang_cach_den_truong': khoang_cach_den_truong,
            'chat_luong_giac_ngu': chat_luong_giac_ngu,
            'so_lan_vang_mat': so_lan_vang_mat
        })
        
        # Kiểm tra R² trên tập dữ liệu
        X_train, X_test, y_train, y_test = train_test_split(X, gpa, test_size=0.2, random_state=42)
        model = LinearRegression()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        current_r2 = r2_score(y_test, y_pred)
        
        print(f"Vòng lặp {i+1}: R² = {current_r2:.4f}, Phương sai nhiễu = {noise_variance:.6f}")
        
        # Điều chỉnh phương sai nhiễu
        if abs(current_r2 - r2_target) < 0.005:
            break  # Đã đạt được R² mục tiêu
        elif current_r2 > r2_target:
            noise_variance *= 1.5  # Tăng nhiễu lên
        else:
            noise_variance *= 0.7  # Giảm nhiễu xuống
    
    # Tạo DataFrame hoàn chỉnh
    df = pd.DataFrame({
        'gio_hoc_moi_tuan': np.round(gio_hoc_moi_tuan, 1),
        'diem_dau_vao': np.round(diem_dau_vao, 2),
        'diem_trung_binh_truoc': np.round(diem_trung_binh_truoc, 2),
        'tham_gia_ngoai_khoa': tham_gia_ngoai_khoa,
        'gio_giai_tri': np.round(gio_giai_tri, 1),
        'lam_them': lam_them,
        'khoang_cach_den_truong': np.round(khoang_cach_den_truong, 1),
        'chat_luong_giac_ngu': np.round(chat_luong_giac_ngu, 1),
        'so_lan_vang_mat': so_lan_vang_mat,
        'gpa': np.round(gpa, 2)
    })
    
    # Lưu vào file CSV
    df.to_csv(file_name, index=False)
    
    # Đánh giá mô hình và hiển thị thông tin
    model = LinearRegression()
    model.fit(X, gpa)
    y_pred = model.predict(X)
    
    final_r2 = r2_score(gpa, y_pred)
    final_mse = mean_squared_error(gpa, y_pred)
    final_mae = mean_absolute_error(gpa, y_pred)
    
    print(f"\nKết quả dữ liệu sinh viên hoàn hảo:")
    print(f"Số lượng mẫu: {n_samples}")
    print(f"R² = {final_r2:.6f}")
    print(f"MSE = {final_mse:.6f}")
    print(f"MAE = {final_mae:.6f}")
    
    print("\nHệ số thực tế của mô hình:")
    for i, col in enumerate(X.columns):
        print(f"{col}: {model.coef_[i]:.4f}")
    print(f"Hệ số chặn: {model.intercept_:.4f}")
    
    print("\nPhân phối GPA:")
    for i in range(0, 11, 2):
        count = ((df['gpa'] >= i) & (df['gpa'] < i+2)).sum()
        percent = count / n_samples * 100
        print(f"GPA {i}-{i+2}: {count} sinh viên ({percent:.2f}%)")
    
    # Vẽ biểu đồ
    plt.figure(figsize=(15, 5))
    
    # Biểu đồ so sánh thực tế và dự đoán
    plt.subplot(1, 2, 1)
    plt.scatter(gpa, y_pred, alpha=0.5)
    plt.plot([0, 10], [0, 10], 'r--')
    plt.xlabel('GPA Thực tế')
    plt.ylabel('GPA Dự đoán')
    plt.title(f'So sánh GPA thực tế và dự đoán (R²={final_r2:.4f})')
    
    # Biểu đồ phân phối của residuals
    residuals = gpa - y_pred
    plt.subplot(1, 2, 2)
    plt.hist(residuals, bins=30)
    plt.xlabel('Residuals')
    plt.ylabel('Tần suất')
    plt.title('Phân phối của Residuals')
    
    plt.tight_layout()
    plt.savefig('perfect_model_evaluation.png')
    
    print(f"\nĐã tạo thành công file {file_name} với R² = {final_r2:.4f}")
    print("Biểu đồ đánh giá mô hình đã được lưu vào 'perfect_model_evaluation.png'")
    
    return df

# Tạo dữ liệu với 5000 mẫu và mục tiêu R² = 0.98
create_perfect_student_data(5000, "du_lieu_sinh_vien_perfect.csv", r2_target=0.98)