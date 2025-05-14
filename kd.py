import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, f_regression

# Thiết lập trang
st.set_page_config(page_title="Dự đoán kết quả học tập", layout="wide")

# Tạo từ điển ánh xạ tên biến và tên hiển thị tiếng Việt
feature_names = {
    'gio_hoc_moi_tuan': 'Giờ học mỗi tuần',
    'diem_dau_vao': 'Điểm đầu vào',
    'diem_trung_binh_truoc': 'Điểm trung bình các kỳ trước',
    'tham_gia_ngoai_khoa': 'Tham gia ngoại khóa',
    'gio_giai_tri': 'Giờ giải trí mỗi tuần',
    'lam_them': 'Làm thêm',
    'khoang_cach_den_truong': 'Khoảng cách đến trường (km)',
    'gpa': 'Điểm GPA'
}

# Tên các mô hình bằng tiếng Việt
model_names = {
    "Linear Regression": "Hồi quy tuyến tính",
    "Ridge Regression": "Hồi quy Ridge",
    "Lasso Regression": "Hồi quy Lasso",
    "Random Forest": "Random Forest"
}

# Tiêu đề ứng dụng
st.title("Dự đoán kết quả học tập của sinh viên")
st.write("Ứng dụng này dự đoán điểm GPA của sinh viên dựa trên các yếu tố đầu vào")

# Sidebar cho việc tải lên dữ liệu và lựa chọn mô hình
with st.sidebar:
    st.header("Tùy chọn")
    uploaded_file = st.file_uploader("Tải lên file CSV", type=["csv"])
    model_type = st.selectbox(
        "Chọn loại mô hình",
        ["Linear Regression", "Ridge Regression", "Lasso Regression", "Random Forest"],
        format_func=lambda x: model_names[x]
    )
    test_size = st.slider("Tỷ lệ dữ liệu kiểm tra", 0.1, 0.5, 0.2)
    feature_selection = st.checkbox("Sử dụng lựa chọn tính năng")
    k_features = st.slider("Số lượng tính năng quan trọng nhất", 1, 10, 5) if feature_selection else None

# Tạo dữ liệu mẫu nếu người dùng chưa tải lên
@st.cache_data
def create_sample_data(n_samples=200):
    np.random.seed(42)
    gio_hoc_moi_tuan = np.random.randint(5, 50, n_samples)
    diem_dau_vao = np.random.uniform(5.0, 10.0, n_samples)
    diem_trung_binh_truoc = np.random.uniform(5.0, 10.0, n_samples)
    tham_gia_ngoai_khoa = np.random.randint(0, 2, n_samples)  # 0: Không, 1: Có
    gio_giai_tri = np.random.randint(5, 40, n_samples)
    lam_them = np.random.randint(0, 2, n_samples)  # 0: Không, 1: Có
    khoang_cach_den_truong = np.random.randint(1, 50, n_samples)  # km
    
    gpa = (0.03 * gio_hoc_moi_tuan + 
           0.3 * diem_dau_vao + 
           0.4 * diem_trung_binh_truoc + 
           0.1 * tham_gia_ngoai_khoa - 
           0.01 * gio_giai_tri - 
           0.1 * lam_them - 
           0.005 * khoang_cach_den_truong +
           np.random.normal(0, 0.3, n_samples))
    
    gpa = np.clip(gpa, 0, 10)
    
    return pd.DataFrame({
        'gio_hoc_moi_tuan': gio_hoc_moi_tuan,
        'diem_dau_vao': diem_dau_vao,
        'diem_trung_binh_truoc': diem_trung_binh_truoc,
        'tham_gia_ngoai_khoa': tham_gia_ngoai_khoa,
        'gio_giai_tri': gio_giai_tri,
        'lam_them': lam_them,
        'khoang_cach_den_truong': khoang_cach_den_truong,
        'gpa': gpa
    })

# Đọc dữ liệu
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.success("Đã tải lên dữ liệu thành công!")
else:
    df = create_sample_data()
    st.info("Sử dụng dữ liệu mẫu. Tải lên CSV của bạn để sử dụng dữ liệu riêng.")

# Tạo bản sao của dữ liệu với tên cột tiếng Việt để hiển thị
df_display = df.copy()
df_display.columns = [feature_names[col] for col in df.columns]

# Hiển thị dữ liệu
st.header("Dữ liệu")
st.write(df_display.head())

# Thống kê mô tả
st.header("Thống kê mô tả")
st.write(df_display.describe())

# Phân tích dữ liệu
st.header("Phân tích dữ liệu")
tab1, tab2 = st.tabs(["Ma trận tương quan", "Biểu đồ phân tán"])

with tab1:
    # Ma trận tương quan với tên tiếng Việt
    corr = df.corr()
    corr_display = pd.DataFrame(
        corr.values,
        columns=[feature_names[col] for col in corr.columns],
        index=[feature_names[col] for col in corr.index]
    )
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr_display, annot=True, cmap='coolwarm', ax=ax)
    plt.title('Ma trận tương quan giữa các biến')
    st.pyplot(fig)

with tab2:
    # Biểu đồ phân tán với tên tiếng Việt
    original_cols = list(df.columns[:-1])
    display_cols = [feature_names[col] for col in original_cols]
    
    # Tạo ánh xạ từ tên hiển thị về tên gốc
    reverse_mapping = {feature_names[col]: col for col in df.columns}
    
    # Chọn đặc trưng X
    feature_display = st.selectbox("Chọn đặc trưng X:", display_cols)
    feature_x = reverse_mapping[feature_display]  # Chuyển về tên gốc
    
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.scatterplot(x=feature_x, y='gpa', data=df, ax=ax)
    plt.title(f'Mối quan hệ giữa {feature_display} và Điểm GPA')
    plt.xlabel(feature_display)
    plt.ylabel(feature_names['gpa'])
    st.pyplot(fig)

# Xây dựng mô hình
st.header("Xây dựng và đánh giá mô hình")

# Tách dữ liệu
X = df.drop('gpa', axis=1)
y = df['gpa']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

# Tạo pipeline
steps = []
if feature_selection:
    steps.append(('feature_selection', SelectKBest(f_regression, k=k_features)))
steps.append(('scaler', StandardScaler()))

# Chọn mô hình dựa trên lựa chọn
if model_type == "Linear Regression":
    steps.append(('model', LinearRegression()))
elif model_type == "Ridge Regression":
    steps.append(('model', Ridge(alpha=1.0)))
elif model_type == "Lasso Regression":
    steps.append(('model', Lasso(alpha=0.1)))
else:
    steps.append(('model', RandomForestRegressor(n_estimators=100, random_state=42)))

# Tạo pipeline
pipeline = Pipeline(steps)

# Huấn luyện mô hình
pipeline.fit(X_train, y_train)

# Đánh giá mô hình
y_pred = pipeline.predict(X_test)
r2 = r2_score(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)

# Hiển thị kết quả đánh giá
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Hệ số xác định (R²)", f"{r2:.4f}")
with col2:
    st.metric("Sai số bình phương trung bình (MSE)", f"{mse:.4f}")
with col3:
    st.metric("Sai số tuyệt đối trung bình (MAE)", f"{mae:.4f}")

# Hiển thị kết quả dự đoán
st.subheader("Kết quả dự đoán")
fig, ax = plt.subplots(figsize=(10, 6))
plt.scatter(y_test, y_pred, alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
plt.xlabel('Điểm GPA thực tế')
plt.ylabel('Điểm GPA dự đoán')
plt.title('So sánh điểm GPA thực tế và dự đoán')
st.pyplot(fig)

# Tính toán tầm quan trọng của các đặc trưng
if model_type in ["Linear Regression", "Ridge Regression", "Lasso Regression"]:
    if feature_selection:
        selected_features = X.columns[pipeline.named_steps['feature_selection'].get_support()]
        coefs = pd.DataFrame(
            pipeline.named_steps['model'].coef_,
            index=[feature_names[feat] for feat in selected_features],
            columns=['Hệ số']
        )
    else:
        coefs = pd.DataFrame(
            pipeline.named_steps['model'].coef_,
            index=[feature_names[col] for col in X.columns],
            columns=['Hệ số']
        )
    st.subheader("Hệ số của các đặc trưng")
    st.write(coefs.sort_values('Hệ số', ascending=False))
else:
    if feature_selection:
        selected_features = X.columns[pipeline.named_steps['feature_selection'].get_support()]
        importances = pd.DataFrame(
            pipeline.named_steps['model'].feature_importances_,
            index=[feature_names[feat] for feat in selected_features],
            columns=['Mức độ quan trọng']
        )
    else:
        importances = pd.DataFrame(
            pipeline.named_steps['model'].feature_importances_,
            index=[feature_names[col] for col in X.columns],
            columns=['Mức độ quan trọng']
        )
    st.subheader("Mức độ quan trọng của các đặc trưng")
    st.write(importances.sort_values('Mức độ quan trọng', ascending=False))

# Công cụ dự đoán
st.header("Dự đoán kết quả học tập cho sinh viên mới")
st.write("Nhập thông tin của sinh viên để dự đoán điểm GPA:")

# Tạo form nhập dữ liệu với tên tiếng Việt
col1, col2 = st.columns(2)

with col1:
    gio_hoc = st.slider(feature_names['gio_hoc_moi_tuan'], 0, 60, 30)
    diem_dau_vao = st.slider(feature_names['diem_dau_vao'], 0.0, 10.0, 8.0)
    diem_tb_truoc = st.slider(feature_names['diem_trung_binh_truoc'], 0.0, 10.0, 7.5)
    tham_gia_nk = st.selectbox(feature_names['tham_gia_ngoai_khoa'], ["Không", "Có"])

with col2:
    gio_giai_tri = st.slider(feature_names['gio_giai_tri'], 0, 60, 15)
    lam_them = st.selectbox(feature_names['lam_them'], ["Không", "Có"])
    kc_truong = st.slider(feature_names['khoang_cach_den_truong'], 0, 50, 10)

# Chuyển đổi giá trị
tham_gia_nk_value = 1 if tham_gia_nk == "Có" else 0
lam_them_value = 1 if lam_them == "Có" else 0

# Tạo DataFrame cho sinh viên mới
new_student = pd.DataFrame({
    'gio_hoc_moi_tuan': [gio_hoc],
    'diem_dau_vao': [diem_dau_vao],
    'diem_trung_binh_truoc': [diem_tb_truoc],
    'tham_gia_ngoai_khoa': [tham_gia_nk_value],
    'gio_giai_tri': [gio_giai_tri],
    'lam_them': [lam_them_value],
    'khoang_cach_den_truong': [kc_truong]
})

# Dự đoán GPA cho sinh viên mới
predicted_gpa = pipeline.predict(new_student)[0]

# Hiển thị kết quả dự đoán
st.subheader("Kết quả dự đoán")
st.markdown(f"### Điểm GPA dự đoán: {predicted_gpa:.2f}/10")

# Đánh giá mức GPA bằng tiếng Việt
if predicted_gpa >= 8.5:
    st.success("🌟 Xuất sắc - Sinh viên có khả năng đạt thành tích cao")
elif predicted_gpa >= 7.0:
    st.info("✅ Khá - Sinh viên có kết quả học tập tốt")
elif predicted_gpa >= 5.0:
    st.warning("⚠️ Trung bình - Sinh viên cần cải thiện phương pháp học tập")
else:
    st.error("❌ Yếu - Sinh viên cần nỗ lực nhiều hơn và tìm kiếm hỗ trợ")

# Gợi ý cải thiện
st.subheader("Gợi ý cải thiện")

suggestions = []

# Phân tích các yếu tố ảnh hưởng và đưa ra gợi ý
if model_type in ["Linear Regression", "Ridge Regression", "Lasso Regression"]:
    if feature_selection:
        selected_features = X.columns[pipeline.named_steps['feature_selection'].get_support()]
        feature_importances = dict(zip(selected_features, pipeline.named_steps['model'].coef_))
    else:
        feature_importances = dict(zip(X.columns, pipeline.named_steps['model'].coef_))
    
    # Đưa ra gợi ý dựa trên các hệ số
    sorted_features = sorted(feature_importances.items(), key=lambda x: abs(x[1]), reverse=True)
    
    for feature, coef in sorted_features[:3]:
        if coef > 0:  # Yếu tố tích cực
            if feature == 'gio_hoc_moi_tuan' and gio_hoc < 30:
                suggestions.append(f"Tăng {feature_names[feature].lower()} từ {gio_hoc} lên khoảng 30-35 giờ/tuần có thể cải thiện điểm GPA.")
            elif feature == 'tham_gia_ngoai_khoa' and tham_gia_nk_value == 0:
                suggestions.append(f"Nên {feature_names[feature].lower()} để giúp cải thiện kỹ năng và điểm GPA.")
            elif feature == 'diem_dau_vao' or feature == 'diem_trung_binh_truoc':
                suggestions.append(f"{feature_names[feature]} tốt là yếu tố quan trọng ảnh hưởng tích cực đến điểm GPA.")
        else:  # Yếu tố tiêu cực
            if feature == 'gio_giai_tri' and gio_giai_tri > 20:
                suggestions.append(f"Giảm {feature_names[feature].lower()} từ {gio_giai_tri} xuống dưới 20 giờ/tuần có thể cải thiện điểm GPA.")
            elif feature == 'lam_them' and lam_them_value == 1:
                suggestions.append(f"{feature_names[feature]} có thể ảnh hưởng tiêu cực đến điểm GPA. Cân nhắc giảm thời gian làm thêm nếu có thể.")
            elif feature == 'khoang_cach_den_truong' and kc_truong > 20:
                suggestions.append(f"{feature_names[feature]} dài ({kc_truong} km) có thể ảnh hưởng tiêu cực đến thời gian học tập. Nếu có thể, hãy cân nhắc tìm chỗ ở gần trường hơn.")
else:
    # Đối với Random Forest, sử dụng feature importance
    if feature_selection:
        selected_features = X.columns[pipeline.named_steps['feature_selection'].get_support()]
        feature_importances = dict(zip(selected_features, pipeline.named_steps['model'].feature_importances_))
    else:
        feature_importances = dict(zip(X.columns, pipeline.named_steps['model'].feature_importances_))
    
    sorted_features = sorted(feature_importances.items(), key=lambda x: x[1], reverse=True)
    
    # Đưa ra gợi ý dựa trên tầm quan trọng của đặc trưng
    for feature, importance in sorted_features[:3]:
        if feature == 'gio_hoc_moi_tuan' and gio_hoc < 30:
            suggestions.append(f"{feature_names[feature]} là yếu tố quan trọng. Tăng từ {gio_hoc} lên khoảng 30-35 giờ/tuần có thể cải thiện điểm GPA.")
        elif feature == 'tham_gia_ngoai_khoa' and tham_gia_nk_value == 0:
            suggestions.append(f"{feature_names[feature]} là yếu tố đáng kể. Việc tham gia các hoạt động ngoại khóa có thể giúp cải thiện kỹ năng và điểm GPA.")
        elif feature == 'gio_giai_tri' and gio_giai_tri > 20:
            suggestions.append(f"{feature_names[feature]} là yếu tố quan trọng. Cân nhắc giảm từ {gio_giai_tri} xuống dưới 20 giờ/tuần và dành thời gian cho học tập.")
        elif feature == 'diem_dau_vao' or feature == 'diem_trung_binh_truoc':
            suggestions.append(f"{feature_names[feature]} là yếu tố dự báo quan trọng cho điểm GPA tương lai.")

# Hiển thị các gợi ý
if suggestions:
    for suggestion in suggestions:
        st.write(f"- {suggestion}")
else:
    st.write("- Duy trì thói quen học tập hiện tại để đạt kết quả tốt.")
    st.write("- Cân bằng giữa học tập và hoạt động khác để tránh kiệt sức.")