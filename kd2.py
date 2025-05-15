import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor, VotingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, f_regression
from scipy import stats

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
    'gpa': 'Điểm GPA',
    # Thêm tên cho các đặc trưng mới
    'hoc_per_giai_tri': 'Tỷ lệ học tập/giải trí',
    'hoc_hieu_qua': 'Thời gian học hiệu quả',
    'diem_nen': 'Điểm nền học thuật',
    'ap_luc': 'Áp lực học tập',
    'can_bang': 'Cân bằng học tập'
}

# Tên các mô hình bằng tiếng Việt
model_names = {
    "Linear Regression": "Hồi quy tuyến tính",
    "Ridge Regression": "Hồi quy Ridge",
    "Lasso Regression": "Hồi quy Lasso",
    "ElasticNet": "Hồi quy ElasticNet",
    "Linear Ensemble": "Ensemble các mô hình tuyến tính"
}
# Tiêu đề ứng dụng
st.title("Dự đoán kết quả học tập của sinh viên (Phiên bản cải tiến)")
st.write("Ứng dụng này dự đoán điểm GPA của sinh viên dựa trên các yếu tố đầu vào")

# Sidebar cho việc tải lên dữ liệu và lựa chọn mô hình
with st.sidebar:
    st.header("Tùy chọn")
    uploaded_file = st.file_uploader("Tải lên file CSV", type=["csv"])
    
    st.subheader("Lựa chọn mô hình")
    model_type = st.selectbox(
        "Chọn loại mô hình",
        ["Linear Regression", "Ridge Regression", "Lasso Regression", "ElasticNet", "Linear Ensemble"],
        format_func=lambda x: model_names[x]
    )
    
    # Tùy chọn đặc biệt cho từng loại mô hình
    if model_type == "Ridge Regression":
        alpha = st.slider("Alpha (regularization strength)", 0.01, 10.0, 1.0, 0.01)
    elif model_type == "Lasso Regression":
        alpha = st.slider("Alpha (regularization strength)", 0.001, 1.0, 0.1, 0.001)
    elif model_type == "ElasticNet":
        alpha = st.slider("Alpha (regularization strength)", 0.001, 1.0, 0.1, 0.001)
        l1_ratio = st.slider("L1 ratio", 0.0, 1.0, 0.5, 0.01)
    
    st.subheader("Xử lý dữ liệu")
    test_size = st.slider("Tỷ lệ dữ liệu kiểm tra", 0.1, 0.5, 0.2)
    remove_outliers_option = st.checkbox("Loại bỏ outliers", value=True)
    outlier_threshold = st.slider("Ngưỡng outlier (Z-score)", 2.0, 4.0, 3.0, 0.1) if remove_outliers_option else 3.0
    
    st.subheader("Feature Engineering")
    feature_selection = st.checkbox("Sử dụng lựa chọn tính năng", value=False)
    k_features = st.slider("Số lượng tính năng quan trọng nhất", 1, 15, 7) if feature_selection else None
    
    poly_degree = st.slider("Bậc đa thức (Polynomial Features)", 1, 3, 1)
    use_interactions = st.checkbox("Sử dụng tính năng tương tác", value=True)
    use_log_transform = st.checkbox("Sử dụng biến đổi Log cho GPA", value=False)
    
    st.subheader("Đánh giá và tìm tham số")
    use_grid_search = st.checkbox("Tìm tham số tối ưu tự động", value=False)
    use_cv = st.checkbox("Sử dụng cross-validation", value=True)
    # Hàm loại bỏ outliers
def remove_outliers(df, columns, threshold=3):
    """Loại bỏ outliers sử dụng phương pháp Z-score"""
    df_clean = df.copy()
    outlier_mask = np.zeros(len(df), dtype=bool)
    
    for col in columns:
        if col in df.columns and df[col].dtype in ['int64', 'float64']:
            z_scores = np.abs((df[col] - df[col].mean()) / df[col].std())
            col_outliers = z_scores > threshold
            outlier_mask = outlier_mask | col_outliers
    
    # Giữ lại các dòng không phải outlier
    df_clean = df_clean[~outlier_mask]
    n_removed = sum(outlier_mask)
    
    return df_clean, n_removed

# Hàm tạo các đặc trưng tương tác
def add_interaction_features(X):
    X_new = X.copy()
    
    # Tỷ lệ học tập/giải trí
    X_new['hoc_per_giai_tri'] = X['gio_hoc_moi_tuan'] / (X['gio_giai_tri'] + 1)
    
    # Thời gian học hiệu quả (học - ảnh hưởng của việc làm thêm)
    X_new['hoc_hieu_qua'] = X['gio_hoc_moi_tuan'] * (1 - 0.3 * X['lam_them'])
    
    # Điểm nền học thuật
    X_new['diem_nen'] = (X['diem_dau_vao'] + X['diem_trung_binh_truoc']) / 2
    
    # Điểm áp lực (khoảng cách x làm thêm)
    X_new['ap_luc'] = X['khoang_cach_den_truong'] * X['lam_them']
    
    # Điểm cân bằng học tập
    X_new['can_bang'] = X['tham_gia_ngoai_khoa'] * (X['gio_hoc_moi_tuan'] / (X['gio_giai_tri'] + 1))
    
    return X_new

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

# Xử lý outliers nếu được chọn
if remove_outliers_option:
    numeric_columns = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    df_clean, n_removed = remove_outliers(df, numeric_columns, outlier_threshold)
    st.info(f"Đã loại bỏ {n_removed} dòng dữ liệu ngoại lai từ {len(df)} dòng.")
    df = df_clean  # Sử dụng dữ liệu đã làm sạch

# Tạo bản sao của dữ liệu với tên cột tiếng Việt để hiển thị
df_display = df.copy()
df_display.columns = [feature_names.get(col, col) for col in df.columns]

# Hiển thị dữ liệu
st.header("Dữ liệu")
st.write(df_display.head())

# Thống kê mô tả
st.header("Thống kê mô tả")
st.write(df_display.describe())
# Phân tích dữ liệu
st.header("Phân tích dữ liệu")
tab1, tab2, tab3 = st.tabs(["Ma trận tương quan", "Biểu đồ phân tán", "Phân phối dữ liệu"])

with tab1:
    # Ma trận tương quan với tên tiếng Việt
    corr = df.corr()
    corr_display = pd.DataFrame(
        corr.values,
        columns=[feature_names.get(col, col) for col in corr.columns],
        index=[feature_names.get(col, col) for col in corr.index]
    )
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr_display, annot=True, cmap='coolwarm', ax=ax)
    plt.title('Ma trận tương quan giữa các biến')
    st.pyplot(fig)

with tab2:
    # Biểu đồ phân tán với tên tiếng Việt
    original_cols = list(df.columns[:-1])
    display_cols = [feature_names.get(col, col) for col in original_cols]
    
    # Tạo ánh xạ từ tên hiển thị về tên gốc
    reverse_mapping = {feature_names.get(col, col): col for col in df.columns}
    
    # Chọn đặc trưng X
    feature_display = st.selectbox("Chọn đặc trưng X:", display_cols)
    feature_x = reverse_mapping[feature_display]  # Chuyển về tên gốc
    
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.scatterplot(x=feature_x, y='gpa', data=df, ax=ax)
    plt.title(f'Mối quan hệ giữa {feature_display} và Điểm GPA')
    plt.xlabel(feature_display)
    plt.ylabel(feature_names['gpa'])
    st.pyplot(fig)

with tab3:
    # Phân phối dữ liệu
    feature_to_plot = st.selectbox("Chọn biến để xem phân phối:", 
                                  df.columns, 
                                  format_func=lambda x: feature_names.get(x, x))
    
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.histplot(df[feature_to_plot], kde=True, ax=ax)
    plt.title(f'Phân phối của {feature_names.get(feature_to_plot, feature_to_plot)}')
    plt.xlabel(feature_names.get(feature_to_plot, feature_to_plot))
    st.pyplot(fig)
    # Xây dựng mô hình
st.header("Xây dựng và đánh giá mô hình")

# Tách dữ liệu
X = df.drop('gpa', axis=1)
y = df['gpa']

# Áp dụng biến đổi log cho biến mục tiêu nếu được chọn
if use_log_transform:
    # Đảm bảo GPA > 0 trước khi áp dụng log
    min_gpa = y.min()
    if min_gpa <= 0:
        y = y + abs(min_gpa) + 0.01  # Tránh log(0)
    
    y_log = np.log(y)
    y = y_log
    st.info("Đã áp dụng biến đổi logarithm cho điểm GPA.")

# Thêm đặc trưng tương tác nếu được chọn
if use_interactions:
    X = add_interaction_features(X)
    # Cập nhật thông tin
    new_features = [col for col in X.columns if col not in df.columns and col != 'gpa']
    st.info(f"Đã thêm {len(new_features)} đặc trưng tương tác mới. Tổng số đặc trưng: {X.shape[1]}")

# Tách dữ liệu huấn luyện và kiểm tra
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

# Tạo pipeline
steps = []

# Thêm Polynomial Features nếu được yêu cầu
if poly_degree > 1:
    steps.append(('poly', PolynomialFeatures(degree=poly_degree, include_bias=False)))
    st.info(f"Sử dụng biến đổi đa thức bậc {poly_degree} để tạo thêm đặc trưng.")

# Thêm feature selection nếu được yêu cầu
if feature_selection:
    steps.append(('feature_selection', SelectKBest(f_regression, k=k_features)))

# Thêm scaling
steps.append(('scaler', StandardScaler()))

# Chọn mô hình dựa trên lựa chọn
if model_type == "Linear Regression":
    steps.append(('model', LinearRegression()))
elif model_type == "Ridge Regression":
    steps.append(('model', Ridge(alpha=alpha)))
elif model_type == "Lasso Regression":
    steps.append(('model', Lasso(alpha=alpha)))
elif model_type == "ElasticNet":
    steps.append(('model', ElasticNet(alpha=alpha, l1_ratio=l1_ratio)))
elif model_type == "Linear Ensemble":
    # Ensemble của các mô hình tuyến tính
    lr = LinearRegression()
    ridge = Ridge(alpha=1.0)
    lasso = Lasso(alpha=0.1)
    elastic = ElasticNet(alpha=0.1, l1_ratio=0.5)
    
    ensemble_model = VotingRegressor([
        ('lr', lr),
        ('ridge', ridge),
        ('lasso', lasso),
        ('elastic', elastic)
    ])
    
    steps.append(('model', ensemble_model))
    st.info("Sử dụng ensemble của 4 mô hình tuyến tính (Linear, Ridge, Lasso, ElasticNet)")

# Tạo pipeline
pipeline = Pipeline(steps)
# Sử dụng GridSearchCV để tìm tham số tối ưu nếu được chọn
if use_grid_search:
    if model_type == "Ridge Regression":
        param_grid = {'model__alpha': [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]}
    elif model_type == "Lasso Regression":
        param_grid = {'model__alpha': [0.0001, 0.001, 0.01, 0.1, 1.0]}
    elif model_type == "ElasticNet":
        param_grid = {
            'model__alpha': [0.001, 0.01, 0.1, 1.0],
            'model__l1_ratio': [0.1, 0.3, 0.5, 0.7, 0.9]
        }
    else:
        param_grid = {}
    
    if param_grid:
        with st.spinner("Đang tìm tham số tối ưu..."):
            grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring='r2')
            grid_search.fit(X_train, y_train)
            pipeline = grid_search.best_estimator_
            st.success(f"Đã tìm thấy tham số tối ưu: {grid_search.best_params_}")
    else:
        pipeline.fit(X_train, y_train)
else:
    # Huấn luyện mô hình
    pipeline.fit(X_train, y_train)

# Đánh giá mô hình
y_pred = pipeline.predict(X_test)

# Chuyển đổi ngược về giá trị thực nếu đã sử dụng log transform
if use_log_transform:
    y_pred_original = np.exp(y_pred)
    y_test_original = np.exp(y_test)
    r2 = r2_score(y_test_original, y_pred_original)
    mse = mean_squared_error(y_test_original, y_pred_original)
    mae = mean_absolute_error(y_test_original, y_pred_original)
else:
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

# Hiển thị kết quả cross-validation nếu được chọn
if use_cv:
    st.subheader("Đánh giá bằng Cross-validation")
    with st.spinner("Đang thực hiện 5-fold cross-validation..."):
        cv_scores = cross_val_score(pipeline, X, y, cv=5, scoring='r2')
        
    st.write(f"R² qua 5-fold cross-validation: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
    
    # Nếu cv_scores.mean() tương đương với r2 từ test set, mô hình ổn định
    if abs(cv_scores.mean() - r2) < 0.1:
        st.success("Mô hình có độ ổn định tốt!")
    else:
        st.warning("Mô hình có thể không ổn định giữa các tập dữ liệu khác nhau.")

# Hiển thị kết quả dự đoán
st.subheader("Kết quả dự đoán")
fig, ax = plt.subplots(figsize=(10, 6))

# Sử dụng dữ liệu gốc nếu đã áp dụng log transform
if use_log_transform:
    plt.scatter(y_test_original, y_pred_original, alpha=0.5)
    plt.plot([y_test_original.min(), y_test_original.max()], 
             [y_test_original.min(), y_test_original.max()], 'r--')
    plt.xlabel('Điểm GPA thực tế')
    plt.ylabel('Điểm GPA dự đoán')
else:
    plt.scatter(y_test, y_pred, alpha=0.5)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
    plt.xlabel('Điểm GPA thực tế')
    plt.ylabel('Điểm GPA dự đoán')

plt.title('So sánh điểm GPA thực tế và dự đoán')
st.pyplot(fig)
# Phân tích dư thừa (residual analysis)
st.subheader("Phân tích dư thừa (Residual Analysis)")

# Sử dụng dữ liệu gốc nếu đã áp dụng log transform
if use_log_transform:
    residuals = y_test_original - y_pred_original
else:
    residuals = y_test - y_pred

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Histogram của residuals
sns.histplot(residuals, kde=True, ax=axes[0])
axes[0].set_title("Phân phối của Residuals")
axes[0].set_xlabel("Residuals")

# Scatter plot của residuals
if use_log_transform:
    sns.scatterplot(x=y_pred_original, y=residuals, ax=axes[1])
else:
    sns.scatterplot(x=y_pred, y=residuals, ax=axes[1])

axes[1].axhline(y=0, color='r', linestyle='-')
axes[1].set_title("Residuals vs Predicted Values")
axes[1].set_xlabel("Predicted Values")
axes[1].set_ylabel("Residuals")

st.pyplot(fig)

# Kiểm tra normalcy của residuals
_, p = stats.shapiro(residuals)
st.write(f"Kiểm tra tính chuẩn của residuals (Shapiro-Wilk test): p-value = {p:.4f}")
if p > 0.05:
    st.success("Residuals có phân phối chuẩn (good)")
else:
    st.warning("Residuals không có phân phối chuẩn (có thể cần biến đổi dữ liệu)")
    # Tính toán và hiển thị hệ số
if model_type in ["Linear Regression", "Ridge Regression", "Lasso Regression", "ElasticNet"]:
    st.subheader("Phân tích chi tiết hệ số hồi quy")
    
    # Lấy feature names sau khi áp dụng các biến đổi
    feature_names_final = X.columns.tolist()
    
    # Nếu sử dụng polynomial features, lấy tên đặc trưng sau biến đổi
    if poly_degree > 1 and 'poly' in pipeline.named_steps:
        feature_names_final = pipeline.named_steps['poly'].get_feature_names_out(X.columns)
    
    # Nếu sử dụng feature selection, chỉ lấy các đặc trưng được chọn
    if feature_selection and 'feature_selection' in pipeline.named_steps:
        selected_indices = pipeline.named_steps['feature_selection'].get_support()
        feature_names_final = np.array(feature_names_final)[selected_indices].tolist()
    
    # Lấy hệ số từ mô hình
    try:
        coefficients = pipeline.named_steps['model'].coef_
        
        # Tạo DataFrame để hiển thị hệ số
        coef_df = pd.DataFrame({
            'Tính năng': [feature_names.get(feat, feat) if feat in feature_names else feat 
                         for feat in feature_names_final],
            'Hệ số': coefficients
        })
        
        # Sắp xếp theo giá trị tuyệt đối của hệ số
        coef_df['Hệ số tuyệt đối'] = np.abs(coef_df['Hệ số'])
        coef_df = coef_df.sort_values('Hệ số tuyệt đối', ascending=False)
        
        # Hiển thị bảng hệ số
        st.write(coef_df)
        
        # Vẽ biểu đồ top 10 tính năng quan trọng nhất
        top_n = min(10, len(coef_df))
        plt.figure(figsize=(10, 6))
        colors = ['green' if x > 0 else 'red' for x in coef_df['Hệ số'][:top_n].values]
        plt.barh(coef_df['Tính năng'][:top_n], coef_df['Hệ số'][:top_n], color=colors)
        plt.xlabel('Hệ số')
        plt.title(f'Top {top_n} tính năng quan trọng nhất')
        plt.tight_layout()
        st.pyplot(plt)
        
        # Hiển thị intercept
        if hasattr(pipeline.named_steps['model'], 'intercept_'):
            if isinstance(pipeline.named_steps['model'].intercept_, (list, np.ndarray)):
                st.write(f"Hệ số chặn (Intercept): {pipeline.named_steps['model'].intercept_[0]:.4f}")
            else:
                st.write(f"Hệ số chặn (Intercept): {pipeline.named_steps['model'].intercept_:.4f}")
            
    except (AttributeError, KeyError) as e:
        st.error(f"Không thể hiển thị hệ số hồi quy: {e}")
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
# Thêm các đặc trưng tương tác nếu đã sử dụng
if use_interactions:
    new_student = add_interaction_features(new_student)

# Dự đoán GPA cho sinh viên mới
predicted_gpa = pipeline.predict(new_student)[0]

# Chuyển đổi dự đoán về giá trị thực nếu đã sử dụng log transform
if use_log_transform:
    predicted_gpa = np.exp(predicted_gpa)

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

# Phân tích các yếu tố ảnh hưởng dựa trên mô hình
if model_type in ["Linear Regression", "Ridge Regression", "Lasso Regression", "ElasticNet"]:
    # Lấy hệ số từ mô hình
    try:
        coefficients = pipeline.named_steps['model'].coef_
        feature_names_final = X.columns.tolist()
        
        # Nếu sử dụng polynomial features, lấy tên đặc trưng sau biến đổi
        if poly_degree > 1 and 'poly' in pipeline.named_steps:
            feature_names_final = pipeline.named_steps['poly'].get_feature_names_out(X.columns)
        
        # Nếu sử dụng feature selection, chỉ lấy các đặc trưng được chọn
        if feature_selection and 'feature_selection' in pipeline.named_steps:
            selected_indices = pipeline.named_steps['feature_selection'].get_support()
            feature_names_final = np.array(feature_names_final)[selected_indices].tolist()
        
        # Tạo từ điển ánh xạ đặc trưng và hệ số
        feature_importances = dict(zip(feature_names_final, coefficients))
        
        # Chỉ xem xét các đặc trưng cơ bản cho gợi ý
        basic_features = ['gio_hoc_moi_tuan', 'diem_dau_vao', 'diem_trung_binh_truoc', 
                         'tham_gia_ngoai_khoa', 'gio_giai_tri', 'lam_them', 'khoang_cach_den_truong']
        
        # Lọc hệ số cho các đặc trưng cơ bản
        basic_importances = {}
        for feature in basic_features:
            for key, value in feature_importances.items():
                if feature in key:  # Xử lý cả trường hợp đa thức
                    if feature not in basic_importances:
                        basic_importances[feature] = value
                    else:
                        basic_importances[feature] += value  # Cộng dồn hệ số nếu đã có
        
        # Sắp xếp theo mức độ quan trọng
        sorted_features = sorted(basic_importances.items(), key=lambda x: abs(x[1]), reverse=True)
        
        # Đưa ra gợi ý dựa trên hệ số
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
    except (AttributeError, KeyError) as e:
        st.error(f"Không thể phân tích hệ số cho gợi ý: {e}")
        # Sử dụng gợi ý chung nếu không thể phân tích cụ thể
        suggestions = [
            "Cân bằng giữa thời gian học tập và giải trí",
            "Tham gia các hoạt động ngoại khóa phù hợp để phát triển kỹ năng mềm",
            "Tìm môi trường học tập phù hợp để tăng hiệu quả"
        ]
else:
    # Gợi ý chung cho các mô hình ensemble
    suggestions = [
        "Tăng thời gian học tập hiệu quả, tập trung vào chất lượng hơn số lượng",
        "Cân bằng giữa học tập và các hoạt động khác",
        "Tìm kiếm sự hỗ trợ từ giảng viên và bạn học khi gặp khó khăn"
    ]

# Hiển thị các gợi ý
if suggestions:
    for suggestion in suggestions:
        st.write(f"- {suggestion}")
else:
    st.write("- Duy trì thói quen học tập hiện tại để đạt kết quả tốt.")
    st.write("- Cân bằng giữa học tập và hoạt động khác để tránh kiệt sức.")
    # Thêm phần so sánh với dự đoán mô hình cơ bản
st.subheader("So sánh với mô hình cơ bản")

# Tạo một mô hình cơ bản để so sánh
basic_model = Pipeline([
    ('scaler', StandardScaler()),
    ('model', LinearRegression())
])

# Huấn luyện mô hình cơ bản trên dữ liệu gốc
X_basic = df.drop('gpa', axis=1)
y_basic = df['gpa']
basic_model.fit(X_basic, y_basic)

# Dự đoán với mô hình cơ bản
basic_prediction = basic_model.predict(new_student[X_basic.columns])[0]

# Hiển thị so sánh
col1, col2 = st.columns(2)
with col1:
    st.metric("Dự đoán từ mô hình cải tiến", f"{predicted_gpa:.2f}")
with col2:
    st.metric("Dự đoán từ mô hình cơ bản", f"{basic_prediction:.2f}", 
              delta=f"{predicted_gpa - basic_prediction:.2f}")

# Thêm thông tin giải thích về sự khác biệt
if abs(predicted_gpa - basic_prediction) > 0.5:
    st.info("""
    **Giải thích sự khác biệt:** Mô hình cải tiến xem xét các tương tác phức tạp giữa các yếu tố 
    và áp dụng các kỹ thuật nâng cao nên có thể cho kết quả dự đoán khác biệt so với mô hình cơ bản.
    """)

# Thêm phần giới thiệu về mô hình
st.header("Thông tin về mô hình")
st.write("""
Ứng dụng này sử dụng các kỹ thuật học máy để dự đoán kết quả học tập (GPA) của sinh viên 
dựa trên các yếu tố đầu vào. Mô hình được cải tiến với:

1. **Feature Engineering nâng cao**: Tạo đặc trưng tương tác, biến đổi đa thức, lựa chọn đặc trưng
2. **Xử lý dữ liệu tốt hơn**: Loại bỏ outliers, biến đổi logarithm cho biến mục tiêu nếu cần
3. **Tối ưu hóa siêu tham số**: Tìm giá trị tốt nhất cho các tham số của mô hình
4. **Đánh giá toàn diện**: Cross-validation, phân tích residual, so sánh mô hình

Các cải tiến này giúp tăng độ chính xác của mô hình, đặc biệt khi làm việc với dữ liệu phức tạp.
""")

# Thêm phần giới thiệu về dự án
st.sidebar.markdown("---")
st.sidebar.subheader("Thông tin dự án")
st.sidebar.info("""
**Dự án Dự đoán kết quả học tập của sinh viên**

Phiên bản: 2.0 (Cải tiến)
""")
