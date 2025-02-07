import structlog
import json
import pandas as pd
import numpy as np
from app.core.config import settings

logger = structlog.get_logger('collaborative_filtering')

# =============================================================================
# 1. TÍNH BASELINE PREDICTORS VÀ PHẦN DƯ (RESIDUALS) DỰA TRÊN CLICK_COUNT
# =============================================================================
def compute_baseline_and_residuals(clicks_df: pd.DataFrame):
    """
    Tính baseline predictors cho mỗi lượt click theo công thức:
        b_ui = μ + b_u + b_i
    Với:
        μ: trung bình click_count toàn hệ thống
        b_u: bias của student (user)
        b_i: bias của mentor (item)
    
    Sau đó, tính phần dư: residual = click_count - b_ui.
    
    :param clicks_df: DataFrame chứa các cột [student_id, mentor_id, click_count]
    :return: global_avg, user_bias (dict), item_bias (dict), clicks_df (với thêm cột 'baseline' và 'residual')
    """
    # Tính trung bình toàn hệ thống
    global_avg = clicks_df['click_count'].mean()
    
    # Tính bias cho student: trung bình click_count của student trừ global_avg
    user_bias = clicks_df.groupby('student_account_id')['click_count'].mean() - global_avg
    user_bias_dict = user_bias.to_dict()
    
    # Tính bias cho mentor (item)
    item_bias = clicks_df.groupby('mentor_account_id')['click_count'].mean() - global_avg
    item_bias_dict = item_bias.to_dict()
    
    # Hàm tính baseline cho từng dòng click
    def baseline(row):
        return global_avg + user_bias_dict.get(row['student_account_id'], 0) + item_bias_dict.get(row['mentor_account_id'], 0)
    
    clicks_df['baseline'] = clicks_df.apply(baseline, axis=1)
    clicks_df['residual'] = clicks_df['click_count'] - clicks_df['baseline']
    
    logger.info(f"Global average click_count: {global_avg}")
    return global_avg, user_bias_dict, item_bias_dict, clicks_df

# =============================================================================
# 2. TÍNH MA TRẬN SIMILARITY GIỮA CÁC MENTOR (ITEM-ITEM)
# =============================================================================
def compute_shrunk_pearson(item_i, item_j, residual_matrix, lambda_param=100):
    """
    Tính hệ số Pearson giữa hai mentor dựa trên vector phần dư (residual) của chúng,
    chỉ tính trên tập các student đã click cả hai mentor. Sau đó, áp dụng shrinkage:
    
        s_ij = (n_ij / (n_ij + λ)) * ρ̂_ij
    
    :param item_i: Mentor id i.
    :param item_j: Mentor id j.
    :param residual_matrix: Ma trận (pivot) với index là student_id, cột là mentor_id, giá trị là residual.
    :param lambda_param: Tham số shrinkage (mặc định 100).
    :return: Giá trị similarity giữa 2 mentor.
    """
    vec_i = residual_matrix[item_i]
    vec_j = residual_matrix[item_j]
    
    # Lấy tập các student mà cả hai mentor đều có giá trị không null
    common = vec_i.index[vec_i.notnull() & vec_j.notnull()]
    n_common = len(common)
    
    if n_common < 2:
        return 0  # Không đủ dữ liệu chung để tính correlation
    
    xi = vec_i.loc[common]
    xj = vec_j.loc[common]
    
    if xi.std(ddof=0) == 0 or xj.std(ddof=0) == 0:
        raw_corr = 0
    else:
        raw_corr = np.corrcoef(xi, xj)[0, 1]
    
    # Áp dụng shrinkage
    shrunk_corr = (n_common / (n_common + lambda_param)) * raw_corr
    return shrunk_corr

def compute_similarity_matrix(residual_matrix: pd.DataFrame, lambda_param=100, transform=False):
    """
    Tính toán ma trận similarity giữa các mentor dựa trên phần dư, với shrinkage.
    Nếu transform=True, bình phương các giá trị similarity.
    
    :param residual_matrix: Ma trận residual (index: student_id, cột: mentor_id).
    :param lambda_param: Tham số shrinkage.
    :param transform: Nếu True, áp dụng bình phương similarity.
    :return: DataFrame similarity với hàng và cột là mentor_id.
    """
    mentor_ids = residual_matrix.columns
    sim_data = {}
    for i in mentor_ids:
        sim_data[i] = {}
        for j in mentor_ids:
            if i == j:
                sim_data[i][j] = 1.0
            else:
                sim = compute_shrunk_pearson(i, j, residual_matrix, lambda_param)
                if transform:
                    sim = sim ** 2
                sim_data[i][j] = sim
    similarity_df = pd.DataFrame(sim_data)
    return similarity_df

# =============================================================================
# 3. DỰ ĐOÁN CLICK_COUNT CHO MỘT STUDENT VỚI MỘT MENTOR CHƯA ĐƯỢC CLICK
# =============================================================================
def predict_click(user_account_id, mentor_account_id, clicks_df, residual_matrix, similarity_df, global_avg, user_bias, item_bias, k=5):
    """
    Dự đoán số lượt click (click_count) của student (user_account_id) đối với mentor (mentor_id)
    chưa có dữ liệu, theo mô hình item-item collaborative filtering dựa trên phần dư:
    
        ŕ_ui = b_ui + (∑_{j∈S^k(i;u)} s_ij (click_{uj} - b_{uj})) / (∑_{j∈S^k(i;u)} |s_ij|)
    
    :param user_account_id: ID của student.
    :param mentor_id: ID của mentor cần dự đoán.
    :param clicks_df: DataFrame chứa các lượt click với cột [student_id, mentor_id, click_count, baseline, residual].
    :param residual_matrix: Ma trận residual (student x mentor).
    :param similarity_df: Ma trận similarity giữa các mentor.
    :param global_avg: Global average click_count.
    :param user_bias: Dictionary bias của student.
    :param item_bias: Dictionary bias của mentor.
    :param k: Số lượng mentor hàng xóm dùng cho dự đoán.
    :return: Giá trị dự đoán click_count.
    """
    # Baseline cho (user, mentor)
    baseline_ui = global_avg + user_bias.get(user_account_id, 0) + item_bias.get(mentor_account_id, 0)
    
    # Lấy các mentor mà student đã click (trong clicks_df)
    user_clicks = clicks_df[clicks_df['student_account_id'] == user_account_id]
    if user_clicks.empty:
        return baseline_ui  # Nếu student chưa có click nào, trả về baseline
    
    # Tập hàng xóm: các mentor mà student đã click (loại trừ mentor_account_id hiện tại)
    candidate_items = user_clicks['mentor_account_id'].unique()
    neighbor_list = []
    for j in candidate_items:
        if j == mentor_account_id:
            continue
        sim = similarity_df.loc[mentor_account_id, j]
        # Lấy phần dư của student đối với mentor j
        resid = user_clicks[user_clicks['mentor_account_id'] == j]['residual'].values[0]
        neighbor_list.append((j, sim, resid))
    
    if not neighbor_list:
        return baseline_ui
    
    # Sắp xếp theo similarity giảm dần và chọn top k
    neighbor_list.sort(key=lambda x: x[1], reverse=True)
    top_neighbors = neighbor_list[:k]
    
    numerator = sum(sim * resid for (_, sim, resid) in top_neighbors)
    denominator = sum(abs(sim) for (_, sim, resid) in top_neighbors)
    
    predicted_click = baseline_ui + (numerator / denominator if denominator != 0 else 0)
    return predicted_click

# =============================================================================
# 4. HÀM RECOMMENDATION CHÍNH (COLLABORATIVE FILTERING DỰA TRÊN CLICK)
# =============================================================================
def collaborative_filtering(student: pd.DataFrame, clicks_df: pd.DataFrame, mentors: pd.DataFrame,
                              top_k_neighbors: int = 5, top_n: int = 10, lambda_param: int = 100,
                              transform_similarity: bool = True):
    """
    Recommend danh sách mentor cho một student dựa trên item-item collaborative filtering.
    Quy trình:
      1. Tính baseline predictors và phần dư (residual) từ click_count.
      2. Xây dựng ma trận residual (pivot table).
      3. Tính toán ma trận similarity giữa các mentor dựa trên residual (áp dụng shrinkage và tùy chọn biến đổi).
      4. Dự đoán click_count cho từng mentor cho student dựa trên các mentor hàng xóm đã click.
      5. Sắp xếp theo dự đoán click_count và trả về danh sách top_n mentor recommended.
    
    :param student: Một pandas Series chứa thông tin của student (phải có trường 'student_id').
    :param clicks_df: DataFrame chứa các lượt click với các cột [student_id, mentor_account_id, click_count].
    :param mentors: DataFrame chứa thông tin mentor (ít nhất cột 'account_id' và 'account_name').
    :param top_k_neighbors: Số lượng mentor hàng xóm dùng cho dự đoán.
    :param top_n: Số lượng mentor recommended cần trả về.
    :param lambda_param: Tham số shrinkage.
    :param transform_similarity: Nếu True, áp dụng biến đổi (bình phương) similarity.
    :return: DataFrame chứa các mentor được recommended kèm theo dự đoán click_count.
    """

    if student.empty:
        logger.error("Student record is empty.")
        return pd.DataFrame()

    # Bước 1: Tính baseline và phần dư cho bảng click
    global_avg, user_bias, item_bias, clicks_df = compute_baseline_and_residuals(clicks_df)
    
    # Bước 2: Xây dựng ma trận residual (pivot table)
    residual_matrix = clicks_df.pivot_table(index='student_account_id', columns='mentor_account_id', values='residual')
    
    # Bước 3: Tính toán ma trận similarity giữa các mentor
    similarity_df = compute_similarity_matrix(residual_matrix, lambda_param=lambda_param,
                                                transform=transform_similarity)
    logger.info("Computed mentor similarity matrix.")
    
    # Bước 4: Dự đoán click_count cho từng mentor cho student dựa trên collaborative filtering
    student_dict = student[['account_id', 'account_name']].to_dict('records')
    logger.info(f"Recommend mentors for student {student_dict}.")
    # Since there's only one element, get the first element from the list
    first_student = student_dict[0]
    logger.info(f"First student: {first_student}")
    student_account_id = first_student['account_id']

    # Lấy danh sách tất cả mentor từ mentors DataFrame (trường account_id)
    all_mentor_ids = mentors['account_id'].unique()
    predictions = []
    for mentor_account_id in all_mentor_ids:
        pred = predict_click(student_account_id, mentor_account_id, clicks_df, residual_matrix,
                              similarity_df, global_avg, user_bias, item_bias, k=top_k_neighbors)
        predictions.append((mentor_account_id, pred))
    pred_df = pd.DataFrame(predictions, columns=['mentor_account_id', 'predicted_click'])
    pred_df = pred_df.sort_values(by='predicted_click', ascending=False).head(top_n)
    
    # Bước 5: Kết hợp thông tin mentor và trả về danh sách recommended
    recommendations = pd.merge(pred_df, mentors, left_on='mentor_account_id', right_on='account_id', how='left')
    recommendations = recommendations[['account_id', 'account_name', 'predicted_click']]
    
    return recommendations