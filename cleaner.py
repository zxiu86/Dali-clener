import os
import glob
import cv2
import easyocr
import numpy as np
from scipy.sparse import lil_matrix, csr_matrix
from scipy.sparse.linalg import spsolve
from scipy.ndimage import uniform_filter

# --- دالة Guided Filter ---
def guided_filter(guide, src, radius, eps):
    guide = guide.astype(np.float64)
    src = src.astype(np.float64)
    mean_guide = uniform_filter(guide, radius)
    mean_src   = uniform_filter(src,   radius)
    mean_guide_src = uniform_filter(guide * src, radius)
    cov_guide_src = mean_guide_src - mean_guide * mean_src
    mean_guide_sq = uniform_filter(guide * guide, radius)
    var_guide = mean_guide_sq - mean_guide * mean_guide
    a = cov_guide_src / (var_guide + eps)
    b = mean_src - a * mean_guide
    mean_a = uniform_filter(a, radius)
    mean_b = uniform_filter(b, radius)
    q = mean_a * guide + mean_b
    return q.clip(0, 255).astype(np.uint8)

# --- بناء مصفوفة لابلاس سريعة ومحمية ---
def _build_laplacian(interior_mask, boundary_mask, known_values):
    H, W = interior_mask.shape
    idx_map = -np.ones((H, W), dtype=int)
    interior_pixels = interior_mask
    n = np.count_nonzero(interior_pixels)

    if n == 0:
        return None, None

    idx_map[interior_pixels] = np.arange(n)
    A = lil_matrix((n, n), dtype=np.float64)
    b = np.zeros(n, dtype=np.float64)
    neighbors = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    # تسريع جلب الإحداثيات لـ GitHub Actions
    y_indices, x_indices = np.where(interior_pixels)
    for y, x in zip(y_indices, x_indices):
        i = idx_map[y, x]
        A[i, i] = -4.0
        for dy, dx in neighbors:
            ny, nx = y + dy, x + dx
            if 0 <= ny < H and 0 <= nx < W:
                if interior_pixels[ny, nx]:
                    j = idx_map[ny, nx]
                    A[i, j] = 1.0
                elif boundary_mask[ny, nx]:
                    b[i] -= known_values[ny, nx]
    return csr_matrix(A), b

# --- نشر الانحدارات المصحح هندسياً ---
def _diffuse_gradients(gx, gy, mask_dilated):
    H, W = mask_dilated.shape
    kernel = np.ones((3, 3), np.uint8)
    exterior = ~mask_dilated
    
    # تصحيح ثغرة الحواف باستخدام التدرج المورفولوجي الصافي
    boundary_mask = mask_dilated & (cv2.dilate(exterior.astype(np.uint8), kernel) > 0)
    interior = mask_dilated & (~boundary_mask)

    vx = gx.copy()
    vy = gy.copy()

    for comp, arr in zip([vx, vy], [gx, gy]):
        A, b = _build_laplacian(interior, boundary_mask, arr)
        if A is dict or A is None: 
            continue
        sol = spsolve(A, b)
        comp[interior] = sol
    return vx, vy

# --- معالج بواسون الذكي المستقر ---
def poisson_inpaint(image, mask, dilate_iter=2):
    img_float = image.astype(np.float64) / 255.0
    H, W, C = img_float.shape

    if mask.ndim == 3:
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    _, mask_bin = cv2.threshold(mask, 127, 1, cv2.THRESH_BINARY)
    mask_bin = mask_bin.astype(bool)

    kernel = np.ones((3, 3), np.uint8)
    mask_dilated = cv2.dilate(mask_bin.astype(np.uint8), kernel, iterations=dilate_iter).astype(bool)

    # حماية السيرفر: إذا كانت المنطقة الفارغة صفر، نرجع الصورة الأصلية فوراً
    if not np.any(mask_dilated):
        return image

    gx = np.zeros_like(img_float)
    gy = np.zeros_like(img_float)
    for c in range(C):
        gx[..., c] = cv2.Sobel(img_float[..., c], cv2.CV_64F, 1, 0, ksize=3)
        gy[..., c] = cv2.Sobel(img_float[..., c], cv2.CV_64F, 0, 1, ksize=3)

    vx = np.zeros_like(gx)
    vy = np.zeros_like(gy)
    for c in range(C):
        vx_c, vy_c = _diffuse_gradients(gx[..., c], gy[..., c], mask_dilated)
        vx[..., c] = vx_c
        vy[..., c] = vy_c

    div = np.zeros_like(img_float)
    for c in range(C):
        div_vx = cv2.Sobel(vx[..., c], cv2.CV_64F, 1, 0, ksize=3)
        div_vy = cv2.Sobel(vy[..., c], cv2.CV_64F, 0, 1, ksize=3)
        div[..., c] = div_vx + div_vy

    exterior = ~mask_dilated
    boundary_mask = mask_dilated & (cv2.dilate(exterior.astype(np.uint8), kernel) > 0)
    interior = mask_dilated & (~boundary_mask)

    result = img_float.copy()
    for c in range(C):
        A, b = _build_laplacian(interior, boundary_mask, result[..., c])
        if A is None: 
            continue
        rhs = div[..., c][interior] + b
        sol = spsolve(A, rhs)
        result[..., c][interior] = sol

    return (np.clip(result, 0, 1) * 255.0).astype(np.uint8)

# --- الدالة النهائية المدمجة للمنظومة ---
def remove_blind_whitening(image, mask, guided_radius=8, guided_eps=1e-6):
    rough = poisson_inpaint(image, mask, dilate_iter=2)
    guided = guided_filter(guide=image, src=rough, radius=guided_radius, eps=guided_eps)

    if mask.ndim == 3:
        mask_gray = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    else:
        mask_gray = mask
    _, mask_bin = cv2.threshold(mask_gray, 127, 255, cv2.THRESH_BINARY)
    
    alpha = cv2.GaussianBlur(mask_bin.astype(np.float32), (2*guided_radius+1, 2*guided_radius+1), 0) / 255.0
    alpha = alpha[..., np.newaxis]

    final = (guided.astype(np.float32) * alpha + image.astype(np.float32) * (1 - alpha)).astype(np.uint8)
    return final

# --- تشغيل محرك الأتمتة الرئيسي ---
if __name__ == "__main__":
    print("⏳ جاري تشغيل عيون الذكاء الاصطناعي الـ OCR...")
    reader = easyocr.Reader(['en'])

    RAW_DIR = "raw_img"
    CLEAN_DIR = "clean_img"
    os.makedirs(RAW_DIR, exist_ok=True)
    os.makedirs(CLEAN_DIR, exist_ok=True)

    images = glob.glob(os.path.join(RAW_DIR, "*"))

    if not images:
        print("❌ ماكو أي صورة بمجلد raw_img!")
        exit()

    for img_path in images:
        img_name = os.path.basename(img_path)
        print(f"🔬 جاري تطبيق معادلات Poisson التدرجية على: {img_name}...")
        
        img = cv2.imread(img_path)
        if img is None: continue
            
        mask = np.zeros(img.shape[:2], dtype="uint8")
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # كشف مزدوج (عادي ومقلوب للأماكن السوداء) لضمان صفر غباء
        res_norm = reader.readtext(img)
        res_inv = reader.readtext(cv2.bitwise_not(gray))
        
        for (bbox, text, prob) in (res_norm + res_inv):
            p1 = (max(0, int(bbox[0][0])), max(0, int(bbox[0][1])))
            p2 = (min(img.shape[1], int(bbox[2][0])), min(img.shape[0], int(bbox[2][1])))
            cv2.rectangle(mask, p1, p2, 255, -1)
        
        # استدعاء الوحش الرياضي المطور
        cleaned_img = remove_blind_whitening(img, mask)
        
        cv2.imwrite(os.path.join(CLEAN_DIR, img_name), cleaned_img)
        print(f"✅ تم التنظيف الاحترافي بنجاح: {img_name}")

    print("🎉 تم قفل التعديل الرياضي الكامل بنجاح ساحق!")
