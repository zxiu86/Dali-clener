#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
pixel.py - احترافي لإزالة التبييض الأعمى (de-whitening) من الصور
باستخدام Poisson Image Editing و Sobel operators و Guided Filter.

الخوارزمية:
1. توسيع القناع قليلاً لضمان عدم استخدام بكسل الحافة المبيضّ.
2. حساب الانحدارات (Gx, Gy) لجميع القنوات خارج القناع.
3. نشر (diffuse) الانحدارات إلى داخل منطقة القناع بحل معادلة لابلاس
   لكل مركبة انحدار على حدة، لنحصل على حقل توجيهي (Vx, Vy) داخل المنطقة.
4. حل معادلة بواسون Δf = div(V) مع شروط ديريشليه عند حدود المنطقة الموسعة
   باستخدام القيم الأصلية للصورة خارج القناع.
5. تطبيق Guided Filter لتنعيم الانتقالات وضمان مطابقة التدرج اللوني
   للخلفية الأصلية دون طمس.
"""

import cv2
import numpy as np
from scipy.sparse import lil_matrix, csr_matrix
from scipy.sparse.linalg import spsolve
from scipy.ndimage import uniform_filter

# ------------------------------------------------------------
# دالة Guided Filter (حسب الورقة الأصلية)
# ------------------------------------------------------------
def guided_filter(guide, src, radius, eps):
    """
    تطبيق مرشح الموجه (Guided Filter) على الصورة `src` باستخدام `guide`.
    
    المعاملات:
        guide : صورة التوجيه (HxWxC) – عادة الصورة الأصلية أو الناتج الأولي.
        src   : الصورة المراد تنعيمها (نفس أبعاد guide).
        radius: نصف قطر نواة box filter.
        eps   : معامل التنظيم (يمنع القسمة على صفر).
    العائد:
        صورة مُرشّحة بنفس أبعاد src.
    """
    # التحويل إلى float64 للدقة
    guide = guide.astype(np.float64)
    src = src.astype(np.float64)

    # حساب متوسطات باستخدام box filter (متوسط متحرك)
    mean_guide = uniform_filter(guide, radius)
    mean_src   = uniform_filter(src,   radius)
    mean_guide_src = uniform_filter(guide * src, radius)
    cov_guide_src = mean_guide_src - mean_guide * mean_src

    mean_guide_sq = uniform_filter(guide * guide, radius)
    var_guide = mean_guide_sq - mean_guide * mean_guide

    # معاملا الانحدار المحلي a و b
    a = cov_guide_src / (var_guide + eps)
    b = mean_src - a * mean_guide

    # متوسط a و b
    mean_a = uniform_filter(a, radius)
    mean_b = uniform_filter(b, radius)

    # الناتج
    q = mean_a * guide + mean_b
    return q.clip(0, 255).astype(np.uint8)


# ------------------------------------------------------------
# بناء مصفوفة لابلاس المتناثرة (للحل السريع)
# ------------------------------------------------------------
def _build_laplacian(interior_mask, boundary_mask, known_values):
    """
    إنشاء نظام المعادلات الخطية (مصفوفة A وطرف أيمن b) لحل معادلة بواسون.
    
    interior_mask : bool array (H,W) للبكسلات الداخلية (غير حدودية) المطلوب حلها.
    boundary_mask : bool array (H,W) للبكسلات الحدودية ذات القيم المعلومة.
    known_values  : float array (H,W) تحتوي القيم المعلومة للبكسلات الحدودية.
    
    العائد:
        A (csr_matrix) , b (1D array)
    """
    H, W = interior_mask.shape
    # تخصيص أرقام للبكسلات الداخلية فقط
    idx_map = -np.ones((H, W), dtype=int)
    interior_pixels = interior_mask
    idx_map[interior_pixels] = np.arange(np.count_nonzero(interior_pixels))
    n = np.count_nonzero(interior_pixels)

    A = lil_matrix((n, n), dtype=np.float64)
    b = np.zeros(n, dtype=np.float64)

    # الجيران الأربعة
    neighbors = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    for y, x in zip(*np.where(interior_pixels)):
        i = idx_map[y, x]
        A[i, i] = -4.0  # معامل البكسل المركزي في لابلاس المتقطع
        for dy, dx in neighbors:
            ny, nx = y + dy, x + dx
            if 0 <= ny < H and 0 <= nx < W:
                if interior_pixels[ny, nx]:
                    j = idx_map[ny, nx]
                    A[i, j] = 1.0
                elif boundary_mask[ny, nx]:
                    # بكسل حدودي: قيمة معروفة تؤثر في الطرف الأيمن
                    b[i] -= known_values[ny, nx]
                else:
                    # بكسل خارج المنطقة كلياً (غير مرغوب)، يُعامل كأنه حدودي بقيمة 0
                    # لكن من المفترض ألا يحدث لأن الحدود تحيط بالمنطقة الداخلية.
                    pass
    return csr_matrix(A), b


# ------------------------------------------------------------
# نشر الانحدارات (Gradient Diffusion) داخل القناع
# ------------------------------------------------------------
def _diffuse_gradients(gx, gy, mask_dilated):
    """
    نشر الانحدارات من خارج القناع إلى داخله بحل معادلة لابلاس لكل مركبة.
    
    gx, gy : مصفوفات الانحدار (لكل قناة) بالحجم (H,W).
    mask_dilated : bool array – المنطقة المراد ملؤها (True = داخل).
    
    العائد:
        vx, vy (مصفوفات بحجم (H,W)) الانحدارات بعد النشر.
    """
    H, W = mask_dilated.shape
    # الحدود: البكسلات داخل القناع والمجاورة لخارجه
    boundary = mask_dilated & ~mask_dilated  # False بالكامل، سنعيد حسابها
    # سوف ننشئ قناع الحدود بأنه البكسلات داخل القناع التي لها جار خارج القناع.
    se = np.ones((3, 3), np.uint8)
    # exterior: خارج القناع
    exterior = ~mask_dilated
    # حدود القناع (البكسلات في mask ولها جار خارجي)
    boundary_mask = mask_dilated & (cv2.dilate(exterior.astype(np.uint8), se) > 0)

    interior = mask_dilated & (~boundary_mask)

    vx = gx.copy()
    vy = gy.copy()

    # حل معادلة لابلاس لكل مركبة
    for comp, arr in zip([vx, vy], [gx, gy]):
        A, b = _build_laplacian(interior, boundary_mask, arr)
        # القيم المبدئية للحل (اختيارية)
        x0 = arr[interior]
        sol = spsolve(A, b)
        comp[interior] = sol

    return vx, vy


# ------------------------------------------------------------
# دالة إعادة بناء بواسون الكاملة
# ------------------------------------------------------------
def poisson_inpaint(image, mask, dilate_iter=1, smooth_grad=False):
    """
    إعادة بناء منطقة القناع في الصورة باستخدام Poisson Image Editing.
    
    المعاملات:
        image       : صورة الإدخال (uint8, HxWx3 BGR or RGB).
        mask        : قناع ثنائي (uint8, 0/255) للمنطقة المراد إزالة تبييضها.
        dilate_iter : عدد مرات توسيع القناع (لتفادي حواف التبييض).
        smooth_grad : تطبيق تنعيم طفيف على الانحدارات بعد النشر (اختياري).
    العائد:
        صورة معالجة (uint8).
    """
    img_float = image.astype(np.float64) / 255.0
    H, W, C = img_float.shape

    # تجهيز القناع: جعله منطقي (True = المنطقة المستهدفة)
    if mask.ndim == 3:
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    _, mask_bin = cv2.threshold(mask, 127, 1, cv2.THRESH_BINARY)
    mask_bin = mask_bin.astype(bool)

    # توسيع القناع لضم الهالة البيضاء المحتملة
    kernel = np.ones((3, 3), np.uint8)
    mask_dilated = cv2.dilate(mask_bin.astype(np.uint8), kernel, iterations=dilate_iter).astype(bool)

    # حساب الانحدارات باستخدام Sobel (على كل قناة)
    gx = np.zeros_like(img_float)
    gy = np.zeros_like(img_float)
    for c in range(C):
        gx[..., c] = cv2.Sobel(img_float[..., c], cv2.CV_64F, 1, 0, ksize=3)
        gy[..., c] = cv2.Sobel(img_float[..., c], cv2.CV_64F, 0, 1, ksize=3)

    # نشر الانحدارات إلى داخل المنطقة الموسعة
    vx = np.zeros_like(gx)
    vy = np.zeros_like(gy)
    for c in range(C):
        vx_c, vy_c = _diffuse_gradients(gx[..., c], gy[..., c], mask_dilated)
        vx[..., c] = vx_c
        vy[..., c] = vy_c

    if smooth_grad:
        # تنعيم طفيف للحقل التوجيهي داخل القناع فقط (اختياري)
        vx = cv2.GaussianBlur(vx, (3, 3), 0.5)
        vy = cv2.GaussianBlur(vy, (3, 3), 0.5)

    # حساب التباعد (divergence) للحقل التوجيهي
    div = np.zeros_like(img_float)
    for c in range(C):
        # التباعد = dVx/dx + dVy/dy
        div_vx = cv2.Sobel(vx[..., c], cv2.CV_64F, 1, 0, ksize=3)
        div_vy = cv2.Sobel(vy[..., c], cv2.CV_64F, 0, 1, ksize=3)
        div[..., c] = div_vx + div_vy

    # حدود المنطقة الموسعة
    exterior = ~mask_dilated
    boundary_mask = mask_dilated & (cv2.dilate(exterior.astype(np.uint8), kernel) > 0)
    interior = mask_dilated & (~boundary_mask)

    result = img_float.copy()
    for c in range(C):
        # بناء النظام وحله للقناة c
        A, b = _build_laplacian(interior, boundary_mask, result[..., c])
        # الطرف الأيمن = التباعد
        rhs = div[..., c][interior] + b  # b يحتوي مساهمة الحدود
        sol = spsolve(A, rhs)
        result[..., c][interior] = sol

    # القص والتحويل لـ uint8
    result = np.clip(result, 0, 1) * 255.0
    return result.astype(np.uint8)


# ------------------------------------------------------------
# الدالة الرئيسية لإزالة التبييض الأعمى
# ------------------------------------------------------------
def remove_blind_whitening(image, mask, guided_radius=8, guided_eps=1e-6):
    """
    تزيل التبييض الأعمى من الصورة باستخدام:
      - Poisson Editing لإعادة بناء المنطقة بقيم متطابقة مع الخلفية.
      - Guided Filter لضمان انسيابية التدرج وعدم وجود ضبابية.
      
    المعاملات:
        image         : صورة الإدخال (uint8, BGR).
        mask          : قناع المنطقة المبيضة (uint8, 0/255).
        guided_radius : نصف قطر Guided Filter.
        guided_eps    : معامل epsilon للمرشح.
    العائد:
        الصورة بعد إزالة التبييض (uint8, BGR).
    """
    # الخطوة 1: إعادة بناء بواسون
    rough = poisson_inpaint(image, mask, dilate_iter=1, smooth_grad=False)

    # الخطوة 2: Guided Filter لتحسين التطابق مع الخلفية
    # نستخدم الصورة الأصلية كدليل (guide) والصورة المُعاد بناؤها كمدخل (src)
    # نطبقه فقط على منطقة القناع ثم ندمج
    guided = guided_filter(guide=image, src=rough, radius=guided_radius, eps=guided_eps)

    # قناع للتطبيق التدريجي (ناعم) لتجنب خط فاصل حاد
    # إنشاء قناع ألفا من القناع الأصلي مع تمويه خفيف
    if mask.ndim == 3:
        mask_gray = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    else:
        mask_gray = mask
    _, mask_bin = cv2.threshold(mask_gray, 127, 255, cv2.THRESH_BINARY)
    alpha = cv2.GaussianBlur(mask_bin.astype(np.float32), (2*guided_radius+1, 2*guided_radius+1), 0) / 255.0
    alpha = alpha[..., np.newaxis]  # (H,W,1)

    # دمج الصورة المرشحة مع الصورة الأصلية خارج القناع
    final = (guided.astype(np.float32) * alpha + image.astype(np.float32) * (1 - alpha)).astype(np.uint8)
    return final


# ------------------------------------------------------------
# مثال بسيط للتجربة
# ------------------------------------------------------------
if __name__ == "__main__":
    import sys

    if len(sys.argv) != 3:
        print("استخدام: python pixel.py <صورة_مدخلة> <قناع>")
        sys.exit(1)

    img = cv2.imread(sys.argv[1])
    mask = cv2.imread(sys.argv[2], cv2.IMREAD_GRAYSCALE)
    if img is None or mask is None:
        print("خطأ في تحميل الصور.")
        sys.exit(1)

    result = remove_blind_whitening(img, mask)
    cv2.imwrite("output_dewhitened.png", result)
    print("تم حفظ النتيجة في output_dewhitened.png")
