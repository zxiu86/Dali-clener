import os
import cv2
# استدعاء الحزمة الشاملة بكلمة واحدة عبر الاختصار السحري الملوكي
import MangaScourX as msx

def main():
    # 1. تحديد مجلدات المدخلات والمخرجات حسب طلبك
    input_dir = "raw_img"
    output_dir = "clean_img"
    
    # التأكد من إنشاء مجلد الحفظ إذا لم يكن موجوداً لمنع الكراش
    os.makedirs(output_dir, exist_ok=True)
    
    print("🚀 جاري تشغيل وحش التبييض والتحرير التلقائي MangaScourX...")
    
    # 2. تهيئة محرك التنظيف الشامل بكامل الأسلحة والترسانة الرياضية
    pipeline = msx.MangaCleanPipeline(
        inpainting_method="patchmatch",  # تشغيل الخوارزمية خماسية الأبعاد الأسطورية
        patch_size=7,                    # الحجم المتوازن والمثالي للبكسلات
        denoise_level=5,                 # تنظيف وتنعيم النويز (التحبب)
        whiten_background=True           # تبييض خلفيات الصفحات تلقائياً
    )
    
    # 3. جلب جميع الصور من مجلد raw_img
    if not os.path.exists(input_dir):
        print(f"⚠️ المجلد '{input_dir}' غير موجود! تأكد من إضافة الصور بداخله.")
        return

    files = [f for f in os.listdir(input_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.webp'))]
    
    if not files:
        print("📥 لا توجد صور جديدة داخل مجلد raw_img للتنظيف.")
        return
        
    print(f"📸 تم العثور على {len(files)} صورة، جاري الطحن والتنظيف بالـ 5D...")

    # 4. الدوران على الصور ومعالجتها واحدة تلو الأخرى
    for file_name in files:
        input_path = os.path.join(input_dir, file_name)
        output_path = os.path.join(output_dir, file_name)
        
        print(f"🧼 جاري تبييض وتنظيف: {file_name} ...")
        
        try:
            # قراءة الصورة بصيغة BGR الافتراضية
            image = cv2.imread(input_path)
            if image is None:
                print(f"❌ فشل في قراءة الصورة: {file_name}")
                continue
                
            # تشغيل معجزة التبييض والـ OCR وسحب المصفوفات الهرمية
            result = pipeline.run(image)
            
            # استخراج الصفحة النهائية النظيفة من القاموس المسترجع
            cleaned_page = result["final_page"]
            
            # حفظ الصورة النظيفة داخل مجلد clean_img عشان الـ Actions يسحبها
            cv2.imwrite(output_path, cleaned_page)
            print(f"✅ تم الحفظ بنجاح في: {output_path}")
            
        except Exception as e:
            print(f"💥 حدث خطأ غير متوقع أثناء معالجة الصورة {file_name}: {str(e)}")

    print("🎉 تم الانتهاء من تبييض وتنظيف جميع الصور بنجاح! السيرفر جاهز للـ Commit.")

if __name__ == "__main__":
    main()
        # المحرك يتوقع ماسك True للـ خراب (النص) و False للـ سليم
        known_mask = mask == 0
        cleaned_img = inpainter.run(img, known_mask)
        
        # حفظ النتيجة النهائية الفخمة
        output_path = os.path.join(CLEAN_DIR, img_name)
        cv2.imwrite(output_path, cleaned_img)
        print(f"✅ تم التطهير المحلي وحفظ: {img_name}")
        
    except Exception as e:
        print(f"⚠️ خطأ غير متوقع أثناء المعالجة: {e}")
        # نظام الطوارئ السريع والتقليدي إذا انضربت أي مصفوفة بالـ Memory
        backup_fill = cv2.inpaint(img, mask, inpaintRadius=5, flags=cv2.INPAINT_NS)
        cv2.imwrite(os.path.join(CLEAN_DIR, img_name), backup_fill)

print("\n🎉 قفلنا السيرفر بالكامل! عوف غباء الـ API وروح شوف الإبداع المحلي الخارق!")
