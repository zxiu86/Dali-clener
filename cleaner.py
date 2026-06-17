import os
import cv2
# ✅ استخدام الاسم الجديد
import mangascourx as msx

def main():
    input_dir = "raw_img"
    output_dir = "clean_img"
    
    os.makedirs(output_dir, exist_ok=True)
    
    print("🚀 جاري تشغيل وحش التبييض والتحرير التلقائي mangascourx...")
    
    pipeline = msx.MangaCleanPipeline(
        inpainting_method="patchmatch",
        patch_size=7,
        denoise_level=5,
        whiten_background=True
    )
    
    if not os.path.exists(input_dir):
        print(f"⚠️ المجلد '{input_dir}' غير موجود!")
        return

    files = [f for f in os.listdir(input_dir) 
             if f.lower().endswith(('.png', '.jpg', '.jpeg', '.webp'))]
    
    if not files:
        print("📥 لا توجد صور جديدة للتنظيف.")
        return
        
    print(f"📸 تم العثور على {len(files)} صورة...")

    for file_name in files:
        input_path = os.path.join(input_dir, file_name)
        output_path = os.path.join(output_dir, file_name)
        
        print(f"🧼 جاري تبييض: {file_name} ...")
        
        try:
            image = cv2.imread(input_path)
            if image is None:
                print(f"❌ فشل في قراءة: {file_name}")
                continue
                
            result = pipeline.run(image)
            cleaned_page = result["final_page"]
            cv2.imwrite(output_path, cleaned_page)
            print(f"✅ تم الحفظ: {output_path}")
            
        except Exception as e:
            print(f"💥 خطأ في {file_name}: {str(e)}")

    print("🎉 تم الانتهاء!")

if __name__ == "__main__":
    main()
