import os
import shutil
from sklearn.model_selection import train_test_split

# Eğitim ve maske dosyalarını al
train_dir = "data/train/train"
train_masks_dir = "data/train_masks/train_masks"

val_dir = "data/val_images/"  # Yeni doğrulama klasörü
val_masks_dir = "data/val_masks/"

# Klasörleri oluştur (Eğer yoksa)
os.makedirs(val_dir, exist_ok=True)
os.makedirs(val_masks_dir, exist_ok=True)

# Eğitim verisini al
all_images = sorted(os.listdir(train_dir))
all_masks = sorted(os.listdir(train_masks_dir))

# %80 eğitim, %20 doğrulama olacak şekilde böl
train_images, val_images, train_masks, val_masks = train_test_split(
    all_images, all_masks, test_size=0.2, random_state=42
)

# 📌 1️⃣ Doğrulama verilerini yeni klasörlere taşı
for img in val_images:
    shutil.move(os.path.join(train_dir, img), os.path.join(val_dir, img))

for mask in val_masks:
    shutil.move(os.path.join(train_masks_dir, mask), os.path.join(val_masks_dir, mask))

print(f"✅ {len(val_images)} görüntü ve {len(val_masks)} maske doğrulama klasörlerine taşındı.")
