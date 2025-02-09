import torch
import torchvision
from datasetLoad import CarvanaDataset
from torch.utils.data import DataLoader

def save_checkpoint(state, filename="my_checkpoint.pth.tar"):
    print("=> Saving checkpoint")
    torch.save(state, filename)

def load_checkpoint(checkpoint, model):
    print("=> Loading checkpoint")
    model.load_state_dict(checkpoint["state_dict"])

def get_loaders(
    train_dir,
    train_maskdir,
    val_dir,
    val_maskdir,
    batch_size,
    train_transform,
    val_transform,
    num_workers=4,
    pin_memory=True,
):
    train_ds = CarvanaDataset(
        image_dir=train_dir,
        mask_dir=train_maskdir,
        transform=train_transform,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=True,
    )

    val_ds = CarvanaDataset(
        image_dir=val_dir,
        mask_dir=val_maskdir,
        transform=val_transform,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=False,
    )

    return train_loader, val_loader

def check_accuracy(loader, model, device="cuda"):
    num_correct = 0
    num_pixels = 0
    dice_score = 0
    model.eval()

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device).unsqueeze(1) / 255.0  # Maskeleri normalize et (0-1 aralığına getir)
            
            preds = torch.sigmoid(model(x))
            preds = (preds > 0.5).float()  # Threshold uygula

            num_correct += (preds == y).sum().item()  # `.item()` ile sayıyı al
            num_pixels += torch.numel(preds)  # Toplam piksel sayısını al

            # Dice Score Hesaplama (Sıfıra bölmeyi önleme)
            intersection = (preds * y).sum()
            union = (preds + y).sum()
            dice_score += (2 * intersection) / (max(1, union))  # Bölmeyi sıfırdan kaçınmak için max(1, union)

    num_pixels = max(1, num_pixels)  # Sıfıra bölmeyi önle
    accuracy = (num_correct / num_pixels * 100)  # Yüzde olarak hesapla
    avg_dice_score = dice_score / len(loader)  # Ortalama Dice Score

    print(f"Got {num_correct}/{num_pixels} with acc {accuracy:.2f}%")
    print(f"Dice score: {avg_dice_score:.4f}")

    model.train()

def save_predictions_as_imgs(loader, model, folder="saved_images/", device="cuda"):
    model.eval()
    for idx, (x, y) in enumerate(loader):
        x = x.to(device=device)
        y = y.to(device=device).unsqueeze(1) / 255.0  # Maskeyi normalize et (0-1 aralığına getir)
        
        with torch.no_grad():
            preds = torch.sigmoid(model(x))
            preds = (preds > 0.5).float()

        # Görselleri kaydet
        torchvision.utils.save_image(preds, f"{folder}/pred_{idx}.png")
        torchvision.utils.save_image(y, f"{folder}/true_{idx}.png")

    print(f"✅ Tahmin edilen maskeler {folder} klasörüne kaydedildi!")
    model.train()
