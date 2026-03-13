from torchvision import transforms

# ──────────────────────────────────────────────
# Directory / split layout
# ──────────────────────────────────────────────
# Food101/
#   images/
#     <class_name>/          ← 101 folders, one per food category
#       <image_id>.jpg       ← JPEG images (750 train + 250 test per class)
#   meta/
#     classes.txt            ← 101 class names (snake_case, one per line)
#     labels.txt             ← 101 human-readable labels (one per line)
#     train.txt              ← 75 750 lines  "class_name/image_id"
#     test.txt               ← 25 250 lines  "class_name/image_id"
#     train.json             ← {class_name: [list of "class/id" paths]}
#     test.json              ← same structure for the test split
# ──────────────────────────────────────────────

train_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.570501, 0.452391, 0.343890], [0.274114, 0.279450, 0.279709])
])

val_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.570501, 0.452391, 0.343890], [0.274114, 0.279450, 0.279709])
])
