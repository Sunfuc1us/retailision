from torchvision import transforms

def training_transform():
    # Preserve printed labels and package orientation: no horizontal flip.
    return transforms.Compose([transforms.Resize((224,224)),transforms.ColorJitter(.15,.15,.1,.02),transforms.RandomRotation(5),transforms.ToTensor(),transforms.Normalize((.485,.456,.406),(.229,.224,.225))])
