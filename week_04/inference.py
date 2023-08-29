import torch
from PIL import Image
from torchvision import transforms

data_transform = transforms.Compose([
    transforms.ToTensor(),
])

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = torch.hub.load('ultralytics/yolov5', 'custom', path='./best.pt')  # local model
model.to(device)
model.eval()

# 选择一张图片进行推理和可视化
image_path = "im2.png"
image = Image.open(image_path)


with torch.no_grad():
    results = model(image)

results.print()
results.show()
