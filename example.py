import timm
from torchvision import transforms
import torch
from PIL import Image



# Minimum code example to infer best model
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])

test_transform = transforms.Compose([
                            transforms.Resize((224,224)),
                            transforms.ToTensor(),
                             normalize,]
                             )

model = timm.create_model('convnext_tiny_in22k', pretrained=False, num_classes=1)
model.load_state_dict(torch.load('./20230809_150806/best.pth'))
model.eval()

img = Image.open('./dataset/test_images/fields/1.jpeg')
img = test_transform(img)
pred = model(img.unsqueeze(0))
print("pred ", pred)