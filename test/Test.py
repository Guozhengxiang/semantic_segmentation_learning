import torch
import torchvision
import torchvision.transforms.functional as F
import time
from PIL import Image
from model import FCN, UNet, SegNet, PSPNet, ENet, LinkNet, ICNet, BiSeNet, DFANet


# model = torch.load('/home/froven/桌面/semantic-segmentation model/model_parameter/BiSeNet.pth').cuda()
# model = torch.load('/home/froven/桌面/semantic-segmentation model/model_parameter/BiSeNet.pth').cuda()
model = BiSeNet(num_class=19).cuda()
model.eval()

# data = torch.rand((1, 3, 1024, 1024)).cuda()
image = Image.open('/home/froven/桌面/newdata/test/image/IMG_2047_img.png')
image = F.crop(image, 200, 500, 512, 512)
transform = torchvision.transforms.Compose(
        [
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])
image = transform(image).float()
data = (image.unsqueeze(0)).cuda()
epoch_start_time = time.time()
predict = model(data)
predict = predict.max(1)[1].squeeze().cpu().data.numpy()
print((time.time() - epoch_start_time))

