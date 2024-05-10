from lang_sam import LangSAM
from PIL import Image
from easy_explain import YOLOv8LRP
from ultralytics import YOLO
import torchvision

model = YOLO('ultralyticsplus/yolov8s')
image = Image.open("/scratch/lg3490/tfv/spatial_frames/frame_0034.png")

desired_size = (512, 640)
transform = torchvision.transforms.Compose([
    torchvision.transforms.Resize(desired_size),
    torchvision.transforms.ToTensor(),
])
image = transform(image)
lrp = YOLOv8LRP(model, power=2, eps=1, device='cpu')
explanation_lrp = lrp.explain(image, cls='person', contrastive=False).cpu()

lrp.plot_explanation(frame=image, explanation = explanation_lrp, contrastive=True, cmap='seismic', title='Explanation for person"')
