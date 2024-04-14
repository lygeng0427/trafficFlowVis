from lang_sam import LangSAM
import matplotlib.pyplot as plt

def get_masks(image_pil, text_prompt, langsam_path='sam_vit_h_4b8939.pth', visualize=False):
    model = LangSAM()
    masks, bboxs, _, _ = model.predict(image_pil, text_prompt)
    masks_np = [mask.squeeze().cpu().numpy() for mask in masks]
    if visualize:
        display_image_with_masks(image_pil, masks_np)
    return masks, bboxs

def display_image_with_masks(image, masks):
    num_masks = len(masks)

    fig, axes = plt.subplots(1, num_masks + 1, figsize=(15, 5))
    axes[0].imshow(image)
    axes[0].set_title("Original Image")
    axes[0].axis('off')

    for i, mask_np in enumerate(masks):
        axes[i+1].imshow(mask_np, cmap='gray')
        axes[i+1].set_title(f"Mask {i+1}")
        axes[i+1].axis('off')

    plt.tight_layout()
    plt.show()