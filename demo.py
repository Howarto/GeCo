from torch.nn import DataParallel
from models.geco_infer import build_model
from utils.arg_parser import get_argparser
import argparse
import torch
from torchvision import transforms as T
from PIL import Image, ImageDraw
from torchvision import ops
from utils.data import resize_and_pad
import numpy as np
import os


@torch.no_grad()
def demo(args):
    img_path = args.image_path

    # Force zero-shot configuration
    args.zero_shot = True

    print(f"Running zero-shot demo on image: {img_path}")

    gpu = 0
    torch.cuda.set_device(gpu)
    device = torch.device(gpu)

    model = DataParallel(
        build_model(args).to(device),
        device_ids=[gpu],
        output_device=gpu
    )
    model.load_state_dict(
        torch.load('GeCo.pth', weights_only=True)['model'], strict=False,
    )

    model.eval()

    image = T.ToTensor()(Image.open(img_path).convert("RGB"))
    original_height, original_width = image.shape[1], image.shape[2]

    print(f"Original image size: {original_width}x{original_height}")

    # For zero-shot, we don't need bounding boxes - create empty tensor
    bboxes = torch.empty(0, 4, dtype=torch.float32)

    img, bboxes, scale = resize_and_pad(image, bboxes, full_stretch=False)
    img = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(img).unsqueeze(0).to(device)
    bboxes = bboxes.unsqueeze(0).to(device)

    print("Running model inference...")
    outputs, _, _, _, masks = model(img, bboxes)
    del _

    idx = 0
    thr = 4

    if len(outputs[idx]['pred_boxes']) > 0:
        keep = ops.nms(outputs[idx]['pred_boxes'][outputs[idx]['box_v'] > outputs[idx]['box_v'].max() / thr],
                       outputs[idx]['box_v'][outputs[idx]['box_v'] > outputs[idx]['box_v'].max() / thr], 0.5)

        boxes = (outputs[idx]['pred_boxes'][outputs[idx]['box_v'] > outputs[idx]['box_v'].max() / thr])[keep]
        pred_bboxes = torch.clamp(boxes, 0, 1)

        print(f"Detected {len(pred_bboxes)} objects")

        # Convert normalized coordinates to original image coordinates
        pred_boxes_original = pred_bboxes.cpu() / torch.tensor([scale, scale, scale, scale]) * img.shape[-1]

        # Create output image with bounding boxes
        output_image = Image.fromarray((image.permute(1, 2, 0).numpy() * 255).astype(np.uint8))
        draw = ImageDraw.Draw(output_image)

        # Draw bounding boxes
        for i, box in enumerate(pred_boxes_original):
            x1, y1, x2, y2 = box.tolist()
            draw.rectangle([x1, y1, x2, y2], outline='red', width=3)
            draw.text((x1, y1-20), f'Object {i+1}', fill='red')

        # Save output image
        output_path = args.output_path
        output_image.save(output_path)
        print(f"Output image saved to: {output_path}")

        # Save mask if requested
        if args.output_masks and len(masks[idx]) > 0:
            masks_ = masks[idx][(outputs[idx]['box_v'] > outputs[idx]['box_v'].max() / thr)[0]]
            N_masks = masks_.shape[0]
            indices = torch.randint(1, N_masks + 1, (1, N_masks), device=masks_.device).view(-1, 1, 1)
            combined_masks = (masks_ * indices).sum(dim=0)
            mask_display = (
                T.Resize((int(img.shape[2] / scale), int(img.shape[3] / scale)), interpolation=T.InterpolationMode.NEAREST)(
                    combined_masks.cpu().unsqueeze(0))[0])[:image.shape[1], :image.shape[2]]

            # Save mask as image
            mask_path = args.mask_path
            mask_normalized = (mask_display / mask_display.max() * 255).numpy().astype(np.uint8)
            mask_image = Image.fromarray(mask_normalized, mode='L')
            mask_image.save(mask_path)
            print(f"Mask image saved to: {mask_path}")

            del masks
            del masks_
            del outputs

        # Print detection results
        print("\nDetection Results:")
        for i, box in enumerate(pred_boxes_original):
            x1, y1, x2, y2 = box.tolist()
            w, h = x2 - x1, y2 - y1
            print(f"Object {i+1}: x={x1:.1f}, y={y1:.1f}, w={w:.1f}, h={h:.1f}")

    else:
        print("No objects detected")

    print("Demo completed successfully!")

if __name__ == '__main__':
    parser = argparse.ArgumentParser('GeCo Zero-Shot Demo', parents=[get_argparser()])

    # Set default values for zero-shot demo
    parser.set_defaults(
        zero_shot=True,
        num_objects=10,  # Maximum number of objects to detect
        output_masks=False,  # Set to True if you want mask outputs
        image_path='./demo_image.jpg'  # Default image path
    )

    # Add output path arguments
    parser.add_argument('--output_path', default=None, type=str,
                       help='Path to save the output image with detections (default: {image_path}_detected.jpg)')
    parser.add_argument('--mask_path', default=None, type=str,
                       help='Path to save the mask image (default: {image_path}_masks.png)')

    args = parser.parse_args()

    # Ensure we're in zero-shot mode
    args.zero_shot = True

    # Set default output paths based on image_path if not provided
    if args.output_path is None:
        args.output_path = os.path.splitext(args.image_path)[0] + '_detected.jpg'
    if args.mask_path is None:
        args.mask_path = os.path.splitext(args.image_path)[0] + '_masks.png'

    print("GeCo Zero-Shot Object Detection Demo")
    print("=====================================")
    print(f"Zero-shot mode: {args.zero_shot}")
    print(f"Max objects: {args.num_objects}")
    print(f"Output masks: {args.output_masks}")
    print(f"Image path: {args.image_path}")
    print(f"Output path: {args.output_path}")
    print(f"Mask path: {args.mask_path}")
    print()

    # Check if image exists
    if not os.path.exists(args.image_path):
        print(f"Error: Image file '{args.image_path}' not found!")
        print("Please provide a valid image path using --image_path argument")
        exit(1)

    demo(args)
