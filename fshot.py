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


def parse_bbox_input(bbox_str):
    """Parse bounding box from string format 'x1,y1,x2,y2'"""
    coords = [float(x) for x in bbox_str.split(',')]
    if len(coords) != 4:
        raise ValueError("Bounding box must have 4 coordinates: x1,y1,x2,y2")
    return coords


@torch.no_grad()
def demo_fewshot(args):
    img_path = args.image_path
    exemplar_bboxes = args.exemplar_bboxes

    print(f"Running few-shot demo on image: {img_path}")
    print(f"Exemplar bboxes (regions): {exemplar_bboxes}")

    gpu = 0
    torch.cuda.set_device(gpu)
    device = torch.device(gpu)

    # Build and load model
    args.zero_shot = False  # Few-shot mode, not zero-shot
    args.num_objects = len(exemplar_bboxes)  # Set number of objects based on exemplars

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

    # Parse exemplar bounding boxes and normalize to [0, 1]
    exemplar_boxes_list = []
    print(f"\nExemplar regions:")
    for i, bbox_str in enumerate(exemplar_bboxes):
        x1, y1, x2, y2 = parse_bbox_input(bbox_str)
        # Normalize to [0, 1]
        norm_box = [x1 / original_width, y1 / original_height,
                    x2 / original_width, y2 / original_height]
        exemplar_boxes_list.append(norm_box)
        print(f"  Exemplar {i+1}: x1={x1:.1f}, y1={y1:.1f}, x2={x2:.1f}, y2={y2:.1f}")

    bboxes = torch.tensor(exemplar_boxes_list, dtype=torch.float32)

    img, bboxes, scale = resize_and_pad(image, bboxes, full_stretch=False)
    img = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(img).unsqueeze(0).to(device)
    bboxes = bboxes.unsqueeze(0).to(device)

    print("\nRunning model inference...")
    outputs, _, _, _, masks = model(img, bboxes)
    del _

    idx = 0
    thr = 4

    if len(outputs[idx]['pred_boxes']) > 0:
        keep = ops.nms(outputs[idx]['pred_boxes'][outputs[idx]['box_v'] > outputs[idx]['box_v'].max() / thr],
                       outputs[idx]['box_v'][outputs[idx]['box_v'] > outputs[idx]['box_v'].max() / thr], 0.5)

        boxes = (outputs[idx]['pred_boxes'][outputs[idx]['box_v'] > outputs[idx]['box_v'].max() / thr])[keep]
        pred_bboxes = torch.clamp(boxes, 0, 1)

        print(f"\nDetected {len(pred_bboxes)} objects similar to exemplars")

        # Convert normalized coordinates to original image coordinates
        pred_boxes_original = pred_bboxes.cpu() / torch.tensor([scale, scale, scale, scale]) * img.shape[-1]

        # Create output image with bounding boxes
        output_image = Image.fromarray((image.permute(1, 2, 0).numpy() * 255).astype(np.uint8))
        draw = ImageDraw.Draw(output_image)

        # Draw exemplar bboxes in blue
        for i, box in enumerate(exemplar_boxes_list):
            x1, y1, x2, y2 = [v * original_width if j % 2 == 0 else v * original_height
                              for j, v in enumerate(box)]
            draw.rectangle([x1, y1, x2, y2], outline='blue', width=2)
            draw.text((x1, y1-20), f'Exemplar {i+1}', fill='blue')

        # Draw detected bboxes in red
        for i, box in enumerate(pred_boxes_original):
            x1, y1, x2, y2 = box.tolist()
            draw.rectangle([x1, y1, x2, y2], outline='red', width=3)
            draw.text((x1, y1-20), f'Detection {i+1}', fill='red')

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
            print(f"Detection {i+1}: x={x1:.1f}, y={y1:.1f}, w={w:.1f}, h={h:.1f}")

    else:
        print("\nNo similar objects detected")

    print("\nFew-shot demo completed successfully!")

if __name__ == '__main__':
    parser = argparse.ArgumentParser('GeCo Few-Shot Demo', parents=[get_argparser()])

    # Set default values for few-shot demo
    parser.set_defaults(
        zero_shot=False,
        image_path='./demo_image.jpg'
    )

    # Add few-shot specific arguments
    parser.add_argument('--exemplar_bboxes',
                       nargs='+',
                       type=str,
                       required=True,
                       help='Bounding boxes defining exemplar regions in the image (format: "x1,y1,x2,y2")')
    parser.add_argument('--output_path', default=None, type=str,
                       help='Path to save the output image with detections (default: {image_path}_fewshot_detected.jpg)')
    parser.add_argument('--mask_path', default=None, type=str,
                       help='Path to save the mask image (default: {image_path}_fewshot_masks.png)')

    args = parser.parse_args()

    # Ensure we're in few-shot mode
    args.zero_shot = False

    # Set default output paths based on image_path if not provided
    if args.output_path is None:
        args.output_path = os.path.splitext(args.image_path)[0] + '_fewshot_detected.jpg'
    if args.mask_path is None:
        args.mask_path = os.path.splitext(args.image_path)[0] + '_fewshot_masks.png'

    print("GeCo Few-Shot Object Detection Demo")
    print("===================================")
    print(f"Zero-shot mode: {args.zero_shot}")
    print(f"Image path: {args.image_path}")
    print(f"Number of exemplar regions: {len(args.exemplar_bboxes)}")
    print(f"Output path: {args.output_path}")
    print(f"Mask path: {args.mask_path}")
    print()

    # Check if image exists
    if not os.path.exists(args.image_path):
        print(f"Error: Image file '{args.image_path}' not found!")
        print("Please provide a valid image path using --image_path argument")
        exit(1)

    demo_fewshot(args)
