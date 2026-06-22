#!/usr/bin/env python3
"""
Feature visualization for DAFNet attention layers.

Usage:
    python scripts/visualize.py \\
        --checkpoint path/to/checkpoint.pth \\
        --image path/to/image.jpg \\
        --model_name dafnet_tiny \\
        --num_classes 3 \\
        --output_dir ./visualizations

This script extracts and visualizes pre-attention and post-attention
feature maps from DAFNet's Attention modules.
"""

import argparse
import os
import sys

import torch
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dafnet.model import create_model, Attention


class FeatureVisualizer:
    """Extract and visualize feature maps from Attention modules."""

    def __init__(self, model):
        self.model = model
        self.features = {}
        self.hooks = []

        def _hook_factory(name):
            def hook(module, input, output):
                self.features[name] = {
                    'pre_attention': input[0].detach(),
                    'post_attention': output.detach(),
                }
            return hook

        # Register hooks on all Attention modules
        for name, module in self.model.named_modules():
            if 'attention' in name and isinstance(module, Attention):
                self.hooks.append(
                    module.register_forward_hook(_hook_factory(name))
                )

    def visualize(self, img_tensor, layer_name, output_dir='.'):
        """Extract and save feature maps for a given layer."""
        self.features.clear()

        device = next(self.model.parameters()).device
        with torch.no_grad():
            _ = self.model(img_tensor.unsqueeze(0).to(device))

        feat_data = self.features.get(layer_name)
        if feat_data is None:
            available = "\n".join(self.features.keys())
            raise ValueError(
                f"Invalid layer name '{layer_name}'. Available layers:\n{available}"
            )

        os.makedirs(output_dir, exist_ok=True)

        def save_pure_feature(feat, filename):
            feat = feat.mean(dim=1) if feat.dim() == 4 else feat
            feat = feat.squeeze().cpu().numpy()

            fig = plt.figure(frameon=False, dpi=500)
            ax = plt.Axes(fig, [0., 0., 1., 1.])
            ax.set_axis_off()
            fig.add_axes(ax)
            ax.imshow(feat, cmap='viridis', aspect='auto')
            plt.savefig(
                os.path.join(output_dir, filename),
                bbox_inches='tight', pad_inches=0,
                transparent=True, dpi=500,
            )
            plt.close(fig)

        save_pure_feature(feat_data['pre_attention'], f"{layer_name}_pre.png")
        save_pure_feature(feat_data['post_attention'], f"{layer_name}_post.png")

        print(f"Feature maps saved to {output_dir}/")

    def list_layers(self):
        """Print all available attention layer names."""
        print("Available attention layers:")
        for name, module in self.model.named_modules():
            if isinstance(module, Attention):
                print(f"  {name}")

    def cleanup(self):
        """Remove all hooks."""
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()


def main():
    parser = argparse.ArgumentParser(
        description="DAFNet Feature Map Visualization"
    )
    parser.add_argument(
        '--checkpoint', type=str, required=True,
        help='Path to model checkpoint (.pth)',
    )
    parser.add_argument(
        '--image', type=str, required=True,
        help='Path to input image',
    )
    parser.add_argument(
        '--model_name', type=str, default='dafnet_tiny',
        help='Model architecture name',
    )
    parser.add_argument(
        '--num_classes', type=int, default=3,
        help='Number of output classes',
    )
    parser.add_argument(
        '--layer', type=str, default=None,
        help='Specific attention layer to visualize (default: first layer)',
    )
    parser.add_argument(
        '--output_dir', type=str, default='./visualizations',
        help='Output directory for feature maps',
    )
    parser.add_argument(
        '--list_layers', action='store_true',
        help='List available attention layers and exit',
    )
    parser.add_argument(
        '--device', type=str, default='cuda:0',
        help='Device to use (default: cuda:0)',
    )

    args = parser.parse_args()

    # Create model
    model = create_model(args.model_name, num_classes=args.num_classes)
    model = model.to(args.device)

    # Load checkpoint
    checkpoint = torch.load(args.checkpoint, map_location=args.device)
    model.load_state_dict(checkpoint["net"], strict=False)
    model.eval()

    visualizer = FeatureVisualizer(model)

    if args.list_layers:
        visualizer.list_layers()
        visualizer.cleanup()
        return

    # Load and preprocess image
    img = Image.open(args.image).convert('RGB')
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ])
    img_tensor = transform(img)

    # Determine layer to visualize
    layer_name = args.layer
    if layer_name is None:
        # Default to first attention layer
        for name, module in model.named_modules():
            if isinstance(module, Attention):
                layer_name = name
                break

    if layer_name is None:
        print("No Attention layers found in the model.")
        visualizer.cleanup()
        return

    print(f"Visualizing layer: {layer_name}")
    try:
        visualizer.visualize(img_tensor, layer_name, args.output_dir)
    except ValueError as e:
        print(f"Error: {e}")

    visualizer.cleanup()


if __name__ == '__main__':
    main()