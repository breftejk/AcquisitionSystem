"""
Generator of test images for system testing
"""
import numpy as np
import cv2
from pathlib import Path
import argparse


def generate_test_sequence(output_folder: str, num_frames: int = 50, width: int = 640, height: int = 480):
    """
    Generates a sequence of test images with a moving object

    Args:
        output_folder: Output folder
        num_frames: Number of frames to generate
        width: Image width
        height: Image height
    """
    output_path = Path(output_folder)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print(f"Generating {num_frames} test images to {output_folder}...")
    
    # Animation parameters
    center_x = width // 2
    center_y = height // 2
    radius = min(width, height) // 4
    
    for i in range(num_frames):
        # Create black background
        img = np.zeros((height, width, 3), dtype=np.uint8)
        
        # Draw grids
        for x in range(0, width, 50):
            cv2.line(img, (x, 0), (x, height), (50, 50, 50), 1)
        for y in range(0, height, 50):
            cv2.line(img, (0, y), (width, y), (50, 50, 50), 1)
        
        # Calculate position of moving circle (circular motion)
        angle = (i / num_frames) * 2 * np.pi
        circle_x = int(center_x + radius * np.cos(angle))
        circle_y = int(center_y + radius * np.sin(angle))
        
        # Draw moving circle
        cv2.circle(img, (circle_x, circle_y), 30, (0, 255, 0), -1)
        cv2.circle(img, (circle_x, circle_y), 30, (255, 255, 255), 2)
        
        # Draw stationary rectangle
        cv2.rectangle(img, (50, 50), (150, 150), (255, 0, 0), -1)
        cv2.rectangle(img, (50, 50), (150, 150), (255, 255, 255), 2)
        
        # Add text with frame number
        text = f"Frame {i}"
        cv2.putText(img, text, (10, height - 10), cv2.FONT_HERSHEY_SIMPLEX, 
                    0.7, (255, 255, 255), 2)
        
        # Add cross in the center
        cv2.line(img, (center_x - 20, center_y), (center_x + 20, center_y), (255, 255, 0), 2)
        cv2.line(img, (center_x, center_y - 20), (center_x, center_y + 20), (255, 255, 0), 2)
        
        # Save image
        filename = f"frame_{i:06d}.png"
        filepath = output_path / filename
        cv2.imwrite(str(filepath), img)
        
        if (i + 1) % 10 == 0:
            print(f"  Generated {i + 1}/{num_frames} frames")
    
    print(f"✓ Successfully generated {num_frames} test images!")
    print(f"  Location: {output_path}")


def generate_gradient_sequence(output_folder: str, num_frames: int = 30):
    """
    Generates a sequence with gradient (for convolution testing)

    Args:
        output_folder: Output folder
        num_frames: Number of frames
    """
    output_path = Path(output_folder)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print(f"Generating {num_frames} gradient test images to {output_folder}...")
    
    width, height = 640, 480
    
    for i in range(num_frames):
        # Horizontal gradient
        img = np.zeros((height, width, 3), dtype=np.uint8)
        
        # Gradient changing over time
        phase = i / num_frames
        
        for x in range(width):
            intensity = int(255 * (x / width) * (0.5 + 0.5 * np.sin(phase * 2 * np.pi)))
            img[:, x] = [intensity, intensity, intensity]
        
        # Add some edges
        cv2.rectangle(img, (200, 150), (440, 330), (255, 255, 255), 3)
        cv2.circle(img, (320, 240), 80, (255, 255, 255), 3)
        
        # Add text
        text = f"Gradient Frame {i}"
        cv2.putText(img, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                    0.7, (255, 255, 255), 2)
        
        # Save
        filename = f"frame_{i:06d}.png"
        filepath = output_path / filename
        cv2.imwrite(str(filepath), img)
    
    print(f"✓ Successfully generated {num_frames} gradient images!")


def main():
    parser = argparse.ArgumentParser(description="Generate test image sequences")
    parser.add_argument("--output", default="test_data/test_sequence", 
                        help="Output folder path")
    parser.add_argument("--frames", type=int, default=50, 
                        help="Number of frames to generate")
    parser.add_argument("--type", choices=["motion", "gradient", "both"], 
                        default="both", help="Type of sequence to generate")
    parser.add_argument("--width", type=int, default=640, help="Image width")
    parser.add_argument("--height", type=int, default=480, help="Image height")
    
    args = parser.parse_args()
    
    if args.type in ["motion", "both"]:
        generate_test_sequence(args.output + "_motion", args.frames, args.width, args.height)
    
    if args.type in ["gradient", "both"]:
        generate_gradient_sequence(args.output + "_gradient", args.frames)
    
    print("\n✓ All test sequences generated successfully!")
    print("\nYou can now use these sequences in the Acquisition System:")
    print("  1. Launch the application: python main.py")
    print("  2. Select 'Image Sequence' as source type")
    print("  3. Browse to the generated folder")
    print("  4. Click 'Open Source'")


if __name__ == "__main__":
    main()
