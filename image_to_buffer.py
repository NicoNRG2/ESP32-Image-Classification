import numpy as np
from PIL import Image

def image_to_static_array(image_path):
    # Open the image and convert to grayscale
    image = Image.open(image_path).convert('L')
    # Resize to 96x96
    image = image.resize((96, 96))
    
    # Convert image to numpy array
    image_data = np.array(image).flatten()
    
    # Convert numpy array to C array format
    c_array_str = ', '.join(map(str, image_data))
    print(f'static const uint8_t image_data[96 * 96] = {{ {c_array_str} }};')

# Example usage
image_path = 'dataset/test/1/data-50i4kvjf_jpg.rf.aa96aa11909b08e33d87c799d01fa062.jpg'
image_to_static_array(image_path)
