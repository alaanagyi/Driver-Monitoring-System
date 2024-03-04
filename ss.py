from PIL import Image

# Open the image
image = Image.open("C:\\Users\\iiala\\OneDrive\\Desktop\\DAD\\DAD\\val01\\rec2\\front_depth\\img_0.png")

# Check the data type of the image array
print(image.mode)
# Output: 'L' for int8 grayscale, 'RGB' for int8 RGB image, etc.
