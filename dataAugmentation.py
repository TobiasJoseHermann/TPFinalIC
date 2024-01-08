from torchvision.transforms import RandomHorizontalFlip, RandomRotation
from PIL import Image
from torchvision import transforms
import os

# Define your augmentation transformations
augmentations = transforms.Compose([
    RandomHorizontalFlip(),
    RandomRotation(30),
    # Add more transformations if you want
])

# Calculate the number of images in the "Early Blight" and "Late Blight" classes
num_blight_images = len(os.listdir('./Potato Train/Potato_Early_Blight')) 

# Calculate how many additional "Healthy" images you need
num_healthy_images = len(os.listdir('./Potato Train/Potato_Healthy'))
num_additional_images = num_blight_images - num_healthy_images

# Apply the augmentations to the "Healthy" images
healthy_images = os.listdir('./Potato Train Augmentation/Potato_Healthy')
for i in range(num_additional_images):
    image_path = os.path.join('./Potato Train Augmentation/Potato_Healthy', healthy_images[i % num_healthy_images])
    image = Image.open(image_path)
    Augmentation_image = augmentations(image)
    # Save the Augmentation image to disk
    Augmentation_image.save(os.path.join('./Potato Train Augmentation/Potato_Healthy', f'Augmentation_{i}.jpg'))


# Train 951 early 912 late 112 heatlh Levantamos todo a 1000? a 951?