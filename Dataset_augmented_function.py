import os
from PIL import Image, ImageEnhance, ImageOps, ImageFilter
import random
import numpy as np
import cv2

def find_perspective_transform_matrix(original_points, new_points):
    # 将点转换为所需格式
    original_points = np.float32(original_points)
    new_points = np.float32(new_points)

    # 计算透视变换矩阵
    matrix = cv2.getPerspectiveTransform(original_points, new_points)
    return matrix

def random_perspective_transform(image):
    width, height = image.size

    # 定义原始点和新点
    original_points = np.float32([[0, 0], [width, 0], [width, height], [0, height]])
    left = random.uniform(0.1, 0.3) * width
    top = random.uniform(0.1, 0.3) * height
    right = width - left
    bottom = height - top
    new_points = np.float32([[left, top], [right, top], [right, bottom], [left, bottom]])

    # 计算透视变换矩阵
    matrix = cv2.getPerspectiveTransform(original_points, new_points)

    # 应用透视变换
    transformed_image = cv2.warpPerspective(np.array(image), matrix, (width, height))

    # 将OpenCV图像转换回PIL格式
    return Image.fromarray(cv2.cvtColor(transformed_image, cv2.COLOR_BGR2RGB))

def random_blur(image):
    if random.random() > 0.5:
        return image.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.1, 2.0)))
    return image


def random_rotation(image):
    """
    Apply a random rotation to the image without cropping it.
    """
    # Randomly choose a rotation angle.
    angle = random.randint(-30, 30)

    # Rotate the image without cropping.
    return image.rotate(angle, expand=True)

def add_random_noise(image):
    """
    Add random noise to an image.
    """
    # Convert image to array
    image_array = np.array(image)

    # Generate noise
    noise = np.random.randint(-25, 25, image_array.shape, dtype='int16')

    # Add noise and ensure values remain in the proper range
    image_array = np.clip(image_array + noise, 0, 255).astype('uint8')

    # Convert array back to image
    noisy_image = Image.fromarray(image_array)
    return noisy_image


def augment_image(image):
    """
    Apply random transformations to an image to create a new, augmented image.
    """
    # Random rotation
    image = random_rotation(image)


    # Random flip
    if random.random() > 0.5:
        image = ImageOps.mirror(image)

    # Random color change
    enhancer = ImageEnhance.Color(image)
    image = enhancer.enhance(random.uniform(0.5, 1.5))


    # Random contrast change
    contrast_enhancer = ImageEnhance.Contrast(image)
    image = contrast_enhancer.enhance(random.uniform(0.5, 1.5))

    # Random brightness change
    brightness_enhancer = ImageEnhance.Brightness(image)
    image = brightness_enhancer.enhance(random.uniform(0.5, 1.5))

    # Random perspective transformation
    if random.random() > 0.5:
        image = random_perspective_transform(image)

    # Random blur
    image = random_blur(image)

    if random.random() > 0.5:
        image = add_random_noise(image)

    return image



def augment_dataset(dataset_dir, augmented_dir, target_count=517):
    """
    Augment the images in each subdirectory of the dataset directory
    until each has `target_count` images, and save them in a new directory.
    """
    # Create a new top-level directory for augmented images
    os.makedirs(augmented_dir, exist_ok=True)

    for brand_dir in os.listdir(dataset_dir):
        brand_path = os.path.join(dataset_dir, brand_dir)
        if not os.path.isdir(brand_path):
            continue

        images = os.listdir(brand_path)
        num_images = len(images)
        augmentations_needed = target_count - num_images

        # Create a new directory for each brand in the augmented directory
        new_brand_dir_path = os.path.join(augmented_dir, brand_dir)
        os.makedirs(new_brand_dir_path, exist_ok=True)

        # Augment images until the target count is reached
        while augmentations_needed > 0:
            for image_name in images:
                if augmentations_needed <= 0:
                    break

                image_path = os.path.join(brand_path, image_name)
                with Image.open(image_path) as img:
                    new_img = augment_image(img)

                    # Create a new image name and save the augmented image
                    new_image_name = f"{brand_dir}-{num_images + 1}.png"
                    new_img.save(os.path.join(new_brand_dir_path, new_image_name))

                    num_images += 1
                    augmentations_needed -= 1


# Example usage
original_dataset_dir = 'E:\\MV-coursework\\Car_Logo_Dataset'  # Replace with your original dataset directory
new_augmented_dir = 'E:\\MV-coursework\\Augmented_Car_Logo_Dataset_enhenced'  # Path for the new directory to store augmented images
augment_dataset(original_dataset_dir, new_augmented_dir)
