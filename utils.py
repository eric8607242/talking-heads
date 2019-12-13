import os
import random

import cv2
import pickle as pkl
import PIL
import imageio
import numpy as np
import matplotlib.pyplot as plt

import torch
import torchvision.transforms.functional as TF
from torchvision import transforms
from face_alignment import FaceAlignment, LandmarksType


def plot_landmarks(frame, landmarks):
    """
    Creates an RGB image with the landmarks. The generated image will be of the same size as the frame where the face
    matching the landmarks.

    The image is created by plotting the coordinates of the landmarks using matplotlib, and then converting the
    plot to an image.

    Things to watch out for:
    * The figure where the landmarks will be plotted must have the same size as the image to create, but matplotlib
    only accepts the size in inches, so it must be converted to pixels using the DPI of the screen.
    * A white background is printed on the image (an array of ones) in order to keep the figure from being flipped.
    * The axis must be turned off and the subplot must be adjusted to remove the space where the axis would normally be.

    :param frame: Image with a face matching the landmarks.
    :param landmarks: Landmarks of the provided frame,
    :return: RGB image with the landmarks as a Pillow Image.
    """
    dpi = 100
    fig = plt.figure(figsize=(frame.shape[0] / dpi, frame.shape[1] / dpi), dpi=dpi)
    ax = fig.add_subplot(111)
    ax.axis('off')
    plt.imshow(np.ones(frame.shape))
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)

    # Head
    ax.plot(landmarks[0:17, 0], landmarks[0:17, 1], linestyle='-', color='green', lw=2)
    # Eyebrows
    ax.plot(landmarks[17:22, 0], landmarks[17:22, 1], linestyle='-', color='orange', lw=2)
    ax.plot(landmarks[22:27, 0], landmarks[22:27, 1], linestyle='-', color='orange', lw=2)
    # Nose
    ax.plot(landmarks[27:31, 0], landmarks[27:31, 1], linestyle='-', color='blue', lw=2)
    ax.plot(landmarks[31:36, 0], landmarks[31:36, 1], linestyle='-', color='blue', lw=2)
    # Eyes
    ax.plot(landmarks[36:42, 0], landmarks[36:42, 1], linestyle='-', color='red', lw=2)
    ax.plot(landmarks[42:48, 0], landmarks[42:48, 1], linestyle='-', color='red', lw=2)
    # Mouth
    ax.plot(landmarks[48:60, 0], landmarks[48:60, 1], linestyle='-', color='purple', lw=2)

    fig.canvas.draw()
    data = PIL.Image.frombuffer('RGB', fig.canvas.get_width_height(), fig.canvas.tostring_rgb(), 'raw', 'RGB', 0, 1)
    plt.close(fig)
    return data



def load_model(model, model_dir, weight_name):
    """
    Args:
        model(nn.Module)
        model_dir(str) : Modle weight dir
        weight_name(str)
        
    """
    filename = f'{type(model).__name__}_{weight_name}.pth'
    state_dict = torch.load(os.path.join(model_dir, filename), map_location={'cuda:2': 'cpu'})

    model.load_state_dict(state_dict)
    return model

def get_e_vector(e_path):
    """
    Args:
        e_path(str) : path to e_vector
    Return:
        e_vector(tensor)
    """
    e_vector = np.load(e_path)
    e_vector = torch.from_numpy(e_vector)
    return e_vector


def generate_image(model, landmark, e_vector, device):
    """ 
    Generator generate image from landmark and e_vector
    Args:
        model(nn.Module) : Generator model which generate image from landmark and e_vector.
        landmark(tensor) : Landmark which type is torch.tensor.
        e_vector(tensor) 
        device(int) : Cuda device number
    Return:
        image(ndarray) : Generated image(RGB)
    """
    e_vector = e_vector.to(device)
    
    image = model(landmark, e_vector)
    image = image.cpu().detach().numpy()
    image = image.transpose(0, 2, 3, 1)
    
    return image

def lm_to_image(model, lm_list, e_vector, device):
    """
    Process a list of landmark to a list of image
    Args:
        model(nn.Module) : Generator model
        lm_list(list) : A list contained landmark
        e_vector(tensor) 
        device(str) : Cuda number or cpu
    """
    image_list = []
    for lm in lm_list:
        lm = TF.to_tensor(lm)
        lm = lm.reshape(1, *lm.shape)
        lm = lm.to(device)

        image = generate_image(model, lm, e_vector, device)
        image_list.append(image[0])
        
    return image_list
        

def generate_lm(input_img , fa):
    """
    Process image to landmark
    Args:
        input_img(ndarray) : Image which will process to landmark
        fa(FaceAlignemnt object)
    Return:
        target_img_lm(ndarray)
    """
    target_img_landmark = fa.get_landmarks(input_img)[0]
    target_img_lm = plot_landmarks(input_img, target_img_landmark)
    target_img_lm = np.array(target_img_lm)
    
    return target_img_lm

def video_to_lm(video_path, device, fa):
    """
    Process a video to a list of landmark
    Args:
        video_path(str)
        fa(FaceAlignment) : FaceAlignment object
    Return:
        lm_list(list): A list of landmark generated from video
        lm_image_list(list) : A list of image
        frame_rate(int) : Frame rate to generated video
    """
    videocap = cv2.VideoCapture(video_path)
    frame_rate = int(videocap.get(cv2.CAP_PROP_FPS))
    
    #fa = FaceAlignment(LandmarksType._2D, device=device)
    lm_list = []
    lm_image_list = []
    
    ret, image = videocap.read()
    while ret:
        #image = cv2.resize(image,(256,256))
        #lm = generate_lm(image , fa)
        lm, image = image_to_lm(image, fa)
        lm_list.append(lm)
        
        image = image[:,:,::-1]
        lm_image_list.append(image)
        
        ret, image = videocap.read()
        
    return lm_list, lm_image_list, frame_rate

def image_to_lm(image, fa):
    """
    Process a image to landmark
    Args:
        image(ndarray) 
        fa(FaceAlignment) : FaceAlignment object
    Return:
        lm(ndarray) : Landmark generated from image
    """
    image = cv2.resize(image, (256, 256))
    landmark = generate_lm(image, fa)
    
    return landmark, image


        
def process_image(image):
    """
    Args:
        image(ndarray): image array
    Return:
        process_image(ndarray)
    """
    image = (image * 255.0).clip(0, 255)
    image = np.uint8(image)
    
    return image
    

def image_to_video(images, lm_images, video_path, frame_rate):
    """
    Merge images to video
    Args:
        images(list) : A list contain several images which is generated from Genterator to merge 
        video_path(str) : Video path        
    """    
    writer = imageio.get_writer(video_path, fps=frame_rate)

    for image, lm_image in zip(images, lm_images):
        image = process_image(image)
        
        frame = np.concatenate((image, lm_image), axis=1)
        
        writer.append_data(frame)
    writer.close()
    
    
def generate_moving_video(model, video_path, e_vector, output_path, device, fa):
    """
    Generate video from a source video
    Args:
        model(nn.Module) : Generate model
        video_path(str) : The video path which will be processed to landmark
        e_vector(tensor)
        output_path(str) : The path generated video save
        device(str) : Cuda number or cpu
        frame_rate(int) : Frame rate to generated video
        fa(FaceAlignemnt object)
    """
    lm_list, lm_image_list, frame_rate = video_to_lm(video_path, device, fa)
    image_list = lm_to_image(model, lm_list, e_vector, device)
    image_to_video(image_list, lm_image_list, output_path, frame_rate)

def generate_moving_image(model, image, e_vector, device, fa):
    """
    Generate image from a image
    Args:
        image(ndarray)
        e_vector(tensor) 
        device(str) : Cuda number or cpu
        fa(FaceAlignemnt object)
    Return:
        g_image(ndarray)
    """
    landmark, _ = image_to_lm(image, fa)
    landmark = landmark[:220, :]
    landmark = cv2.resize(landmark, (256, 256))
    plt.imshow(landmark)
    g_image = lm_to_image(model, [landmark], e_vector, device)[0]
    return g_image


def generate_e_vector(model, save_path, fa, device, image=None, video_path=None):
    """
    Generate e_vector and save in npy file
    Args:
        model(nn.Module) : Embedder_network
        save_path(str) : the path to save npy file
        fa(FaceAlignment object)
        device(str) : Cuda number or cpu
        image(ndarray)
        video_path(str)
    """
    if image is not None:
        lm, image = image_to_lm(image, fa)
        lms, images = [], []
        for i in range(8):
            lms.append(lm)
            images.append(image)
            
    elif video is not None:
        lms, images, _ = video_to_lm(video_path, device, fa)
    else:
        print("No video and image input")
        raise
        
    embedding = []
    for lm, image in zip(lms, images):
        x = PIL.Image.fromarray(image, 'RGB')
        y = plot_landmarks(image, lm)
        x, y = _transform(x, y)
        
        embedding.append(torch.stack((x, y)))
    embedding = torch.stack(embedding)
    
    K = embedding.shape[0]
    # Calculate average encoding vector for video
    x, y = embedding[:, 0, ...], embedding[:, 1, ...]
    x, y = x.to(device), y.to(device)

    e_vectors = model(x, y).reshape(1, K, -1)  # K, len(e)
    e_hat = e_vectors.mean(dim=1)
        
    np.save(save_path, e_hat.cpu().detach().numpy())    
    
    
    
def _transform(image, landmark):
    """
    Data augmentation
    """
    r_seed = random.randrange(210, 300)
    p_value = (256 - 210) //2 if r_seed < 256 else 0
    resize = transforms.Resize(size=(r_seed, r_seed))
    image = resize(image)
    landmark = resize(landmark)
        
    pixel = np.array(image)
        
    pad = transforms.Pad(p_value, (pixel[0, 0, 0], pixel[0, 0, 1], pixel[0, 0, 2]))
    image = pad(image)
    landmark = pad(landmark)

    i, j, h, w = transforms.RandomCrop.get_params(
        image, output_size=(256, 256))

    image = TF.crop(image, i, j, h, w)
    landmark = TF.crop(landmark, i, j, h, w)

    if random.random() > 0.5:
        image = TF.hflip(image)
        landmark = TF.hflip(landmark)

    resize = transforms.Resize(size=(256, 256))
    image = resize(image)
    landmark = resize(landmark)

    image = TF.to_tensor(image)
    landmark = TF.to_tensor(landmark)

    return image, landmark