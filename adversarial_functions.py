import torch
import torch.nn.functional as F

def fgsm_attack(image, epsilon, data_grad):
    """
    Generates an adversarial example by applying the FGSM attack.
    
    Args:
        image (torch.Tensor): The original input image.
        epsilon (float): The perturbation size.
        data_grad (torch.Tensor): The gradient of the loss w.r.t. the input image.
    
    Returns:
        torch.Tensor: The perturbed image.
    """
    sign_data_grad = data_grad.sign()
    
    # Create perturbed image by adjusting each pixel of the input image
    perturbed_image = image + epsilon * sign_data_grad
    
    # clipping to maintain [0,1] range
    perturbed_image = torch.clamp(perturbed_image, 0, 1)
    
    return perturbed_image

def pgd_attack(image, epsilon, data_grad, step_size, num_steps):
    """
    Generates an adversarial example by applying the PGD attack.
    
    Args:
        image (torch.Tensor): The original input image.
        epsilon (float): The perturbation size.
        data_grad (torch.Tensor): The gradient of the loss w.r.t. the input image.
        step_size (float): The step size for the gradient descent.
        num_steps (int): The number of steps to take in the gradient descent.
    
    Returns:
        torch.Tensor: The perturbed image.
    """
    # Initialize perturbed image to input image
    perturbed_image = image.clone().detach()
    
    # Perform gradient descent to find the adversarial example
    for _ in range(num_steps):
        perturbed_image.requires_grad_(True)
        loss = torch.sum(torch.abs(data_grad * (perturbed_image - image)))
        loss.backward()
        
        # Update the perturbed image while respecting the epsilon constraint
        perturbed_image = perturbed_image - step_size * torch.sign(perturbed_image.grad)
        perturbed_image = torch.clamp(perturbed_image, image - epsilon, image + epsilon)
        perturbed_image = torch.clamp(perturbed_image, 0, 1)
        
        perturbed_image.grad.zero_()
    
    return perturbed_image

def l2_attack(image, epsilon, data_grad):
    """
    Generates an adversarial example by applying the L2 attack.
    
    Args:
        image (torch.Tensor): The original input image.
        epsilon (float): The perturbation size.
        data_grad (torch.Tensor): The gradient of the loss w.r.t. the input image.
    
    Returns:
        torch.Tensor: The perturbed image.
    """
    # Normalize gradient by its L2 norm
    data_grad_norm = torch.norm(data_grad, p=2)
    normalized_grad = data_grad / (data_grad_norm + 1e-8)
    
    # perturb image by adding the normalized gradient scaled by epsilon
    perturbed_image = image + epsilon * normalized_grad
    
    # clipping to maintain [0,1] range
    perturbed_image = torch.clamp(perturbed_image, 0, 1)
    
    return perturbed_image

def rotate_image(image, angle):
    """
    Rotates the input image by the specified angle.
    
    Args:
        image (torch.Tensor): The input image tensor.
        angle (float): The angle of rotation in degrees.
        
    Returns:
        torch.Tensor: The rotated image tensor.
    """
    angle_rad = torch.deg2rad(torch.tensor(angle))
    rotation_matrix = torch.tensor([[torch.cos(angle_rad), -torch.sin(angle_rad), 0],
                                   [torch.sin(angle_rad),  torch.cos(angle_rad), 0]])
    rotation_matrix = rotation_matrix.unsqueeze(0).expand(image.size(0), -1, -1)
    
    grid = F.affine_grid(rotation_matrix, image.size(), align_corners=True)
    rotated_image = F.grid_sample(image, grid, align_corners=True)
    
    return rotated_image

def translate_image(image, dx, dy):
    """
    Translates the input image by the specified amounts in the x and y directions.
    
    Args:
        image (torch.Tensor): The input image tensor.
        dx (float): The translation in the x direction.
        dy (float): The translation in the y direction.
        
    Returns:
        torch.Tensor: The translated image tensor.
    """
    translation_matrix = torch.tensor([[1, 0, dx],
                                      [0, 1, dy]])
    translation_matrix = translation_matrix.unsqueeze(0).expand(image.size(0), -1, -1)
    
    grid = F.affine_grid(translation_matrix, image.size(), align_corners=True)
    translated_image = F.grid_sample(image, grid, align_corners=True)
    
    return translated_image

def scale_image(image, sx, sy):
    """
    Scales the input image by the specified amounts in the x and y directions.
    
    Args:
        image (torch.Tensor): The input image tensor.
        sx (float): The scaling factor in the x direction.
        sy (float): The scaling factor in the y direction.
        
    Returns:
        torch.Tensor: The scaled image tensor.
    """
    scaling_matrix = torch.tensor([[sx, 0, 0],
                                  [0, sy, 0]])
    scaling_matrix = scaling_matrix.unsqueeze(0).expand(image.size(0), -1, -1)
    
    grid = F.affine_grid(scaling_matrix, image.size(), align_corners=True)
    scaled_image = F.grid_sample(image, grid, align_corners=True)
    
    return scaled_image

def gaussian_blur(image, kernel_size=5, sigma=0):
    """
    Applies Gaussian blurring to the input image.
    
    Args:
        image (torch.Tensor): The input image tensor.
        kernel_size (int): The size of the Gaussian kernel. Default is 5.
        sigma (float): The standard deviation of the Gaussian kernel. Default is 0.
        
    Returns:
        torch.Tensor: The blurred image tensor.
    """
    blurred_image = F.gaussian_blur(image, kernel_size=kernel_size, sigma=sigma)
    return blurred_image

