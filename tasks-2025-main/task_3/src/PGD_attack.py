import torch
import torch.nn as nn

class PGDAttack:
    def __init__(self, model, eps=0.1, alpha=0.01, iters=7):
        self.model = model
        self.eps = eps  # The maximum perturbation (epsilon)
        self.alpha = alpha  # Step size for each iteration
        self.iters = iters  # Number of iterations to apply perturbation
        self.loss_fn = nn.CrossEntropyLoss()
    
    def perturb(self, images, labels):
        images = images.clone().detach().requires_grad_()
        labels = labels
        
        perturbation = torch.zeros_like(images)
        perturbation.requires_grad = True
        
        for _ in range(self.iters):
            outputs = self.model(images + perturbation)
            loss = self.loss_fn(outputs, labels)
            self.model.zero_grad()
            loss.backward()

            grad_sign = perturbation.grad.data.sign()
            perturbation.data = perturbation.data + self.alpha * grad_sign
            perturbation.data = torch.clamp(perturbation.data, -self.eps, self.eps)
            perturbation.data = torch.clamp(images + perturbation.data, 0, 1) - images
            perturbation.data = torch.clamp(perturbation.data, -self.eps, self.eps)
            perturbation.grad.data.zero_()

        return images + perturbation
