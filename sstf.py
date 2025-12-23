import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from torchvision.models import resnet50  # Add this import
import os
import numpy as np
from PIL import Image
import random
from tqdm import tqdm

class SemiSupervisedDataset(Dataset):
    def __init__(self, root_dir, classes, transform=None, is_labeled=True):
        self.root_dir = root_dir
        self.classes = classes
        self.transform = transform
        self.is_labeled = is_labeled
        
        self.samples = []
        self.labels = []
        
        for class_idx, class_name in enumerate(classes):
            class_path = os.path.join(root_dir, class_name)
            for img_name in os.listdir(class_path):
                img_path = os.path.join(class_path, img_name)
                if is_labeled:
                    self.samples.append(img_path)
                    self.labels.append(class_idx)
                else:
                    self.samples.append(img_path)
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path = self.samples[idx]
        img = Image.open(img_path).convert('RGB')
        
        if self.transform:
            img = self.transform(img)
        
        if self.is_labeled:
            label = self.labels[idx]
            return img, label
        else:
            return img

class TeacherStudentFramework:
    def __init__(self, config):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize models
        self.student = self._create_model().to(self.device)
        self.teacher = self._create_model().to(self.device)
        self.teacher.load_state_dict(self.student.state_dict())
        
        # Quality estimation branch
        self.quality_branch = QualityEstimationBranch().to(self.device)
        
        # Optimizer
        self.optimizer = torch.optim.AdamW(
            list(self.student.parameters()) + list(self.quality_branch.parameters()),
            lr=config['learning_rate'],
            weight_decay=config['weight_decay']
        )
        
        # Results folder
        os.makedirs('results', exist_ok=True)
    
    def _create_model(self):
        """Create base classification model"""
        model = resnet50(pretrained=True)
        num_features = model.fc.in_features
        model.fc = nn.Linear(num_features, self.config['num_classes'])
        return model
    
    def weak_augmentation(self, x):
        """Weak augmentation for teacher"""
        return x
    
    def strong_augmentation(self, x):
        """Strong augmentation for student"""
        transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
        ])
        return transform(x)
    
    def focal_loss(self, logits, targets, alpha=0.25, gamma=2.0):
        """Focal loss implementation"""
        ce_loss = F.cross_entropy(logits, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = alpha * (1 - pt) ** gamma * ce_loss
        return focal_loss.mean()
    
    def train_epoch(self, labeled_loader, unlabeled_loader):
        self.student.train()
        self.teacher.eval()
        
        total_loss = 0.0
        num_batches = 0
        
        for (labeled_imgs, labels), unlabeled_imgs in zip(labeled_loader, unlabeled_loader):
            labeled_imgs = labeled_imgs.to(self.device)
            labels = labels.to(self.device)
            unlabeled_imgs = unlabeled_imgs.to(self.device)
            
            # Get batch size
            batch_size = labeled_imgs.size(0)
            
            # Weak and strong augmentations for unlabeled data
            weak_views = self.weak_augmentation(unlabeled_imgs)
            strong_views = self.strong_augmentation(unlabeled_imgs)
            
            # Teacher predictions (pseudo-labels)
            with torch.no_grad():
                teacher_logits = self.teacher(weak_views)
                pseudo_labels = F.softmax(teacher_logits, dim=1)
                max_confidence = torch.max(pseudo_labels, dim=1)[0]
                mask = (max_confidence >= self.config['confidence_threshold'])
            
            # Student predictions
            student_logits = self.student(strong_views)
            
            # Consistency loss
            consistency_loss = F.mse_loss(
                student_logits[mask], teacher_logits[mask],
                reduction='mean'
            )
            
            # Quality scores for labeled data
            quality_scores = self.quality_branch(labeled_imgs)
            
            # Supervised loss with quality weighting
            supervised_loss = (quality_scores * self.focal_loss(
                self.student(labeled_imgs), labels
            )).mean()
            
            # Total loss
            loss = supervised_loss + self.config['lambda_cons'] * consistency_loss
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            # Update teacher with EMA
            for teacher_param, student_param in zip(
                self.teacher.parameters(), self.student.parameters()
            ):
                teacher_param.data = (self.config['ema_alpha'] * teacher_param.data + 
                                   (1 - self.config['ema_alpha']) * student_param.data)
            
            total_loss += loss.item()
            num_batches += 1
        
        return total_loss / num_batches
    
    def train(self, labeled_loader, unlabeled_loader, num_epochs):
        for epoch in range(num_epochs):
            loss = self.train_epoch(labeled_loader, unlabeled_loader)
            
            # Ramp up consistency weight
            if epoch < self.config['ramp_up_epochs']:
                self.config['lambda_cons'] = epoch / self.config['ramp_up_epochs']
            
            print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss:.4f}, "
                  f"Consistency Weight: {self.config['lambda_cons']:.2f}")
            
            # Save checkpoint
            if (epoch + 1) % 5 == 0:
                torch.save({
                    'student_state_dict': self.student.state_dict(),
                    'teacher_state_dict': self.teacher.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'epoch': epoch
                }, os.path.join('results', f'checkpoint_epoch_{epoch+1}.pth'))
        
        print("Training completed!")
        print(f"Results saved to: results/")

if __name__ == "__main__":
    # Configuration
    config = {
        'num_classes': 4,
        'learning_rate': 3e-4,
        'weight_decay': 1e-4,
        'confidence_threshold': 0.95,
        'lambda_cons': 0.0,  # Will be ramped up
        'ema_alpha': 0.999,
        'ramp_up_epochs': 5,
        'num_epochs': 40
    }
    
    # Data preparation
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    classes = ['diabetic_retinopathy', 'cataract', 'glaucoma', 'normal']
    dataset_path = 'retina_disease_classification'
    
    # Create labeled and unlabeled datasets
    labeled_dataset = SemiSupervisedDataset(dataset_path, classes, transform, is_labeled=True)
    unlabeled_dataset = SemiSupervisedDataset(dataset_path, classes, transform, is_labeled=False)
    
    # Create data loaders
    labeled_loader = DataLoader(labeled_dataset, batch_size=16, shuffle=True, num_workers=4)
    unlabeled_loader = DataLoader(unlabeled_dataset, batch_size=16, shuffle=True, num_workers=4)
    
    # Initialize and train framework
    framework = TeacherStudentFramework(config)
    framework.train(labeled_loader, unlabeled_loader, config['num_epochs'])