from torch.utils.data import Dataset
from torchvision import transforms
import tifffile as tif
from wavelet import wavelet_adapt
from config import DEVICE

normalize_05 = transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))
to_tensor = transforms.ToTensor()
imageTransform = transforms.Compose([to_tensor, normalize_05])

class LandslideDataset(Dataset):
    def __init__(self, img_list, mask_list):
        super().__init__()
        self.img_list = img_list
        self.mask_list = mask_list
    
    def __len__(self):
        return len(self.img_list)
    
    def __getitem__(self, index):
        img = self.img_list[index]
        mask = self.mask_list[index]
        
        img = imageTransform(tif.imread(img))
        mask = (to_tensor(tif.imread(mask)) > 0).float()
        
        return img, mask

class AdaptedDataset(Dataset):
    def __init__(self, img_tensors, mask_list):
        self.img_tensors = img_tensors
        self.mask_list = mask_list
    
    def __len__(self):
        return len(self.img_tensors)
    
    def __getitem__(self, idx):
        img = normalize_05(self.img_tensors[idx])
        mask = (to_tensor(tif.imread(self.mask_list[idx])) > 0).float()
        
        return img, mask
