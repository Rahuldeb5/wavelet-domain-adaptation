from torch.utils.data import Dataset
from torchvision import transforms
import tifffile as tif

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
