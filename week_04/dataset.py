import os,shutil,uuid
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split

class ArmorDataset(Dataset):
    def __init__(self):
        self.root_dir = "datasets/rmset/data"
        self.img_size = 256
        self.transform = transforms.Compose([
        transforms.Resize((self.img_size, self.img_size)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        self.file_list = self._get_file_list()

    def _get_file_list(self):
        file_list = []
        for group_dir in os.listdir(self.root_dir):
            group_path = os.path.join(self.root_dir, group_dir, "images")
            if os.path.exists(group_path):
                for filename in os.listdir(group_path):
                    if filename.endswith(".jpg") or filename.endswith(".png"):
                        file_list.append((group_dir,filename))
        return file_list
        
    def decode_label(self,label_path):
        with open(label_path, 'r') as f:
            label = {
                "labels": [],
                "boxes": []
            }
            while True:
                label_line = f.readline()
                if not label_line:
                    break
                label_parts = label_line.strip().split()
                x1,y1,x2,y2,x3,y3,x4,y4 = [float(x) for x in label_parts[1:]]
                x = [x1,x2,x3,x4]
                y = [y1,y2,y3,y4]
                label["labels"].append(int(label_parts[0]))
                label["boxes"].append( 
                    # 左下和右上
                    [(min(x)+max(x))/2,(min(y)+max(y))/2,max(x)-min(x),max(y)-min(y)]
                )
        return label

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        group_dir,img_filename = self.file_list[idx]
        img_path = os.path.join(self.root_dir,group_dir,"images",img_filename)
        image = Image.open(img_path).convert("RGB")

        label_filename = img_filename.replace(".jpg", ".txt").replace(".png", ".txt")
        label_path = os.path.join(self.root_dir,group_dir,"labels", label_filename)
        label = self.decode_label(label_path)

        if self.transform:
            image = self.transform(image)

        return image, label

def fmt_dataset(dataset,file_list,save_path):
    os.makedirs(save_path,exist_ok=True)
    os.makedirs(os.path.join(save_path,'images'),exist_ok=False)
    os.makedirs(os.path.join(save_path,'labels'),exist_ok=False)
    for group_dir,img_filename in file_list:
        rid = uuid.uuid4()
        src_img_path = os.path.join(dataset.root_dir,group_dir,"images",img_filename)
        label_filename = img_filename.replace(".jpg", ".txt").replace(".png", ".txt")
        label_path = os.path.join(dataset.root_dir,group_dir,"labels", label_filename)
        label_dict = dataset.decode_label(label_path)
        with open(os.path.join(save_path,'labels',f'{rid}.txt'),mode='w+') as fp:
            for label,bbox in zip(label_dict['labels'],label_dict['boxes']):
                fp.write(f'{label} {bbox[0]} {bbox[1]} {bbox[2]} {bbox[3]}\n')
        dst_img_path = os.path.join(save_path,'images',f'{rid}{os.path.splitext(img_filename)[-1]}')
        shutil.copy(src_img_path,dst_img_path)

if __name__=='__main__':
    dataset = ArmorDataset()
    train_fl,test_fl =  train_test_split(dataset.file_list, train_size=0.8, random_state=42)
    fmt_dataset(dataset,test_fl,'datasets/rmset/test')
    fmt_dataset(dataset,train_fl,'datasets/rmset/train')
