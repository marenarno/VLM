import pandas as pd
from PIL import Image
from torchvision import transforms
import torch
from transformers import CLIPTokenizer
from transformers import CLIPModel, AdamW
from tqdm import tqdm

# Load your dataset
data_path = 'dataset_JGCJMV_ROI.csv'
df = pd.read_csv(data_path)

# Check the first few entries of the dataframe
#print(df.head())


class CustomCLIPDataset(torch.utils.data.Dataset):
    def __init__(self, dataframe, tokenizer, transform=None):
        self.dataframe = dataframe
        self.tokenizer = tokenizer
        self.transform = transform or transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        row = self.dataframe.iloc[idx]
        image_path = row['image']  # Ensure this column name matches your CSV
        caption = row['caption']  # Ensure this column name matches your CSV

        image = Image.open(image_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        
        # Ensure caption is a string and handle NaN or missing values
        if pd.isna(caption):
            caption = " "  # Replace NaN with a placeholder string
        elif not isinstance(caption, str):
            caption = str(caption)  # Convert non-string captions to string

        inputs = self.tokenizer(caption, padding="max_length", truncation=True, max_length=77, return_tensors="pt")
        
        pixel_values = inputs['pixel_values'].squeeze(0) if 'pixel_values' in inputs else None
        input_ids = inputs['input_ids'].squeeze(0)
        attention_mask = inputs['attention_mask'].squeeze(0)

        return image, input_ids, attention_mask

tokenizer = CLIPTokenizer.from_pretrained('openai/clip-vit-base-patch32')


dataset = CustomCLIPDataset(df, tokenizer)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=16, shuffle=True)

# Example of how to iterate over the DataLoader
for images, input_ids, attention_masks in dataloader:
    # Now you can feed this data into your model
    pass

model = CLIPModel.from_pretrained('openai/clip-vit-base-patch32')
model.train()
model.to('cuda')  # or 'cpu' if you don't have a GPU

optimizer = AdamW(model.parameters(), lr=5e-6)

epochs = 4
for epoch in range(epochs):
    loop = tqdm(dataloader, leave=True)
    for images, input_ids, attention_mask in loop:
        images, input_ids, attention_mask = images.to('cuda'), input_ids.to('cuda'), attention_mask.to('cuda')

        outputs = model(input_ids=input_ids, pixel_values=images, attention_mask=attention_mask, return_loss=True)
        loss = outputs.loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loop.set_description(f'Epoch {epoch+1}')
        loop.set_postfix(loss=loss.item())

model.save_pretrained('fine_tuned_model')
