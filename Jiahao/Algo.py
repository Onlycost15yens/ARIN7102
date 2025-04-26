import torch
import numpy as np
import pandas as pd
from transformers import BertTokenizer, BertModel
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Initialize label encoder
label_encoder = LabelEncoder()

# Function to create embeddings for text
def get_bert_embedding(text, model, tokenizer):
    # Tokenize and convert to tensor
    inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True, max_length=512)
    
    # Get embeddings from the model
    with torch.no_grad():
        outputs = model(**inputs)
    
    # Use the [CLS] token embedding as the sentence embedding
    return outputs.last_hidden_state[:, 0, :].numpy()[0]

def embed_questions_from_xlsx(xlsx_file_path, output_npy_path):
    """
    Load questions from an xlsx file, create BERT embeddings, and save embeddings 
    and question types to a numpy file.
    
    Args:
        xlsx_file_path (str): Path to the xlsx file containing questions
        output_npy_path (str): Path where to save the numpy file
    """
    # Load pre-trained BERT model and tokenizer
    print("Loading BERT model and tokenizer...")
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertModel.from_pretrained('bert-base-uncased')
    
    # Load the xlsx file
    print(f"Loading data from {xlsx_file_path}...")
    df = pd.read_excel(xlsx_file_path)
    

    
    # Create embeddings for questions in the dataset
    print("Creating embeddings for questions...")
    question_embeddings = []
    question_types = []
    
    for i, row in tqdm(df.iterrows(), total=len(df)):
        question = row['Question']
        question_type = row['Question Type'] if 'Question Type' in df.columns else None
        
        embedding = get_bert_embedding(question, model, tokenizer)
        question_embeddings.append(embedding)
        question_types.append(question_type)
    
    # Save embeddings and question types to numpy file
    print(f"Saving embeddings and question types to {output_npy_path}...")
    np.save(output_npy_path, {
        'embeddings': np.array(question_embeddings),
        'question_types': question_types
    })
    
    print(f"Successfully created and saved embeddings for {len(question_embeddings)} questions")


def load_embeddings(npy_path):
    data = np.load(npy_path, allow_pickle=True).item()
    embeddings = data['embeddings']
    question_types = data['question_types']
    return embeddings, question_types


def knn_classifier(embeddings, question_types, inference_embedding):
    from sklearn.neighbors import KNeighborsClassifier
    
    # Initialize KNN classifier
    knn = KNeighborsClassifier(n_neighbors=1)
    
    # Train the KNN classifier
    knn.fit(embeddings, question_types) 
    
    # inference
    prediction = knn.predict([inference_embedding])
    
    return prediction
    
def random_forest_classifier(embeddings, question_types, inference_embedding):
    from sklearn.ensemble import RandomForestClassifier
    
    # Initialize Random Forest classifier
    rf = RandomForestClassifier(n_estimators=200, random_state=42)
    
    # Train the Random Forest classifier
    rf.fit(embeddings, question_types)
    
    # inference
    prediction = rf.predict([inference_embedding])
    
    return prediction

def svm_classifier(embeddings, question_types, inference_embedding):
    from sklearn.svm import SVC
    
    # Initialize SVM classifier with best parameters
    svm = SVC(C=8, gamma=0.01, kernel='rbf')
    
    # Train the SVM classifier
    svm.fit(embeddings, question_types)
    
    # inference
    prediction = svm.predict([inference_embedding])
    
    return prediction



class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.conv3 = nn.Conv1d(out_channels, out_channels * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm1d(out_channels * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes):
        super(ResNet, self).__init__()
        self.in_channels = 64
        # Adapt the first layer to work with embeddings
        # Reshape embeddings to have a channel dimension
        self.conv1 = nn.Conv1d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm1d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, out_channels, blocks, stride=1):
        downsample = None
        if stride != 1 or self.in_channels != out_channels * block.expansion:
            downsample = nn.Sequential(
                nn.Conv1d(self.in_channels, out_channels * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(out_channels * block.expansion),
            )

        layers = []
        layers.append(block(self.in_channels, out_channels, stride, downsample))
        self.in_channels = out_channels * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.in_channels, out_channels))

        return nn.Sequential(*layers)

    def forward(self, x):
        # Add channel dimension
        x = x.unsqueeze(1)
        
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x
    
# Training function
def train_model(model, train_loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
    
    train_loss = running_loss / len(train_loader)
    train_acc = correct / total
    
    # save the model
    torch.save(model.state_dict(), 'Checkpoints/resnet_model.pth')
    
    return train_loss, train_acc

# Validation function
def validate_model(model, val_loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    val_loss = running_loss / len(val_loader)
    val_acc = correct / total
    return val_loss, val_acc, all_preds, all_labels


def inference_model(model, inference_embedding):
    model.eval()
    with torch.no_grad():
        outputs = model(inference_embedding)
        _, predicted = outputs.max(1)
        return predicted.item()
    
def all_models(text):
    # Load pre-trained BERT model and tokenizer
    print("Loading BERT model and tokenizer...")
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertModel.from_pretrained('bert-base-uncased')
    inference_text = text
    inference_embedding = get_bert_embedding(inference_text, model, tokenizer)
    
    # load preprocessed embeddings
    embeddings, question_types = load_embeddings("Checkpoints/question_embeddings.npy")
    
    # Split the data into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(
        embeddings, question_types, test_size=0.01, random_state=42
    )

    predictions = []

    # Convert labels to numerical values
    label_encoder = LabelEncoder()
    y_train_encoded = label_encoder.fit_transform(y_train)
    y_val_encoded = label_encoder.transform(y_val)
    
    # KNN classifier
    prediction = knn_classifier(embeddings, question_types, inference_embedding)
    predictions.append(prediction[0])
    print(f"Given by the KNN classifier, the question type is {prediction[0]}")
    
    # Random Forest classifier
    prediction = random_forest_classifier(embeddings, question_types, inference_embedding)
    predictions.append(prediction[0])
    print(f"Given by the Random Forest classifier, the question type is {prediction[0]}")
    
    # SVM classifier
    prediction = svm_classifier(embeddings, question_types, inference_embedding)
    predictions.append(prediction[0])
    print(f"Given by the SVM classifier, the question type is {prediction[0]}")
    
    # ResNet classifier
    # Load pre-trained ResNet model
    num_classes = len(set(question_types))
    resnet = ResNet(Bottleneck, [3, 4, 6, 3], num_classes=num_classes)
    resnet.load_state_dict(torch.load('Checkpoints/resnet_model.pth', weights_only=True))
    inference_embedding_tensor = torch.tensor(inference_embedding, dtype=torch.float32)
    inference_embedding_tensor = inference_embedding_tensor.unsqueeze(0)    
    
    # Make prediction
    prediction_idx = inference_model(resnet, inference_embedding_tensor)
    prediction_label = label_encoder.inverse_transform([prediction_idx])[0]
    predictions.append(prediction_label)
    print(f"Given by the ResNet classifier, the question type is {prediction_label}")
    
    # return the most frequent prediction
    return max(set(predictions), key=predictions.count), predictions
    

if __name__ == "__main__":
    text = "What color is 30mg prednisone?"
    answer, predictions = all_models(text)
    print(f"The question type is {answer}")
    print(f"The predictions are {predictions}")