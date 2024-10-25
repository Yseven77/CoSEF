from transformers import AutoModel, AutoTokenizer
import torch.nn.functional as F
import random
import numpy as np
from torch.utils.data import Dataset
import os
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
import torch
from sklearn.metrics import accuracy_score,matthews_corrcoef,confusion_matrix, roc_auc_score
from sklearn import  metrics
from datetime import timedelta
import pathlib
import time

#参数
class Config(object):
    _instance = None  # 类变量用于存储实例

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    # """配置参数"""
    def __init__(self):
        if not hasattr(self, 'initialized'):
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.batch_size = 1
            self.num_classes = 2
            self.num_epochs = 12
            self.learning_rate = 0.00001
            self.max_length = 1024
            self.pub_max_length = 512
            self.folder_path = './data/'
            self.save_path = './save_dict/model.ckpt'
            self.best_model = './save_dict/best_model.ckpt'
            self.log_file = './result.txt'
            self.esm_model_name = "./esm"
            self.esm_model = AutoModel.from_pretrained(self.esm_model_name).to(self.device)
            self.tokenizer = AutoTokenizer.from_pretrained(self.esm_model_name)
            self.pubmedbert_model_name = './pubmedbert'
            self.pubmedbert_model = AutoModel.from_pretrained(self.pubmedbert_model_name).to(self.device)
            self.pubmedbert_tokenizer = AutoTokenizer.from_pretrained(self.pubmedbert_model_name)
            special_tokens_dict = {'additional_special_tokens': ["[aspragine]", "[/aspragine]"]}
            self.pubmedbert_tokenizer.add_special_tokens(special_tokens_dict)
            self.pubmedbert_model.resize_token_embeddings(len(self.pubmedbert_tokenizer))
            self.filter_sizes = (1,3,5,7,9,11,13,15,17,19)  # 卷积核尺寸
            self.pub_filter_sizes = (1,3,5,7,9,11,13,15,17,19)    # pubmedbert卷积核尺寸
            self.num_filters = 32  # 卷积核数量(channels数)
            self.hidden_size = 640     # ESM的输出维度
            self.pub_hidden_size = 768  # PUBMEDBERT的输出维度
            self.dropout = 0.3
            self.num = 5
            self.num_layers = 2
            self.lstm_outdim = 16
            self.cap_in_dim = 32
            self.cap_out_dim = 16
            self.seed = 3

#模型部分
class Squash(nn.Module):
    def __init__(self, epsilon=1e-8):
        super().__init__()
        self.epsilon = epsilon

    def forward(self, s: torch.Tensor):  # s: [batch_size, n_capsules, n_features]
        s2 = (s ** 2).sum(dim=-1, keepdims=True)
        return (s2 / (1 + s2)) * (s / torch.sqrt(s2 + self.epsilon))

class Router(nn.Module):
    def __init__(self, in_caps: int, out_caps: int, in_d: int, out_d: int, iterations: int):  # int_d: 前一层胶囊的特征数目
        super().__init__()
        self.in_caps = in_caps  # 胶囊数目
        self.out_caps = out_caps
        self.iterations = iterations
        self.softmax = nn.Softmax(dim=1)
        self.squash = Squash()

        # maps each capsule in the lower layer to each capsule in this layer
        self.weight = nn.Parameter(torch.randn(in_caps, out_caps, in_d, out_d), requires_grad=True)

    def forward(self, u: torch.Tensor):  # 低层胶囊的输入
        """
        input(s) shape: [batch_size, n_capsules, n_features]
        output shape: [batch_size, n_capsules, n_features]
        """

        u_hat = torch.einsum('ijnm,bin->bijm', self.weight, u)
        b = u.new_zeros(u.shape[0], self.in_caps, self.out_caps)
        v = None
        for i in range(self.iterations):
            c = self.softmax(b)
            s = torch.einsum('bij,bijm->bjm', c, u_hat)
            v = self.squash(s)
            a = torch.einsum('bjm,bijm->bij', v, u_hat)
            b = b + a
        return v


config = Config()



class Classifier(nn.Module):

    def __init__(self, num_classes):
        super(Classifier, self).__init__()
        self.esm_model = config.esm_model
        self.pubmedbert_model = config.pubmedbert_model
        self.convs = nn.ModuleList(
            [nn.Conv2d(1, config.num_filters, (k, config.hidden_size)) for k in config.filter_sizes])
        self.convs_pub = nn.ModuleList(
            [nn.Conv2d(1, config.num_filters, (k, config.pub_hidden_size)) for k in config.pub_filter_sizes])
        self.dropout = nn.Dropout(config.dropout)
        self.lstm = nn.LSTM(config.num_filters, config.lstm_outdim, config.num_layers,
                            bidirectional=True, batch_first=True, dropout=config.dropout)
        self.capsule_layer = Router(in_caps=len(config.filter_sizes)+len(config.pub_filter_sizes), out_caps=config.num_classes,
                                    in_d=config.cap_in_dim, out_d=config.cap_out_dim,iterations=5)
        self.fc = nn.Linear(config.num_classes*config.cap_out_dim, num_classes)

    def conv_and_pool(self, x, conv):
        x = F.relu(conv(x)).squeeze(3)
        x = F.max_pool1d(x, x.size(2)).squeeze(2).unsqueeze(1)
        return x

    def forward(self, input_ids, attention_mask,pub_input_ids,pub_attention_mask):
        esm_features = self.esm_model(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state[:,1:-1,:]
        esm_features = esm_features.unsqueeze(1)
        esm_cnn_out = torch.cat([self.conv_and_pool(esm_features, conv) for conv in self.convs], 1)
        out_c, _ = self.lstm(esm_cnn_out)
        pub_features = self.pubmedbert_model(input_ids=pub_input_ids,
                                             attention_mask=pub_attention_mask).last_hidden_state[:,1:-1,:]
        pub_features = pub_features.unsqueeze(1)
        pub_cnn_out = torch.cat([self.conv_and_pool(pub_features, conv) for conv in self.convs_pub], 1)
        out_p, _ = self.lstm(pub_cnn_out)
        out = torch.cat((out_c, out_p), dim=1)
        out = self.capsule_layer(out)
        out = out.reshape(config.batch_size, config.num_classes * config.cap_out_dim)
        logits = self.fc(out)
        return logits



#迭代器
class ProteinDataset(Dataset):
    def __init__(self, sequences, labels, pub_sequences):
        self.sequences = sequences
        self.labels = labels
        self.pub_sequences = pub_sequences

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        sequence = self.sequences[idx]
        label = self.labels[idx]
        pub_squence = self.pub_sequences[idx]

        return sequence, label, pub_squence

# 数据处理
def train_set():
    folder_path = config.folder_path
    fasta_files = ['UniSwiss-Tr-posi.fasta', 'UniSwiss-Tr-nega.fasta']

    Tr_DBP_sequences = []
    Tr_nonDBP_sequences = []

    for fasta_file in fasta_files:
        file_path = os.path.join(folder_path, fasta_file)
        category = "DBP" if "posi" in fasta_file else "nonDBP"

        with open(file_path, 'r') as file:
            lines = file.readlines()
            sequence = ""
            for line in lines:
                if line.startswith(">"):
                    if sequence:
                        if category == "DBP":
                            Tr_DBP_sequences.append(sequence)
                        else:
                            Tr_nonDBP_sequences.append(sequence)
                        sequence = ""
                else:
                    sequence += line.strip()
            if sequence:
                if category == "DBP":
                    Tr_DBP_sequences.append(sequence)
                else:
                    Tr_nonDBP_sequences.append(sequence)

    return Tr_DBP_sequences, Tr_nonDBP_sequences

def train_set_pub():
    folder_path = config.folder_path
    fasta_files = ['UniSwiss-Tr-posi_en.txt', 'UniSwiss-Tr-nega_en.txt']

    Tr_DBP_sequences = []
    Tr_nonDBP_sequences = []

    for fasta_file in fasta_files:
        file_path = os.path.join(folder_path, fasta_file)
        category = "DBP" if "posi" in fasta_file else "nonDBP"

        with open(file_path, 'r') as file:
            lines = file.readlines()
            sequence = ""
            for line in lines:
                if line.startswith(">"):
                    if sequence:
                        if category == "DBP":
                            Tr_DBP_sequences.append(sequence)
                        else:
                            Tr_nonDBP_sequences.append(sequence)
                        sequence = ""
                else:
                    sequence += line.strip()
            if sequence:
                if category == "DBP":
                    Tr_DBP_sequences.append(sequence)
                else:
                    Tr_nonDBP_sequences.append(sequence)

    return Tr_DBP_sequences, Tr_nonDBP_sequences

def test_set():
    folder_path = config.folder_path
    fasta_files = ['UniSwiss-Tst-posi.fasta', 'UniSwiss-Tst-nega.fasta']

    Tst_DBP_sequences = []
    Tst_nonDBP_sequences = []

    for fasta_file in fasta_files:
        file_path = os.path.join(folder_path, fasta_file)
        category = "DBP" if "posi" in fasta_file else "nonDBP"

        with open(file_path, 'r') as file:
            lines = file.readlines()
            sequence = ""
            for line in lines:
                if line.startswith(">"):
                    if sequence:
                        if category == "DBP":
                            Tst_DBP_sequences.append(sequence)
                        else:
                            Tst_nonDBP_sequences.append(sequence)
                        sequence = ""
                else:
                    sequence += line.strip()
            if sequence:
                if category == "DBP":
                    Tst_DBP_sequences.append(sequence)
                else:
                    Tst_nonDBP_sequences.append(sequence)

    return Tst_DBP_sequences, Tst_nonDBP_sequences

def test_set_pub():
    folder_path = config.folder_path
    fasta_files = ['UniSwiss-Tst-posi_en.txt', 'UniSwiss-Tst-nega_en.txt']

    Tst_DBP_sequences = []
    Tst_nonDBP_sequences = []

    for fasta_file in fasta_files:
        file_path = os.path.join(folder_path, fasta_file)
        category = "DBP" if "posi" in fasta_file else "nonDBP"

        with open(file_path, 'r') as file:
            lines = file.readlines()
            sequence = ""
            for line in lines:
                if line.startswith(">"):
                    if sequence:
                        if category == "DBP":
                            Tst_DBP_sequences.append(sequence)
                        else:
                            Tst_nonDBP_sequences.append(sequence)
                        sequence = ""
                else:
                    sequence += line.strip()
            if sequence:
                if category == "DBP":
                    Tst_DBP_sequences.append(sequence)
                else:
                    Tst_nonDBP_sequences.append(sequence)

    return Tst_DBP_sequences, Tst_nonDBP_sequences

rewrite_print = print

# 把数据存入txt文件中
def print(*arg):
    log_file = config.log_file
    log_file = pathlib.Path(log_file)
    rewrite_print(*arg) # 打印到控制台
    with open(log_file,"a", encoding='utf-8') as f:
        rewrite_print(*arg,file=f)

#训练和测试部分

classifier = Classifier(config.num_classes).to(config.device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(classifier.parameters(), lr=config.learning_rate)


def best_exp():
    classifier.eval()
    all_labels = []
    all_preds = []
    all_probs = []  # 存储每个样本属于正类的概率
    with torch.no_grad():
        classifier.load_state_dict(torch.load(config.best_model))
        for sequences, labels, pub_sequences in test_loader:
            sequences_dict = config.tokenizer(sequences, padding=True, truncation=True, return_tensors='pt',
                                              max_length=config.max_length)
            sequences_dict = {k: v.to(config.device) for k, v in sequences_dict.items()}
            pub_sequences = config.pubmedbert_tokenizer(pub_sequences, padding=True, truncation=True,
                                                        return_tensors='pt', max_length=config.pub_max_length)
            pub_sequences = {k: v.to(config.device) for k, v in pub_sequences.items()}
            labels = torch.as_tensor(labels).to(config.device)
            logits = classifier(input_ids=sequences_dict['input_ids'],
                                attention_mask=sequences_dict['attention_mask'],
                                pub_input_ids=pub_sequences['input_ids'],
                                pub_attention_mask=pub_sequences['attention_mask'])
            preds = logits.argmax(dim=1)
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            # 获取正类的概率
            probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()  # 假设正类为索引0
            all_probs.extend(probs)

        accuracy = accuracy_score(all_labels, all_preds)
        conf_matrix = confusion_matrix(all_labels, all_preds)
        print(conf_matrix)
        tp = conf_matrix[0, 0]
        fp = conf_matrix[1, 0]
        fn = conf_matrix[0, 1]
        tn = conf_matrix[1, 1]
        specificity = tn / (tn + fp)
        recall = tp / (tp + fn)
        f1_score = 2 * tp / (2 * tp + fn + fp)
        precision = tp / (tp + fp)
        auc = roc_auc_score(all_labels, all_probs)  # 计算AUC
        mcc = matthews_corrcoef(all_labels, all_preds)
        print(f" sen: {recall:.4f} - spe: {specificity:.4f} - acc: {accuracy:.4f} - pre: {precision:.4f}"
              f" - f1_score: {f1_score:.4f} - MCC: {mcc:.4f} - AUC: {auc:.4f}")
        print('****************************************')

def get_time_dif(start_time):
    """获取已使用时间"""
    end_time = time.time()
    time_dif = end_time - start_time
    return timedelta(seconds=int(round(time_dif)))

# 运行部分
def seed_everything(seed=config.seed):
    """
    设置整个开发环境的seed
    :param seed:
    :return:
    """
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # some cudnn methods can be random even after fixing the seed
    # unless you tell it to be deterministic
    # torch.backends.cudnn.deterministic = True

if __name__ == '__main__':
    # 获取命令行参数
    Tr_DBP_sequences, Tr_nonDBP_sequences = train_set()
    Tst_DBP_sequences, Tst_nonDBP_sequences = test_set()
    Tr_DBP_sequences_pub, Tr_nonDBP_sequences_pub = train_set_pub()
    Tst_DBP_sequences_pub, Tst_nonDBP_sequences_pub = test_set_pub()
    # DataLoader
    train_dataset = ProteinDataset(Tr_DBP_sequences + Tr_nonDBP_sequences,
                                   [0] * len(Tr_DBP_sequences) + [1] * len(Tr_nonDBP_sequences),
                                   Tr_DBP_sequences_pub + Tr_nonDBP_sequences_pub)
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    test_dataset = ProteinDataset(Tst_DBP_sequences + Tst_nonDBP_sequences,
                                  [0] * len(Tst_DBP_sequences) + [1] * len(Tst_nonDBP_sequences),
                                  Tst_DBP_sequences_pub + Tst_nonDBP_sequences_pub)
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False)
    best_exp()

