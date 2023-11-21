import torch
import pandas
from torch.utils.data import Dataset, DataLoader

torch.manual_seed(2023)

class TitanicDataset(Dataset):
    def __init__(self):
        super().__init__()
        self.df = pandas.read_csv('https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv')
        # оставляем только полезные столбцы
        self.df = self.df[['Survived', 'Pclass', 'Sex', 'Age', 'SibSp', 'Fare']]
        # находим моду для каждого столбца и заменяем на нее NA
        for column in self.df:
            col_no_na = self.df[column].dropna()
            col_mode = col_no_na.mode()[0]
            self.df[column] = self.df[column].fillna(col_mode)

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, item):
        row = self.df.iloc[item]
        alive = torch.Tensor([1, 0])
        dead = torch.Tensor([0, 1])
        # добавляем признак пола
        sex = 1 if row['Sex'] == 'female' else 0
        y = alive if row['Survived'] else dead
        x = torch.Tensor([row['Age'], row['Fare'], row['SibSp'], row['Pclass'], sex])
        return x, y


titanic_dataset = TitanicDataset()
dataloader = DataLoader(dataset=titanic_dataset, batch_size=2, shuffle=True)

class SurvivalPredictorPerceptron(torch.nn.Module):
    def __init__(self, input_size: int, hidden_size: int, output_size: int):
        super().__init__()
        self.fully_connected_layer = torch.nn.Linear(input_size, hidden_size, bias=True)
        self.out_layer = torch.nn.Linear(hidden_size, output_size, bias=True)
        self.sigmoid = torch.nn.Sigmoid()
        self.relu = torch.nn.ReLU()
        self.softmax = torch.nn.Softmax()

    def forward(self, x):
        x = self.fully_connected_layer(x)
        x = self.relu(x)
        x = self.out_layer(x)
        x = self.softmax(x)

        return x

model = SurvivalPredictorPerceptron(input_size=5, hidden_size=50, output_size=2)
optimizer = torch.optim.SGD(model.parameters(), lr=1e-4)
loss_fn = torch.nn.CrossEntropyLoss()

num_epochs = 20

for epoch in range(num_epochs):
    error = 0
    for x, y in dataloader:
        optimizer.zero_grad()
        prediction = model(x)
        loss = loss_fn(prediction, y)
        error += loss

        loss.backward()
        optimizer.step()

    print(error/len(titanic_dataset))
