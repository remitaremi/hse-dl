import torch

# предсказываем оценку за д/з по глубинному обучению
# признаки: оценка за курс по питону, оценка за курс по машинному обучению, качество сна в день выполнения д/з

vanya = torch.tensor([[0.6, 0.8, 0.6]])
masha = torch.tensor([[0.4, 0.5, 1.0]])
misha = torch.tensor([[0.5, 0.6, 0.9]])
sasha = torch.tensor([[1.0, 0.9, 0.2]])
katya = torch.tensor([[1.0, 1.0, 0.8]])

dataset = [
    (vanya, torch.tensor([[0.7]])),
    (masha, torch.tensor([[0.5]])),
    (misha, torch.tensor([[0.6]])),
    (sasha, torch.tensor([[0.8]])),
    (katya, torch.tensor([[1.0]]))
]

torch.manual_seed(2020)

weights = torch.rand((1, 3), requires_grad=True)
bias = torch.rand((1, 1), requires_grad=True)

mse_loss_fn = torch.nn.MSELoss()

optimizer = torch.optim.SGD([weights, bias], lr=1e-5)

def predict_dl_grade(obj: torch.Tensor) -> torch.Tensor:
    return obj @ weights.T + bias

def calc_loss(predicted_value: torch.Tensor, ground_truth: torch.Tensor) -> torch.Tensor:
    return mse_loss_fn(predicted_value, ground_truth)

num_epochs = 10

for i in range(num_epochs):
    for x, y in dataset:
        optimizer.zero_grad()
        grade = predict_dl_grade(x)

        loss = calc_loss(grade, y)
        loss.backward()
        print(loss)
        optimizer.step()


