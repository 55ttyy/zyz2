import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

# 定义超参数
batch_size = 64
learning_rate = 0.001
num_epochs = 5
#
# 数据预处理
# transform = transforms.Compose([
#     transforms.ToTensor(),
#     transforms.Normalize((0.5,), (0.5,))
# ])

# # 加载MNIST数据集
# train_dataset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
# train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
#
# test_dataset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
# test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)
# for images, labels in train_loader:
#     print(images.shape)
#     print(labels.shape)
# 定义神经网络模型
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.fc1 = nn.Linear(64 * 7 * 7, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 10)

    def forward(self, x):
        x = self.pool(torch.relu(self.bn1(self.conv1(x))))
        x = self.pool(torch.relu(self.bn2(self.conv2(x))))
        x = x.view(-1, 64 * 7 * 7)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

model = Net()

def hook_fn(module, input, output):
    print(f"层名称：{module.__class__.__name__},输入性状{input[0].shape},输出性状:{output.shape}")
for layer in model.children():
    layer.register_forward_hook(hook_fn)
a=torch.randn(2,1,28,28)
b=model(a)
print(b.shape)
#
# # 定义损失函数和优化器
# criterion = nn.CrossEntropyLoss()
# optimizer = optim.Adam(model.parameters(), lr=learning_rate)
#
# # 训练模型
# for epoch in range(num_epochs):
#     for i, (images, labels) in enumerate(train_loader):
#         # 前向传播
#         outputs = model(images)
#         loss = criterion(outputs, labels)
#
#         # 反向传播和优化
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()
#
#         if (i + 1) % 100 == 0:
#             print(f'Epoch [{epoch + 1}/{num_epochs}], Step [{i + 1}/{len(train_loader)}], Loss: {loss.item():.4f}')
#
# print("训练完成")
#
# # 测试模型
# model.eval()  # 将模型设置为评估模式
# with torch.no_grad():
#     correct = 0
#     total = 0
#     for images, labels in test_loader:
#         outputs = model(images)
#         _, predicted = torch.max(outputs.data, 1)
#         total += labels.size(0)
#         correct += (predicted == labels).sum().item()
#
#     print(f'测试集上的准确率: {100 * correct / total:.2f}%')
#
# # 进行预测
# def predict(image):
#     model.eval()
#     with torch.no_grad():
#         image = transform(image).unsqueeze(0)  # 添加批次维度
#         outputs = model(image)
#         _, predicted = torch.max(outputs.data, 1)
#         return predicted.item()
#
# # 示例预测
# # 获取测试数据集中一张图片
# test_image, test_label = test_dataset[0]
#
# # 显示测试图片
# plt.imshow(test_image.squeeze(), cmap='gray')
# plt.title(f'真实标签: {test_label}')
# plt.show()
#
# # 进行预测
# predicted_label = predict(test_image)
# print(f'预测的标签: {predicted_label}')
