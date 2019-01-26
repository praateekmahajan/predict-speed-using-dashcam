# Resnet + Conv3d
class FlowCalculator(nn.Module):
    def __init__(self):
        super(FlowCalculator, self).__init__()
        resnet = list(models.resnet18(pretrained=True).children())[:-2] #all layer expect last 2 layers
        self.resnet = nn.Sequential(*resnet)
        for param in self.resnet.parameters():
            param.requires_grad = False
        self.avgpool = nn.AdaptiveAvgPool2d((4,4))
        self.conv1 = nn.Conv3d(512, 128, kernel_size=(3,3,3), padding=(1,0,0), bias=False)
        self.bn1 = nn.BatchNorm3d(128)
        self.conv2 = nn.Conv3d(128, 64, kernel_size=(3,2,3), padding=(1,0,0), bias=False)
        self.bn2 = nn.BatchNorm3d(64)
        self.conv3 = nn.Conv3d(64, 32, kernel_size=(2,1,3), padding=(0,0,0), bias=False)
        self.bn3 = nn.BatchNorm3d(32)
#         self.conv4 = nn.Conv3d(32, 16, kernel_size=(2,3,3))
#         self.bn4 = nn.BatchNorm3d(16)
        self.fc1 = nn.Linear(128, 32, bias=False)
        self.bn5 = nn.BatchNorm1d(32)
        self.fc2 = nn.Linear(32, 16, bias=False)
        self.bn6 = nn.BatchNorm1d(16)
        self.fc3 = nn.Linear(16, 1)
        self.relu = nn.ReLU(inplace=True)
        
        self.dp1 = nn.Dropout()
        self.dp2 = nn.Dropout(p=0.3)
        
    def forward(self, x1, x2):
        x1 = self.resnet(x1)
        x2 = self.resnet(x2)
        x = torch.cat([x1.unsqueeze(dim=2), x2.unsqueeze(dim=2)], dim=2)
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.relu(self.bn3(self.conv3(x)))
        x = x.view(-1, 128)
        x = self.relu((self.fc1(x)))
        x = self.relu((self.fc2(x)))
        x = self.fc3(x)
        return x

model = FlowCalculator().to(device)