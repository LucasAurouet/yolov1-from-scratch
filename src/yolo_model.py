import torch.nn as nn

class YOLOv1(nn.Module):
    def __init__(self, S=7, B=2, C=52, batch_size=16):
        super().__init__()
        self.S = S
        self.B = B
        self.C = C
        self.batch_size = batch_size
        
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(512, 1024, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),

            nn.Conv2d(1024, 1024, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
        )

        # Fully connected layer pour S*S*(B*5 + C)
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(1024 * 7 * 7, 4096),
            nn.ReLU(),
            nn.Linear(4096, self.S*self.S*(self.B*5 + self.C))
        )
    
    def forward(self, x):
        # forward pass through the network
        x = self.features(x)
        x = self.fc(x)
        # reshape output to (N, S, S, C+B*5) grid-like format
        x = x.view( self.batch_size, self.S, self.S, self.C + self.B * 5)
        
        return x

    

