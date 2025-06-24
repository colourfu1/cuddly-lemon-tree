# 示例：在C3模块中加入CBAM（假设你已定义CBAM）
class C3CBAM(nn.Module):
    def __init__(self, c1, c2, n=1, shortcut=True):
        super().__init__()
        self.cv1 = Conv(c1, c2, 1)
        self.cv2 = Conv(c2, c2, 3)
        self.cv3 = Conv(c2, c2, 1)
        self.cbam = CBAM(c2)  # 加入CBAM
        self.m = nn.Sequential(*[Bottleneck(c2, c2) for _ in range(n)])
        self.shortcut = shortcut

    def forward(self, x):
        y1 = self.cv1(x)
        y2 = self.cv2(y1)
        y3 = self.cv3(y2)
        y4 = self.cbam(y2)  # 经过CBAM
        y5 = self.m(y4)
        if self.shortcut:
            return y5 + y1
        else:
            return y5