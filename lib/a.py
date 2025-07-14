def forward(self, x):
    if self.weight is not None:
        x = x * self.r
        x = x @ self.weight.T
        x = x + self.bias
    else:
        x = x + x * self.r + self.bias
    return x
