x_data = [1.0, 2.0, 3.0]
y_data = [2.0, 4.0, 6.0]

w = 10.0  # a random guess: random value
a = 0.01  # learning rate


# our model forward pass
def forward(x):  # linear
    return x * w


# Loss function
def loss(x, y):
    y_pred = forward(x)
    return (y_pred - y) * (y_pred - y)


# compute gradient
def gradient(x, y):  # d_loss/d_w
    # loss function 을 w 에 대해 직접 편미분한 결과
    return 2 * x * (x * w - y)


# Before training
print("predict (before training)", "4 hours", forward(4))

# Training loop
for epoch in range(20):
    for x_val, y_val in zip(x_data, y_data):
        grad = gradient(x_val, y_val)
        # Gradient Descent to find the minimum
        w = w - a * grad
        print("\tgrad:", x_val, y_val, round(grad, 2))
        l = loss(x_val, y_val)
    print("progress:", epoch, "w=", round(w, 2), "loss=", round(l, 2))

# After training
print("predict (after training)", "4 hours", forward(4))