from matplotlib import pyplot as plt
learning_rates = [0.01, 0.005, 0.001, 0.0005, 0.0001]
batch_sizes = [32, 64, 128, 256]
dropout_rates = [0.1, 0.2, 0.3, 0.4, 0.5]

for learning_rate in learning_rates:
    for batch_size in batch_sizes:
        for dropout_rate in dropout_rates:
            plt.scatter(learning_rate, batch_size)
plt.xlabel('Learning Rate')
plt.ylabel('Batch Size')
plt.title('Grid Search')
plt.savefig('grid_search.png')
plt.show()


