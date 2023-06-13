import numpy as np

def noise_augmentation(method, dataset):

    if method == 'base':
        # noise size = 4
        base_noise = np.random.randint(4, size= dataset.shape)
        aug_dataset = dataset + base_noise

    elif method == 'gaussian':
        mu, sigma = np.min(dataset), np.max(dataset) # mean and standard deviation
        gaussian_noise = np.random.normal(mu, sigma, dataset.shape)
        aug_dataset = dataset + gaussian_noise        

    return aug_dataset