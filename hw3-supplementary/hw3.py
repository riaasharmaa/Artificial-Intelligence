from scipy.linalg import eigh
import numpy as np
import matplotlib.pyplot as plt

def load_and_center_dataset(filename):
    x = np.load(filename)
    x = x - np.mean(x, axis=0)
    return x

def get_covariance(dataset):
    return (np.dot(np.transpose(dataset), dataset) / (len(dataset) - 1))

def get_eig(S, m):
    w, v = eigh(S, subset_by_index=[len(S)-m, len(S)-1])
    i = np.flip(np.argsort(w))  
    return np.diag(w[i]), v[:, i]

def get_eig_prop(S, prop):
    w, v = eigh(S)
    proportion = np.sum(w) * prop
    new_w, new_v = eigh(S, subset_by_value=[proportion, np.inf])
    i = np.flip(np.argsort(new_w))
    return np.diag(new_w[i]), new_v[:, i]

def project_image(image, U):
    alphas = np.dot(np.transpose(U), image)
    return np.dot(U, alphas)

def display_image(orig, proj):
    # Please use the format below to ensure grading consistency
    # fig, (ax1, ax2) = plt.subplots(figsize=(9,3), ncols=2)
    # return fig, ax1, ax2
    orig = np.reshape(orig, (32, 32)).T 
    proj = np.reshape(proj, (32, 32)).T  

    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(9, 3))
    ax1.set_title('Original')
    ax2.set_title('Projection')

    ax1Map = ax1.imshow(orig, aspect='equal', origin='upper')
    fig.colorbar(ax1Map, ax=ax1)
    ax2Map = ax2.imshow(proj, aspect='equal', origin='upper')
    fig.colorbar(ax2Map, ax=ax2)

    return fig, ax1, ax2

def main():
    x = load_and_center_dataset('YaleB_32x32.npy')
    S = get_covariance(x)
    Lambda, U = get_eig(S, 2)
    projection = project_image(x[0], U)
    fig, ax1, ax2 = display_image(x[0], projection)
    plt.show()

if __name__ == '__main__':
    main()

