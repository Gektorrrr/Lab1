import cv2
import numpy as np


# 4. Функція розрахування ентропії Шенона
def calculate_shannon_entropy(image):
    # Отримуємо гістограму і нормалізуємо її для отримання ймовірностей
    hist = cv2.calcHist([image], [0], None, [256], [0, 256])
    probs = hist.flatten() / hist.sum()
    # Відфільтровуємо нульові значення для уникнення log(0)
    probs = probs[probs > 0]
    # Формула: H = -sum(p * log2(p))
    entropy = -np.sum(probs * np.log2(probs))
    return entropy


# 5. Функція розрахування міри Хартлі
def calculate_hartley_measure(image):
    # Міра Хартлі базується на кількості унікальних станів (рівнів яскравості)
    unique_values = np.unique(image)
    N = len(unique_values)
    if N > 0:
        return np.log2(N)
    return 0


# 6. Функція розрахування Марковського процесу першого порядку (умовна ентропія)
def calculate_markov_entropy(image):
    # Розраховуємо умовну ентропію H(X|Y) = H(X,Y) - H(Y)
    # Розглядаємо горизонтальні пари сусідніх пікселів
    y = image[:, :-1].flatten()  # поточний піксель
    x = image[:, 1:].flatten()  # наступний піксель

    # Спільна ймовірність p(x, y) через 2D гістограму
    joint_hist, _, _ = np.histogram2d(y, x, bins=256, range=[[0, 256], [0, 256]])
    joint_probs = joint_hist.flatten() / joint_hist.sum()
    joint_probs = joint_probs[joint_probs > 0]
    joint_entropy = -np.sum(joint_probs * np.log2(joint_probs))

    # Маргінальна ймовірність p(y)
    marginal_probs = joint_hist.sum(axis=1) / joint_hist.sum()
    marginal_probs = marginal_probs[marginal_probs > 0]
    marginal_entropy = -np.sum(marginal_probs * np.log2(marginal_probs))

    # Умовна ентропія
    conditional_entropy = joint_entropy - marginal_entropy
    return conditional_entropy


# Застосування до вашого зображення
img = cv2.imread('edited-image.jpg')

entropy_s = calculate_shannon_entropy(img)
measure_h = calculate_hartley_measure(img)
entropy_m = calculate_markov_entropy(img)

print(f"Ентропія Шенона: {entropy_s:.4f} біт")
print(f"Міра Хартлі: {measure_h:.4f} біт")
print(f"Марковська ентропія (1-го порядку): {entropy_m:.4f} біт")