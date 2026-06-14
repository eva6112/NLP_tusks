import numpy as np
import matplotlib.pyplot as plt


def perform_pca(X, n_components):
    n = X.shape[0]

    #центрированная матрица
    mean_X = np.mean(X, axis=0)
    Xc = X - mean_X

    #ковариационная матрица
    C = (1 / (n - 1)) * (Xc.T @ Xc)

    #собственные числа и векторы
    eigenvalues, eigenvectors = np.linalg.eigh(C)

    #сортировка собственных чисел и векторов по убыванию
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]

    #выбор k главных направлений - матрица перехода
    V_k = eigenvectors[:, :n_components]

    #расчет координат проекций
    projections = Xc @ V_k

    #общая дисперсия и объясненная доля
    total_variance = np.sum(eigenvalues)
    explained_ratio = eigenvalues / total_variance

    return mean_X, projections, eigenvectors, eigenvalues, explained_ratio

print("Пример 1: снижение размерности 2D -> 1D")
X1 = np.array([[1, 2],
               [3, 3],
               [5, 4]])

mean_X1, p1, e_vecs1, e_vals1, expl_ratio1 = perform_pca(X1, n_components=1)
v1 = e_vecs1[:, 0]  #первая главная компонента

print(f"Собственные числа: {e_vals1}")
print(f"Первая главная компонента (направление): {v1}")
print(f"Проекции точек:\n{p1}")

#визуализация
plt.figure(figsize=(8, 6))
plt.scatter(X1[:, 0], X1[:, 1], color='black', s=100, label='Исходные точки A, B, C')

#отрисовка линии главной компоненты
t = np.linspace(-3, 3, 100)     #массив из 100 точек (-3 до 3), в качестве параметра t
line_x = mean_X1[0] + v1[0] * t
line_y = mean_X1[1] + v1[1] * t
plt.plot(line_x, line_y, color='red', linestyle='--', label='Первая главная компонента')

#восстановление координат проекций в 2D для отрисовки
X_reconstructed = p1 @ e_vecs1[:, :1].T + mean_X1
plt.scatter(X_reconstructed[:, 0], X_reconstructed[:, 1], color='green', marker='X', s=50, label='Проекции')

plt.title('Геометрический взгляд на PCA')
plt.xlabel('X1')
plt.ylabel('X2')
plt.axis('equal')
plt.grid(True)
plt.legend()
plt.show()

print("\nЗадание 4: снижение размерности 3D -> 2D")
np.random.seed(42)
X4 = np.array([
    [2.5, 2.4, 1.1],
    [0.5, 0.7, 0.2],
    [2.2, 2.9, 0.9],
    [1.9, 2.2, 0.8],
    [3.1, 3.0, 1.3]
])

mean_X4, p4, e_vecs4, e_vals4, expl_ratio4 = perform_pca(X4, n_components=2)

print("Исходные 3D данные:")
print(X4)
print("\nСпроецированные данные в 2D:")
print(p4)

#анализ потери информации
total_info = np.sum(e_vals4)
retained_info = np.sum(e_vals4[:2])
lost_info = e_vals4[2]                  #

print(f"\nОбщая дисперсия исходных данных: {total_info:.4f}")
print(f"Сохраненная дисперсия (в 2D): {retained_info:.4f} (или {retained_info/total_info*100:.2f}%)")
print(f"Потерянная дисперсия (при отбросе 3-й оси): {lost_info:.4f} (или {lost_info/total_info*100:.2f}%)")

print(f"\nВывод: При снижении размерности с 3 до 2 мы теряем {lost_info/total_info*100:.2f}% информации.")