import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report

# 1. Generate a synthetic non-linearly separable dataset (moons dataset)
X, y = datasets.make_moons(n_samples=200, noise=0.2, random_state=42)

# 2. Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 3. Standardize the features (important for SVMs, especially with RBF kernel)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 4. Initialize and train the Kernel SVC model (using RBF kernel)
# You can experiment with different kernels and hyperparameters
# For example:
# svc_rbf = SVC(kernel='rbf', C=1.0, gamma='scale', random_state=42) # 'scale' uses 1 / (n_features * X.var())
# svc_poly = SVC(kernel='poly', degree=3, C=1.0, random_state=42)
# svc_sigmoid = SVC(kernel='sigmoid', C=1.0, random_state=42)

svc_rbf = SVC(kernel='rbf', C=1.0, gamma=0.5, random_state=42)
svc_rbf.fit(X_train_scaled, y_train)

# 5. Make predictions on the test set
y_pred = svc_rbf.predict(X_test_scaled)

# 6. Evaluate the model
print("--- Kernel SVC (RBF Kernel) ---")
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# 7. Visualize the decision boundary (optional but highly recommended for understanding)

def plot_decision_boundary(X, y, model, title):
    h = .02  # step size in the mesh
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))

    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    plt.contourf(xx, yy, Z, cmap=plt.cm.coolwarm, alpha=0.8)
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.coolwarm, s=50, edgecolors='k')
    plt.title(title)
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.show()

# Plot on the original (unscaled) data for better visual intuition,
# but remember the model was trained on scaled data.
# We'll need to inverse transform the meshgrid points if we want to predict on them with the scaler.
# A simpler approach for visualization is to plot the decision boundary on the scaled data and then plot the original data points.

# To plot the decision boundary on the original scale, we need to transform the meshgrid
# points *before* passing them to the model's predict method.
# For simplicity, let's plot the decision boundary on the scaled data and then scatter the original points.

# Re-train a model for visualization on the original data (for conceptual clarity of boundaries)
# This is a bit of a hack for plotting, in real use cases, you'd plot based on the scaled model's decision function.
# For a more robust visualization of the decision boundary when using StandardScaler, you would typically:
# 1. Train the model on scaled data.
# 2. Create a meshgrid for the *scaled* feature space.
# 3. Make predictions on this scaled meshgrid.
# 4. Plot the contours on the scaled meshgrid.
# 5. Overlay the original data points.

# Let's create a cleaner visualization function for scaled data
def plot_scaled_decision_boundary(X_scaled, y, model, scaler, title):
    # Create a meshgrid of points that covers the scaled feature space
    x_min, x_max = X_scaled[:, 0].min() - 1, X_scaled[:, 0].max() + 1
    y_min, y_max = X_scaled[:, 1].min() - 1, X_scaled[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),
                         np.arange(y_min, y_max, 0.02))

    # Predict on the scaled meshgrid points
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    plt.contourf(xx, yy, Z, cmap=plt.cm.coolwarm, alpha=0.8)
    plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=y, cmap=plt.cm.coolwarm, s=50, edgecolors='k')
    plt.title(title)
    plt.xlabel("Scaled Feature 1")
    plt.ylabel("Scaled Feature 2")
    plt.show()

plot_scaled_decision_boundary(X_train_scaled, y_train, svc_rbf, scaler, "Kernel SVC (RBF) Decision Boundary on Scaled Training Data")
plot_scaled_decision_boundary(X_test_scaled, y_test, svc_rbf, scaler, "Kernel SVC (RBF) Decision Boundary on Scaled Test Data")