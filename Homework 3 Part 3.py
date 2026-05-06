#Homework 3 Part 3
from sklearn.feature_selection import SequentialFeatureSelector
import numpy as np

rkf_search = RepeatedKFold(n_splits=10, n_repeats=5, random_state=42)
feature_names = x_final.columns.tolist()

best_subsets = {}

print(f"{'Algorithm':<20} | {'Best Features Found'}")
print("-" * 80)

for name, model in models.items():
    sfs = SequentialFeatureSelector(
        estimator=model,
        n_features_to_select='auto', 
        direction='forward',
        scoring='accuracy',
        cv=rkf_search,
        n_jobs=-1
    )
    
    sfs.fit(x_final, y_final)
    
    selected_mask = sfs.get_support()
    selected_features = np.array(feature_names)[selected_mask].tolist()
    best_subsets[name] = selected_features
    
    print(f"{name:<20} | {len(selected_features)} features: {selected_features}")

print("\n--- Final Performance with Optimized Subsets ---")
print(f"{'Algorithm':<20} | {'Mean Acc':<10} | {'Std Dev':<10}")
print("-" * 50)

rkf_final = RepeatedKFold(n_splits=10, n_repeats=100, random_state=42)

for name, model in models.items():
    features = best_subsets[name]
    x_subset = x_final[features]
    
    scores = cross_val_score(model, x_subset, y_final, cv=rkf_final, scoring='accuracy', n_jobs=-1)
    
    print(f"{name:<20} | {scores.mean():.4f}    | {scores.std():.4f}")