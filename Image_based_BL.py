import numpy as np
import pandas as pd
import kde_utils
from scipy.stats import gaussian_kde
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix


def load_data(data_path, rock_grade_path):
    data0 = pd.read_excel(data_path, header=None)
    data = np.asarray(data0.iloc[:, :], dtype=np.float64)

    rock_grade0 = pd.read_excel(rock_grade_path, header=None)
    rock_grade = rock_grade0.iloc[:, :].values

    return data, rock_grade


def split_features_and_labels(data):
    geo_features = data[:, [3, 4]]
    tbm_features = data[:, [5, 6, 7]]
    y_true = data[:, 2].astype(int)
    chainage = data[:, 1]

    return geo_features, tbm_features, y_true, chainage


def generate_image_representation(geo_features, y_true, tbm_features):
    geo_part = geo_features.T
    rock_part = y_true.reshape(1, -1).astype(np.float64)
    tbm_part = tbm_features.T

    X_image = np.vstack([geo_part, rock_part, tbm_part])
    return X_image


def standardize_features(X_image, current_step, geo_row_num=2, rock_row_num=1):
    geo_rows = slice(0, geo_row_num)
    tbm_rows = slice(geo_row_num + rock_row_num, X_image.shape[0])

    X_geo_all = X_image[geo_rows, :]
    geo_mean = np.mean(X_geo_all, axis=1, keepdims=True)
    geo_std = np.std(X_geo_all, axis=1, keepdims=True) + 1e-12
    X_geo_std = (X_geo_all - geo_mean) / geo_std

    X_tbm_hist = X_image[tbm_rows, :current_step + 1]
    tbm_mean = np.mean(X_tbm_hist, axis=1, keepdims=True)
    tbm_std = np.std(X_tbm_hist, axis=1, keepdims=True) + 1e-12

    X_tbm_all = X_image[tbm_rows, :]
    X_tbm_std = (X_tbm_all - tbm_mean) / tbm_std

    return X_geo_std, X_tbm_std


def extract_training_and_test_templates(
    X_image,
    X_geo_std,
    X_tbm_std,
    y_true,
    current_step,
    horizon,
    geo_row_num=2,
    rock_row_index=2
):
    X_train_list = []
    y_train_list = []

    for t in range(0, current_step - horizon + 1):
        tbm_vec = X_tbm_std[:, t]
        geo_vec = X_geo_std[:, t + horizon]
        x_vec = np.concatenate([tbm_vec, geo_vec], axis=0)

        X_train_list.append(x_vec)
        y_train_list.append(y_true[t + horizon])

    X_train = np.asarray(X_train_list, dtype=np.float64)
    y_train = np.asarray(y_train_list, dtype=int)

    X_test_tbm = X_tbm_std[:, current_step]
    X_test_geo = X_geo_std[:, current_step + horizon]
    X_test = np.concatenate([X_test_tbm, X_test_geo], axis=0).reshape(1, -1)

    y_test = int(y_true[current_step + horizon])

    return X_train, y_train, X_test, y_test


def fit_classwise_kde_models(
    X_train,
    y_train,
    classes=(2, 3, 4, 5),
    correlation_length=14
):
    kde_models = {}

    for cls in classes:
        X_cls = X_train[y_train == cls]

        if len(X_cls) < 2:
            kde_models[cls] = None
        else:
            optimal_bw = kde_utils.compute_kde_and_optimal_bandwidth(
                X_cls,
                correlation_length
            )
            kde_models[cls] = gaussian_kde(X_cls.T, bw_method=optimal_bw)

    return kde_models


def compute_class_conditional_probabilities(
    kde_models,
    X_test,
    classes=(2, 3, 4, 5)
):
    conditional_probs = []

    for cls in classes:
        kde_model = kde_models.get(cls, None)

        if kde_model is None:
            conditional_probs.append(1e-12)
        else:
            prob = float(kde_model(X_test.T)[0])
            conditional_probs.append(max(prob, 1e-12))

    return np.array(conditional_probs, dtype=np.float64)


def compute_prior_probabilities(
    current_step,
    chainage,
    rock_grade,
    correlation_length=14,
    classes=(2, 3, 4, 5)
):
    stake = chainage[current_step]
    stake_ahead = stake + correlation_length

    grade_lengths = {cls: 0.0 for cls in classes}
    stake_i = None
    stake_ahead_i = None

    for i, interval in enumerate(rock_grade):
        start, end, grade = interval
        grade = int(grade)

        if start <= stake <= end and start <= stake_ahead <= end:
            grade_lengths[grade] += correlation_length
            stake_i = i
            stake_ahead_i = i
            break
        elif start <= stake <= end:
            grade_lengths[grade] += end - stake
            stake_i = i
        elif start <= stake_ahead <= end:
            grade_lengths[grade] += stake_ahead - start
            stake_ahead_i = i
            break

    if (
        stake_i is not None
        and stake_ahead_i is not None
        and stake_ahead_i - stake_i > 1
    ):
        for j in range(stake_i + 1, stake_ahead_i):
            grade = int(rock_grade[j, 2])
            grade_lengths[grade] += rock_grade[j, 1] - rock_grade[j, 0]

    priors = []
    for cls in classes:
        priors.append((grade_lengths[cls] + 1.0) / (correlation_length + len(classes)))

    return np.array(priors, dtype=np.float64), stake


def predict_one_horizon(
    X_image,
    y_true,
    chainage,
    rock_grade,
    current_step,
    horizon,
    correlation_length=14,
    classes=(2, 3, 4, 5)
):
    X_geo_std, X_tbm_std = standardize_features(
        X_image=X_image,
        current_step=current_step,
        geo_row_num=2,
        rock_row_num=1
    )

    X_train, y_train, X_test, y_test = extract_training_and_test_templates(
        X_image=X_image,
        X_geo_std=X_geo_std,
        X_tbm_std=X_tbm_std,
        y_true=y_true,
        current_step=current_step,
        horizon=horizon,
        geo_row_num=2,
        rock_row_index=2
    )

    kde_models = fit_classwise_kde_models(
        X_train=X_train,
        y_train=y_train,
        classes=classes,
        correlation_length=correlation_length
    )

    conditional_probs = compute_class_conditional_probabilities(
        kde_models=kde_models,
        X_test=X_test,
        classes=classes
    )

    prior_probs, stake = compute_prior_probabilities(
        current_step=current_step,
        chainage=chainage,
        rock_grade=rock_grade,
        correlation_length=correlation_length,
        classes=classes
    )

    posterior_unnormalized = conditional_probs * prior_probs
    evidence = np.sum(posterior_unnormalized)

    if evidence <= 0 or np.isnan(evidence):
        posterior_probs = np.ones(len(classes), dtype=np.float64) / len(classes)
    else:
        posterior_probs = posterior_unnormalized / evidence

    y_pred = classes[int(np.argmax(posterior_probs))]

    return {
        "current_step": current_step,
        "horizon": horizon,
        "stake": stake,
        "y_true": y_test,
        "y_pred": y_pred,
        "evidence": evidence,
        "posterior_probs": posterior_probs,
        "conditional_probs": conditional_probs,
        "prior_probs": prior_probs
    }


def predict_next_10_steps(
    X_image,
    y_true,
    chainage,
    rock_grade,
    current_step,
    correlation_length=14,
    classes=(2, 3, 4, 5)
):
    results = []

    for horizon in range(1, 11):
        result = predict_one_horizon(
            X_image=X_image,
            y_true=y_true,
            chainage=chainage,
            rock_grade=rock_grade,
            current_step=current_step,
            horizon=horizon,
            correlation_length=correlation_length,
            classes=classes
        )
        results.append(result)

    return results


def summarize_predictions(results):
    rows = []

    for res in results:
        row = {
            "current_step": res["current_step"],
            "horizon": res["horizon"],
            "stake": res["stake"],
            "y_true": res["y_true"],
            "y_pred": res["y_pred"],
            "evidence": res["evidence"]
        }

        for i, cls in enumerate([2, 3, 4, 5]):
            row[f"posterior_{cls}"] = res["posterior_probs"][i]
            row[f"conditional_{cls}"] = res["conditional_probs"][i]
            row[f"prior_{cls}"] = res["prior_probs"][i]

        rows.append(row)

    return pd.DataFrame(rows)


def evaluate_10_step_prediction(results):
    y_true = np.array([res["y_true"] for res in results], dtype=int)
    y_pred = np.array([res["y_pred"] for res in results], dtype=int)

    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average="macro")
    cm = confusion_matrix(y_true, y_pred, labels=[2, 3, 4, 5])

    return {
        "accuracy": acc,
        "f1_macro": f1,
        "confusion_matrix": cm
    }


def main():
    data_path = "Yinsong data.xlsx"
    rock_grade_path = "Rock grade.xlsx"

    current_step = 4000
    correlation_length = 14

    data, rock_grade = load_data(data_path, rock_grade_path)

    geo_features, tbm_features, y_true, chainage = split_features_and_labels(data)

    X_image = generate_image_representation(
        geo_features=geo_features,
        y_true=y_true,
        tbm_features=tbm_features
    )

    results = predict_next_10_steps(
        X_image=X_image,
        y_true=y_true,
        chainage=chainage,
        rock_grade=rock_grade,
        current_step=current_step,
        correlation_length=correlation_length
    )

    results_df = summarize_predictions(results)
    metrics = evaluate_10_step_prediction(results)

    print("\nPrediction results for the next 10 steps:")
    print(results_df)

    print("\nSummary metrics over horizons 1-10:")
    print(f"Accuracy : {metrics['accuracy']:.4f}")
    print(f"F1-score : {metrics['f1_macro']:.4f}")
    print("Confusion matrix (labels = [2, 3, 4, 5]):")
    print(metrics["confusion_matrix"])

    return results_df, metrics


if __name__ == "__main__":
    main()