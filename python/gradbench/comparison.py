import numpy as np


# This definition is ported from ADBench's JacobianComparison.cs.
def difference(x, y):
    absX = np.abs(x)
    absY = np.abs(y)
    absdiff = np.abs(x - y)
    normCoef = np.clip(absX + absY, a_min=1, a_max=None)
    return absdiff / normCoef


def compare_json_objects(expected, actual, tolerance=1e-4, path=""):
    """Compare two Python objects corresponding to JSON objects and
    report mismatches.

    :param expected: The reference value.

    :param actual: The obtained value.

    :param path: Path within the nested structure (used for reporting mismatches)

    :return: List of mismatches, each as a string describing the
    location and nature of the mismatch

    """
    mismatches = []

    if isinstance(expected, dict) and isinstance(actual, dict):
        all_keys = set(expected.keys()).union(set(actual.keys()))
        for key in all_keys:
            new_path = f"{path}.{key}" if path else key
            if key not in expected:
                mismatches.append(f"Unexpected key '{new_path}'.")
            elif key not in actual:
                mismatches.append(f"Key '{new_path}' is missing.")
            else:
                mismatches.extend(
                    compare_json_objects(
                        expected[key], actual[key], tolerance, new_path
                    )
                )

    elif isinstance(expected, list) and isinstance(actual, list):
        expected_len = len(expected)
        actual_len = len(actual)
        if expected_len != actual_len:
            mismatches.append(
                f"{path}: Expected array of length {expected_len}, got array of length {actual_len}."
            )
        for i in range(expected_len):
            mismatches.extend(
                compare_json_objects(expected[i], actual[i], tolerance, f"{path}[{i}]")
            )

    elif isinstance(expected, float) and isinstance(actual, float):
        if difference(np.array(actual), np.array(expected)) > tolerance:
            mismatches.append(f"{path}: {expected} != {actual}")

    else:
        if expected != actual:
            mismatches.append(f"{path}: {expected} != {actual}")

    return mismatches
