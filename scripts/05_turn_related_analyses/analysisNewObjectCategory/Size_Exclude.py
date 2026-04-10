import pandas as pd
import numpy as np

"""
This script:

1. Loads your merged eye-tracking + Unity dataset.
2. Computes retinal image size (visual angle) for each fixation on a valid object.
3. Applies an HTC Vive Pro Eye field-of-view (FOV) filter:
   - Excludes objects whose retinal image would not fit into the headset FOV.
4. Saves the final cleaned dataset.

ASSUMPTIONS:
- Input CSV file: "eyetracking_with_unity_and_exclude_object.csv"
- Columns present:
    - exclude_object      : 1.0 = fixation on valid object; 0.0 = saccade or excluded
    - Eucledian_distance  : distance from eye to hit point (meters)
    - SizeX               : horizontal object size (meters)
    - SizeY               : vertical object size (meters)
    - Interpolated_collider (optional, just for reference in prints)
"""

# -------------------- 1. Load data --------------------
input_path = r"C:\Users\ameer\Downloads\combined_dataframe_obj.csv"
df = pd.read_csv(input_path)

# Make sure exclude_object is numeric (0.0 / 1.0)
df["exclude_object"] = df["exclude_object"].astype(float)

# -------------------- 2. Compute retinal image size (visual angle) --------------------
# We only compute visual angle where:
# - exclude_object == 1.0 (valid object fixation)
# - distance is present and > 0
mask = (
    (df["exclude_object"] == 1.0) &
    df["Eucledian_distance"].notna() &
    (df["Eucledian_distance"] > 0)
)

# Horizontal visual angle (deg) from SizeX
df.loc[mask, "visual_angle_width_deg"] = 2 * np.degrees(
    np.arctan2(
        df.loc[mask, "SizeX"] / 2.0,
        df.loc[mask, "Eucledian_distance"]
    )
)

# Vertical visual angle (deg) from SizeY
df.loc[mask, "visual_angle_height_deg"] = 2 * np.degrees(
    np.arctan2(
        df.loc[mask, "SizeY"] / 2.0,
        df.loc[mask, "Eucledian_distance"]
    )
)

# Optional: approximate retinal image area in deg^2
df.loc[mask, "retinal_image_area_deg2"] = (
    df.loc[mask, "visual_angle_width_deg"] *
    df.loc[mask, "visual_angle_height_deg"]
)

print("Example rows with computed visual angles:")
print(
    df.loc[mask, [
        "Interpolated_collider",
        "SizeX", "SizeY", "Eucledian_distance",
        "visual_angle_width_deg", "visual_angle_height_deg"
    ]]
    .head()
)

# -------------------- 3. Apply HTC Vive Pro Eye FOV filter --------------------
# Vive Pro Eye: ~110° diagonal FOV. We approximate:
H_FOV_DEG = 100.0  # approx horizontal FOV
V_FOV_DEG = 100.0  # approx vertical FOV

# Rows where we have valid visual angle and a valid object (exclude_object == 1.0)
mask_valid_angle = (
    (df["exclude_object"] == 1.0) &
    df["visual_angle_width_deg"].notna() &
    df["visual_angle_height_deg"].notna()
)

# Does the object fit inside the (approximate) HMD FOV?
df.loc[mask_valid_angle, "fits_horiz_fov"] = (
    df.loc[mask_valid_angle, "visual_angle_width_deg"] <= H_FOV_DEG
)
df.loc[mask_valid_angle, "fits_vert_fov"] = (
    df.loc[mask_valid_angle, "visual_angle_height_deg"] <= V_FOV_DEG
)
df.loc[mask_valid_angle, "fits_full_fov"] = (
    df.loc[mask_valid_angle, "fits_horiz_fov"] &
    df.loc[mask_valid_angle, "fits_vert_fov"]
)

# Exclude objects that do NOT fit the FOV:
# set exclude_object = 0.0 for them
df.loc[
    mask_valid_angle & (~df["fits_full_fov"].fillna(False)),
    "exclude_object"
] = 0.0

# -------------------- 4. Report and save --------------------
total_rows = len(df)
n_valid_before = int(mask.sum())
n_valid_after = int((df["exclude_object"] == 1.0).sum())

print("\n--- Summary ---")
print(f"Total rows in dataset: {total_rows}")
print(f"Rows with valid objects and distance (before FOV filter): {n_valid_before}")
print(f"Rows with exclude_object == 1.0 after FOV filter: {n_valid_after}")

output_path = "eyetracking_with_visual_angle_and_vive_fov_filter.csv"
df.to_csv(output_path, index=False)
print(f"\nSaved updated dataset with visual angles and FOV filter to:\n{output_path}")