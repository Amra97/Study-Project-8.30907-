import pandas as pd

# ---------- 1) Load data ----------
unity_path = r"C:\Users\ameer\Downloads\UnityObjectsSizes.csv"
eyetrack_path = r"C:\Users\ameer\Downloads\combined_dataframe (1).csv"


et = pd.read_csv(eyetrack_path)
unity = pd.read_csv(unity_path)

# Remove duplicated columns in ET if present
et = et.loc[:, ~et.columns.duplicated()]

# Clean column names (trim spaces)
et.columns = et.columns.str.strip()
unity.columns = unity.columns.str.strip()

# ---------- 2) Define key column names ----------
ET_NAME_COL = "Interpolated_collider"   # object names in ET
ET_COLLIDER_COL = "hitColliderType"     # collider type in ET
UNITY_NAME_COL = "GameObject"           # object names in Unity
UNITY_COLLIDER_COL = "ColliderType"     # collider type in Unity

# Sanity check
for col, dfname, df in [
    (ET_NAME_COL, "eye-tracking file", et),
    (ET_COLLIDER_COL, "eye-tracking file", et),
    (UNITY_NAME_COL, "Unity file", unity),
    (UNITY_COLLIDER_COL, "Unity file", unity),
]:
    if col not in df.columns:
        raise ValueError(f"Column '{col}' not found in {dfname}. Available: {list(df.columns)}")

# ---------- 3) Normalise object names for matching ----------
et["obj_name_norm"] = (
    et[ET_NAME_COL]
    .astype(str)
    .str.strip()
    .str.lower()
)

unity["obj_name_norm"] = (
    unity[UNITY_NAME_COL]
    .astype(str)
    .str.strip()
    .str.lower()
)

# ---------- 4) Normalise collider-type labels ----------
# ET: "UnityEngine.MeshCollider" -> "meshcollider", etc.
et["collider_type_norm"] = (
    et[ET_COLLIDER_COL]
    .astype(str)
    .str.replace("UnityEngine.", "", regex=False)
    .str.strip()
    .str.lower()
)

# Unity: "MeshCollider" / "BoxCollider" -> same format
unity["collider_type_norm"] = (
    unity[UNITY_COLLIDER_COL]
    .astype(str)
    .str.strip()
    .str.lower()
)

# ---------- 5) One collider type per object in ET ----------
# (use the most frequent collider type per object, in case of noise)
et_obj_colliders = (
    et.dropna(subset=["obj_name_norm"])
      .groupby("obj_name_norm")["collider_type_norm"]
      .agg(lambda x: x.mode().iat[0])   # most common collider type for that object
      .reset_index(name="et_collider_type")
)

# Unity collider per object
unity_obj_colliders = (
    unity.dropna(subset=["obj_name_norm"])[["obj_name_norm", "collider_type_norm"]]
    .rename(columns={"collider_type_norm": "unity_collider_type"})
)

# ---------- 6) Merge and compare ----------
merged = unity_obj_colliders.merge(
    et_obj_colliders,
    on="obj_name_norm",
    how="left"      # keep all Unity objects
)

# True if same collider type in both datasets
merged["collider_matches"] = merged["unity_collider_type"] == merged["et_collider_type"]

# ---------- 7) Summary statistics ----------
total_objects = len(merged)
n_with_et_info = merged["et_collider_type"].notna().sum()
n_matches = merged["collider_matches"].sum()
n_mismatches = (merged["collider_matches"] == False).sum()  # includes cases where ET type exists but differs

print(f"Total Unity objects: {total_objects}")
print(f"Objects that appear in ET data: {n_with_et_info}")
print(f"Objects with matching collider type: {n_matches}")
print(f"Objects with different collider type: {n_mismatches}")

# ---------- 8) Inspect mismatches and save ----------
mismatches = merged[(merged["et_collider_type"].notna()) & (~merged["collider_matches"])]

print("\nExample mismatches:")
print(
    mismatches[["obj_name_norm", "unity_collider_type", "et_collider_type"]]
    .head(20)
)

# Save full comparison tables if you like
merged.to_csv("objects_collider_comparison.csv", index=False)
mismatches.to_csv("objects_collider_mismatches.csv", index=False)