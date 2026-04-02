# compose_go2_z1.py
"""
compose_go2_z1.py — Definitive composite USD for Go2 + Z1.

Root cause of all previous failures, now resolved:
  PhysX discovers articulation bodies exclusively by following joint body0/body1
  relationships. root_joint (body0=base, body1=link00) is the REQUIRED edge that
  lets PhysX enter the Z1 chain. Without it, no Z1 joint is ever found.

  Every previous attempt either:
    (a) Deleted root_joint, severing the traversal edge, or
    (b) Set body0 via Usd.Stage.SetTargets() which writes to the USD session layer
        and is NEVER captured by UsdUtils.FlattenLayerStack.

  This script fixes body0 by editing the SdfRelationshipSpec directly on the
  flattened SdfLayer — the only API guaranteed to survive Save().

Strategy:
  1. Flatten z1.usd fully (eliminates sublayers). Export to z1_clean.usd.
  2. Edit z1_clean.usd as a raw SdfLayer: delete OmniGraph specs only.
     root_joint is deliberately KEPT.
  3. Compose Go2 + z1_clean in memory. Strip duplicate ArticulationRootAPI.
  4. Apply RigidBodyAPI to Z1 links on composed stage.
  5. Flatten to go2_z1.usd.
  6. Re-open go2_z1.usd as a raw SdfLayer. Set root_joint body0 directly
     via SdfRelationshipSpec.targetPathList — bypasses session layer entirely.
  7. Validate from a fresh Usd.Stage.Open.

Run:
    ./isaaclab.sh -p isaaclab_master/compose_go2_z1.py
"""

from isaaclab.app import AppLauncher
import argparse
parser = argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
app_launcher = AppLauncher(args_cli)

from pxr import Usd, UsdGeom, UsdPhysics, UsdUtils, Gf, Sdf
try:
    from pxr import PhysxSchema
    _HAS_PHYSX = True
except ImportError:
    _HAS_PHYSX = False
    print("WARNING: PhysxSchema unavailable.")

import os

# =========================================================================
# Paths
# =========================================================================
GO2_USD_PATH    = "/workspace/isaaclab/isaaclab_master/Go2.usd"
Z1_USD_PATH     = "/workspace/isaaclab/isaaclab_master/z1.usd"
Z1_CLEAN_PATH   = "/workspace/isaaclab/isaaclab_master/z1_clean.usd"
OUTPUT_USD_PATH = "/workspace/isaaclab/isaaclab_master/go2_z1.usd"
Z1_PRIM_PATH    = "/Robot/base/Z1"
Z1_LINK_NAMES   = ["link00","link01","link02","link03","link04","link05","link06"]

joint_types = {
    "PhysicsRevoluteJoint","PhysicsPrismaticJoint",
    "PhysicsSphericalJoint","PhysicsFixedJoint",
}
omni_types = {"OmniGraph","OmniGraphNode"}

print(f"Go2 source : {GO2_USD_PATH}")
print(f"Z1 source  : {Z1_USD_PATH}")
print(f"Output     : {OUTPUT_USD_PATH}\n")

for label, path in [("Go2", GO2_USD_PATH), ("Z1", Z1_USD_PATH)]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"{label} USD not found: {path}")


# =========================================================================
# Helpers
# =========================================================================

def _walk_sdf(spec, visitor):
    """Depth-first walk of SdfPrimSpec tree."""
    visitor(spec)
    for child in list(spec.nameChildren.values()):
        _walk_sdf(child, visitor)


def _set_rel_on_layer(layer, prim_path_str, rel_name, target_path_strs):
    """
    Set relationship targets directly on an SdfRelationshipSpec.

    This is the only method guaranteed to survive Sdf.Layer.Save() and a
    subsequent Usd.Stage.Open(). Any approach that goes through a Usd.Stage
    (SetTargets, SetEditTarget, etc.) risks writing to the session layer,
    which is transient and never persisted to disk.
    """
    spec = layer.GetPrimAtPath(Sdf.Path(prim_path_str))
    if spec is None:
        raise RuntimeError(f"Prim spec not found: '{prim_path_str}'")
    rel_spec = spec.relationships.get(rel_name)
    if rel_spec is None:
        rel_spec = Sdf.RelationshipSpec(spec, rel_name, custom=False)
    rel_spec.targetPathList.explicitItems = [Sdf.Path(p) for p in target_path_strs]


def _reload_layer(path):
    """Force-reload a cached SdfLayer and return the fresh instance."""
    cached = Sdf.Layer.Find(path)
    if cached:
        cached.Reload(force=True)
    layer = Sdf.Layer.FindOrOpen(path)
    if layer is None:
        raise RuntimeError(f"Cannot open SdfLayer: {path}")
    return layer


# =========================================================================
# Step 1 — Flatten z1.usd and strip OmniGraph specs
# =========================================================================
print("=== Step 1: Producing z1_clean.usd ===")

# Flatten all sublayers of z1.usd into a single file.
# This eliminates the sublayer resurrection problem: a spec deleted from
# the primary layer is silently re-introduced from a surviving sublayer at
# Usd.Stage.Open time.
z1_stage_src = Usd.Stage.Open(Z1_USD_PATH)
flat_z1 = UsdUtils.FlattenLayerStack(z1_stage_src)
flat_z1.Export(Z1_CLEAN_PATH)
print(f"  Exported flattened z1 → {Z1_CLEAN_PATH}")

clean_layer = _reload_layer(Z1_CLEAN_PATH)

if clean_layer.subLayerPaths:
    raise RuntimeError(
        f"z1_clean.usd still has sublayers: {list(clean_layer.subLayerPaths)}"
    )
print(f"  Confirmed: 0 sublayers in z1_clean.usd")

# Delete OmniGraph specs. root_joint is deliberately kept.
omni_paths = []
def _collect_omni(spec):
    if hasattr(spec, "typeName") and spec.typeName in omni_types:
        omni_paths.append(spec.path)
_walk_sdf(clean_layer.pseudoRoot, _collect_omni)
omni_paths = sorted(omni_paths, key=lambda p: str(p).count("/"), reverse=True)

deleted = 0
for sdf_path in omni_paths:
    parent = clean_layer.GetPrimAtPath(sdf_path.GetParentPath())
    if parent and sdf_path.name in parent.nameChildren:
        del parent.nameChildren[sdf_path.name]
        print(f"  Deleted: {sdf_path}")
        deleted += 1

clean_layer.Save()
print(f"  Deleted {deleted} OmniGraph spec(s).")

# Verify root_joint is present (required for PhysX chain traversal).
rj_in_clean = clean_layer.GetPrimAtPath(Sdf.Path("/World/z1/root_joint"))
if rj_in_clean is None:
    raise RuntimeError(
        "root_joint NOT found in z1_clean.usd after flattening. "
        "It must be present so PhysX can traverse from /Robot/base into the Z1 chain."
    )
print(f"  root_joint confirmed present (typeName='{rj_in_clean.typeName}'). Step 1 OK.\n")


# =========================================================================
# Step 2 — Inspect Z1 joint inventory
# =========================================================================
print("=== Step 2: Z1 joint inventory ===")
verify_clean = Usd.Stage.Open(Z1_CLEAN_PATH)
z1_joints = sorted([str(p.GetPath()) for p in verify_clean.Traverse()
                    if p.GetTypeName() in joint_types])
z1_revolute = [j for j in z1_joints
               if verify_clean.GetPrimAtPath(j).GetTypeName() == "PhysicsRevoluteJoint"]

for j in z1_joints:
    t = verify_clean.GetPrimAtPath(j).GetTypeName()
    print(f"  [{t:28s}] {j}")

if len(z1_revolute) != 6:
    raise RuntimeError(f"Expected 6 revolute Z1 joints, found {len(z1_revolute)}")

# Determine Z1 first link name from root_joint body1 target.
z1_rj_prim = next(
    (verify_clean.GetPrimAtPath(j) for j in z1_joints if "root_joint" in j),
    None
)
if z1_rj_prim is None:
    raise RuntimeError("root_joint not found in z1_clean.usd stage traversal.")
z1_body1_targets = UsdPhysics.Joint(z1_rj_prim).GetBody1Rel().GetTargets()
z1_first_link = str(z1_body1_targets[0]).split("/")[-1] if z1_body1_targets else "link00"
print(f"\n  First link (root_joint body1): {z1_first_link}. Step 2 OK.\n")


# =========================================================================
# Step 3 — Compose Go2 + z1_clean in memory
# =========================================================================
print("=== Step 3: Composing Go2 + z1_clean ===")
composed = Usd.Stage.CreateInMemory()
root_prim = composed.DefinePrim("/Robot", "Xform")
root_prim.GetReferences().AddReference(GO2_USD_PATH)
composed.SetDefaultPrim(root_prim)

z1_ref_prim = composed.DefinePrim(Z1_PRIM_PATH, "Xform")
z1_ref_prim.GetReferences().AddReference(Z1_CLEAN_PATH)
UsdGeom.Xformable(z1_ref_prim).AddTranslateOp().Set(Gf.Vec3f(0.05, 0.0, 0.05))

# Strip duplicate ArticulationRootAPI from Z1 subtree.
removed_apis = 0
for prim in composed.Traverse():
    if not str(prim.GetPath()).startswith(Z1_PRIM_PATH):
        continue
    if UsdPhysics.ArticulationRootAPI(prim):
        prim.RemoveAPI(UsdPhysics.ArticulationRootAPI)
        removed_apis += 1
    if _HAS_PHYSX and PhysxSchema.PhysxArticulationAPI(prim):
        prim.RemoveAPI(PhysxSchema.PhysxArticulationAPI)
        removed_apis += 1
print(f"  Removed {removed_apis} duplicate articulation API(s).")

# Apply RigidBodyAPI and MassAPI to Z1 links.
for link_name in Z1_LINK_NAMES:
    lp = composed.GetPrimAtPath(f"{Z1_PRIM_PATH}/z1/{link_name}")
    if not lp.IsValid():
        print(f"  WARNING: {link_name} not found")
        continue
    if not UsdPhysics.RigidBodyAPI(lp):
        UsdPhysics.RigidBodyAPI.Apply(lp)
    if not UsdPhysics.MassAPI(lp):
        UsdPhysics.MassAPI.Apply(lp).GetMassAttr().Set(0.5)

composed_joints = sorted([str(p.GetPath()) for p in composed.Traverse()
                          if p.GetTypeName() in joint_types])
composed_revolute = [j for j in composed_joints
                     if composed.GetPrimAtPath(j).GetTypeName() == "PhysicsRevoluteJoint"]
print(f"  Composed stage: {len(composed_joints)} joints, {len(composed_revolute)} revolute")

if len(composed_revolute) != 18:
    raise RuntimeError(f"Expected 18 revolute joints in composed stage, got {len(composed_revolute)}")
print("  Step 3 OK.\n")


# =========================================================================
# Step 4 — Flatten composed stage to disk
# =========================================================================
print(f"=== Step 4: Flattening → {OUTPUT_USD_PATH} ===")
flat_layer = UsdUtils.FlattenLayerStack(composed)
flat_layer.Export(OUTPUT_USD_PATH)
print("  Exported. Step 4 OK.\n")


# =========================================================================
# Step 5 — Set root_joint body0 directly on the SdfLayer spec
# =========================================================================
# This is the core fix. All previous iterations used Usd.Stage.SetTargets(),
# which writes to the session layer (a transient in-memory layer that is
# never persisted by Save()). The SdfRelationshipSpec API operates directly
# on the layer's data model with no stage involvement — it is the only
# method guaranteed to produce a relationship target that survives to disk.
print("=== Step 5: Setting root_joint body0 via SdfRelationshipSpec ===")

out_layer = _reload_layer(OUTPUT_USD_PATH)

RJ_PATH    = "/Robot/base/Z1/z1/root_joint"
BODY0      = "/Robot/base"
BODY1      = f"/Robot/base/Z1/z1/{z1_first_link}"

rj_spec = out_layer.GetPrimAtPath(Sdf.Path(RJ_PATH))
if rj_spec is None:
    raise RuntimeError(
        f"root_joint spec not found at '{RJ_PATH}' in flattened USD. "
        "UsdUtils.FlattenLayerStack did not inline the Z1 content."
    )
print(f"  root_joint spec found (typeName='{rj_spec.typeName}').")

_set_rel_on_layer(out_layer, RJ_PATH, "physics:body0", [BODY0])
_set_rel_on_layer(out_layer, RJ_PATH, "physics:body1", [BODY1])

# Verify on the SdfLayer directly (not via a Stage) before saving.
b0_spec = rj_spec.relationships.get("physics:body0")
b1_spec = rj_spec.relationships.get("physics:body1")
b0_items = [str(p) for p in (b0_spec.targetPathList.explicitItems if b0_spec else [])]
b1_items = [str(p) for p in (b1_spec.targetPathList.explicitItems if b1_spec else [])]

print(f"  SdfLayer body0 explicit targets: {b0_items}")
print(f"  SdfLayer body1 explicit targets: {b1_items}")

if b0_items != [BODY0]:
    raise RuntimeError(f"body0 not set correctly. Got: {b0_items}")
if b1_items != [BODY1]:
    raise RuntimeError(f"body1 not set correctly. Got: {b1_items}")

out_layer.Save()
print("  Saved. Step 5 OK.\n")


# =========================================================================
# Step 6 — Apply PhysxJointAPI directly on SdfLayer
# =========================================================================
if _HAS_PHYSX:
    print("=== Step 6: Applying PhysxJointAPI via SdfLayer ===")
    out_layer.Reload()

    z1_revolute_specs = []
    def _find_z1_rev(spec):
        if "/Robot/base/Z1" in str(spec.path) and spec.typeName == "PhysicsRevoluteJoint":
            z1_revolute_specs.append(spec)
    _walk_sdf(out_layer.pseudoRoot, _find_z1_rev)

    applied_count = 0
    for spec in z1_revolute_specs:
        api_info = spec.GetInfo("apiSchemas")
        token = "PhysxJointAPI"
        existing = list(api_info.prependedItems) if api_info else []
        if token not in existing:
            new_info = Sdf.TokenListOp.Create(prependedItems=existing + [token])
            spec.SetInfo("apiSchemas", new_info)
            applied_count += 1

    out_layer.Save()
    print(f"  Applied PhysxJointAPI to {applied_count} Z1 joint spec(s). Saved. Step 6 OK.\n")


# =========================================================================
# Step 7 — Final validation from a fresh Stage open
# =========================================================================
print("=== Step 7: Final validation ===")
out_layer.Reload(force=True)
check = Usd.Stage.Open(OUTPUT_USD_PATH)

check_joints   = sorted([str(p.GetPath()) for p in check.Traverse()
                          if p.GetTypeName() in joint_types])
check_revolute = [j for j in check_joints
                  if check.GetPrimAtPath(j).GetTypeName() == "PhysicsRevoluteJoint"]
check_roots    = [str(p.GetPath()) for p in check.Traverse()
                  if UsdPhysics.ArticulationRootAPI(p)]
check_omni     = [str(p.GetPath()) for p in check.Traverse()
                  if p.GetTypeName() in omni_types and "/Robot/base/Z1" in str(p.GetPath())]
check_rb_miss  = [n for n in Z1_LINK_NAMES
                  if not UsdPhysics.RigidBodyAPI(check.GetPrimAtPath(f"/Robot/base/Z1/z1/{n}"))]

rj_check = check.GetPrimAtPath(RJ_PATH)
rj_b0 = UsdPhysics.Joint(rj_check).GetBody0Rel().GetTargets() if rj_check.IsValid() else []
rj_b1 = UsdPhysics.Joint(rj_check).GetBody1Rel().GetTargets() if rj_check.IsValid() else []

print(f"  Total joints        : {len(check_joints)}")
print(f"  Revolute joints     : {len(check_revolute)} (expected 18)")
print(f"  ArticulationRoot    : {check_roots} (expected 1 at /Robot/base)")
print(f"  OmniGraph prims     : {len(check_omni)} (expected 0)")
print(f"  Missing RigidBodyAPI: {check_rb_miss} (expected [])")
print(f"  root_joint body0    : {rj_b0} (expected [Sdf.Path('{BODY0}')])")
print(f"  root_joint body1    : {rj_b1} (expected [Sdf.Path('{BODY1}')])")

print("\n  Z1 joint body targets:")
for prim in check.Traverse():
    if "/Robot/base/Z1/z1" not in str(prim.GetPath()):
        continue
    if prim.GetTypeName() not in joint_types:
        continue
    j = UsdPhysics.Joint(prim)
    name = str(prim.GetPath()).split("/")[-1]
    print(f"    {name:12s}  body0={j.GetBody0Rel().GetTargets()}  "
          f"body1={j.GetBody1Rel().GetTargets()}")

errors = []
if len(check_revolute) != 18:
    errors.append(f"Revolute count: {len(check_revolute)} ≠ 18")
if len(check_roots) != 1:
    errors.append(f"ArticulationRootAPI count: {len(check_roots)}")
if check_omni:
    errors.append(f"OmniGraph prims: {check_omni}")
if check_rb_miss:
    errors.append(f"RigidBodyAPI missing: {check_rb_miss}")
if not rj_b0 or str(rj_b0[0]) != BODY0:
    errors.append(f"root_joint body0 wrong: {rj_b0}")
if not rj_b1 or str(rj_b1[0]) != BODY1:
    errors.append(f"root_joint body1 wrong: {rj_b1}")

if errors:
    for e in errors:
        print(f"  FAIL: {e}")
    raise RuntimeError("Validation failed — see above.")

print(f"\nAll checks passed.")
print(f"Composite USD ready: {OUTPUT_USD_PATH}")