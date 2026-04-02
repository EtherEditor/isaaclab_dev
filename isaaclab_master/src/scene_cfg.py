"""
Phase 1: Scene & Asset Configuration
Filename: scene_cfg.py
Innovation 1 addition: VoxelSDFCfg placeholder (Python-side SDF buffer).
"""
import os
import logging
import isaaclab.sim as sim_utils
from isaaclab.utils import configclass
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.assets import ArticulationCfg, RigidObjectCfg
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.sensors import (
    ContactSensorCfg,
    RayCasterCfg,
    patterns,
    FrameTransformerCfg,
    OffsetCfg,
)

_log = logging.getLogger(__name__)

# =========================================================================
# Composite USD Path — direct reference, no asset library fallback.
# This is the flattened file produced by compose_go2_z1.py.
# =========================================================================
_COMPOSITE_USD = os.path.join(os.path.dirname(os.path.abspath(__file__)), "go2_z1.usd")

if not os.path.exists(_COMPOSITE_USD):
    raise FileNotFoundError(
        f"Composite USD not found at '{_COMPOSITE_USD}'.\n"
        "Run the following command to generate it:\n"
        "    ./python.sh isaaclab_master/compose_go2_z1.py"
    )

_log.info(f"Composite robot USD located: {_COMPOSITE_USD}")

# =========================================================================
# Joint name expressions — currently only the leg joints are driven.
# =========================================================================
_LEG_JOINT_EXPR = [".*_hip_joint", ".*_thigh_joint", ".*_calf_joint"]


# =========================================================================
# Obstacle generation helper
# =========================================================================
def generate_obstacles(num_obstacles: int) -> dict:
    """Explicitly maps N obstacles to distinct prim paths."""
    obstacles = {}
    for i in range(num_obstacles):
        obstacles[f"obstacle_{i}"] = RigidObjectCfg(
            prim_path=f"{{ENV_REGEX_NS}}/Obstacles/cylinder_{i}",
            spawn=sim_utils.CylinderCfg(
                radius=0.15,
                height=0.5,
                rigid_props=sim_utils.RigidBodyPropertiesCfg(),
                collision_props=sim_utils.CollisionPropertiesCfg(),
            ),
            init_state=RigidObjectCfg.InitialStateCfg(pos=(1.0 + i * 0.5, 0.0, 0.25)),
        )
    return obstacles


# =========================================================================
# Scene Configuration
# =========================================================================
@configclass
class Go2RetrievalSceneCfg(InteractiveSceneCfg):
    """Scene configuration for the armed Go2 mobile manipulation task."""

    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        collision_group=-1,
    )

    robot: ArticulationCfg = ArticulationCfg(
        prim_path="{ENV_REGEX_NS}/Robot",
        spawn=sim_utils.UsdFileCfg(
            usd_path=_COMPOSITE_USD,
            activate_contact_sensors=True,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False,
                retain_accelerations=False,
                linear_damping=0.0,
                angular_damping=0.0,
                max_linear_velocity=1000.0,
                max_angular_velocity=1000.0,
                max_depenetration_velocity=1.0,
            ),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=True,
                solver_position_iteration_count=4,
                solver_velocity_iteration_count=0,
                fix_root_link=False,
            ),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.4),
            joint_pos={
                ".*_hip_joint":   0.0,
                ".*_thigh_joint": 0.8,
                ".*_calf_joint":  -1.5,
            },
        ),
        actuators={
            "go2_legs": ImplicitActuatorCfg(
                joint_names_expr=_LEG_JOINT_EXPR,
                stiffness=20.0,
                damping=0.5,
                velocity_limit=100.0,
                effort_limit=33.5,
            ),
        },
    )

    target_object = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/TargetObject",
        spawn=sim_utils.SphereCfg(
            radius=0.06,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            physics_material=sim_utils.RigidBodyMaterialCfg(
                static_friction=1.2,
                dynamic_friction=1.0,
            ),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(2.0, 0.0, 0.06)),
    )

    # Assign generated obstacles as scene entities so they are actually spawned.
    # Previously, generate_obstacles() was called but its return value was
    # discarded, causing the obstacles to never appear in the scene.
    obstacle_0: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Obstacles/cylinder_0",
        spawn=sim_utils.CylinderCfg(
            radius=0.15,
            height=0.5,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(),
            collision_props=sim_utils.CollisionPropertiesCfg(),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(1.0, 0.0, 0.25)),
    )
    obstacle_1: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Obstacles/cylinder_1",
        spawn=sim_utils.CylinderCfg(
            radius=0.15,
            height=0.5,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(),
            collision_props=sim_utils.CollisionPropertiesCfg(),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(1.5, 0.0, 0.25)),
    )
    obstacle_2: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Obstacles/cylinder_2",
        spawn=sim_utils.CylinderCfg(
            radius=0.15,
            height=0.5,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(),
            collision_props=sim_utils.CollisionPropertiesCfg(),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(2.0, 0.0, 0.25)),
    )

    base_contact = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/base",
        update_period=0.0,
        track_air_time=False,
    )

    arm_contact = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/.*",
        update_period=0.0,
        track_air_time=False,
    )

    ee_frame = FrameTransformerCfg(
        prim_path="{ENV_REGEX_NS}/Robot/base",
        target_frames=[
            FrameTransformerCfg.FrameCfg(
                prim_path="{ENV_REGEX_NS}/Robot/base/Z1/z1/link05",
                name="end_effector",
            )
        ],
    )

    lidar = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/base",
        offset=OffsetCfg(pos=(0.0, 0.0, 0.1)),
        ray_alignment="yaw",
        pattern_cfg=patterns.LidarPatternCfg(
            channels=1,
            vertical_fov_range=[0.0, 0.0],
            horizontal_fov_range=[-180.0, 180.0],
            horizontal_res=10.0,
        ),
        max_distance=3.0,
        mesh_prim_paths=["/World/ground"],
        debug_vis=False,
    )

    # =========================================================================
    # Innovation 3: Depth camera for visual encoder training
    # =========================================================================
    depth_camera = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/base",
        offset=OffsetCfg(pos=(0.3, 0.0, 0.05)),  # forward-facing
        ray_alignment="base",
        pattern_cfg=patterns.GridPatternCfg(
            resolution=1.0 / 64,
            size=(64, 64),
        ),
        max_distance=5.0,
        mesh_prim_paths=["/World/ground"],
        debug_vis=False,
    )

    # =========================================================================
    # Innovation 1: SDF buffer descriptor
    # The VoxelSDF lives entirely in Python/PyTorch GPU memory — no USD prim
    # is created.  This dataclass documents the shared parameters so both
    # scene_cfg and sdf_guidance stay in sync.
    # =========================================================================
    @configclass
    class VoxelSDFCfg:
        """Parameters for the Python-side voxel SDF used by SDFGuidance."""
        grid_res: int = 32
        world_min: tuple[float, float, float] = (-1.0, -1.5, -0.1)
        world_max: tuple[float, float, float] = ( 4.0,  1.5,  2.0)
        obstacle_radius: float = 0.15
        obstacle_height: float = 0.50

    voxel_sdf: VoxelSDFCfg = VoxelSDFCfg()