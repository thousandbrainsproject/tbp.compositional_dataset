"""Blender picker for placing and baking a PNG sticker onto a mesh texture."""

from __future__ import annotations

import argparse
from pathlib import Path
import site
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
USER_SITE = site.getusersitepackages()
if USER_SITE not in sys.path:
    site.addsitedir(USER_SITE)

from PIL import Image

from tbp.compositional_datasets import blender_scene
from tbp.compositional_datasets.blender_cli import argv_after_blender_separator
from tbp.compositional_datasets.blender_texture_baking import (
    DEFAULT_MIN_COVERAGE,
    TextureStampError,
    TextureStampPlacement,
    stamp_texture_placements_on_object,
    triangulate_mesh_object,
)
from tbp.compositional_datasets.geometry import (
    is_degenerate_triangle,
    sticker_quad_corners,
    surface_frame,
)
from tbp.compositional_datasets.metadata import generation_metadata, write_metadata
from tbp.compositional_datasets.placements import SurfaceAttachment
from tbp.compositional_datasets.sticker_layout import (
    rotation_aligning_bitangent_to_world_up,
    size_for_longest_side,
)
from tbp.compositional_datasets.uv_stamping import barycentric_coordinates_3d

import bpy

DEFAULT_INITIAL_SCALE_FRACTION = 0.2


class PickerState:
    """Mutable state shared by Blender picker operators.

    Attributes:
        args: Parsed command-line arguments.
        target_object: Mesh object selected by the latest pick.
        hit_face_index: Face index selected by the latest pick.
        barycentric: Barycentric coordinates for the selected point.
        preview_object: Temporary preview mesh shown in the scene.
    """

    def __init__(self, args: argparse.Namespace) -> None:
        """Create picker state for one Blender session.

        Args:
            args: Parsed command-line arguments.
        """
        self.args = args
        self.target_object: Any | None = None
        self.hit_face_index: int | None = None
        self.barycentric: Any | None = None
        self.preview_object: Any | None = None


STATE: PickerState | None = None


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse texture-stamp picker command-line arguments.

    Args:
        argv: Optional argument list. Uses sys.argv when None.

    Returns:
        Parsed command-line namespace.
    """
    parser = argparse.ArgumentParser(
        description="Interactively pick a mesh surface point and bake a PNG sticker into the parent texture."
    )
    parser.add_argument("--object", required=True, type=Path, help="Parent mesh path.")
    parser.add_argument("--sticker", required=True, type=Path, help="PNG sticker path.")
    parser.add_argument(
        "--out",
        type=Path,
        help="Output GLB path. Defaults to outputs/<object>_<sticker>_texture_stamp.glb.",
    )
    args = parser.parse_args(argv)
    if args.out is None:
        args.out = Path("outputs") / f"{args.object.stem}_{args.sticker.stem}_texture_stamp.glb"
    return args


def _initialize_sticker_scale(args: argparse.Namespace, bounds_radius: float) -> None:
    """Initialize derived sticker scale fields on parsed picker arguments.

    Args:
        args: Parsed picker arguments with a `sticker` path.
        bounds_radius: Parent object bounds radius in object-space units.
    """
    with Image.open(args.sticker) as image:
        width, height = image.size
    args.sticker_aspect_ratio = width / height
    args.sticker_scale = DEFAULT_INITIAL_SCALE_FRACTION * float(bounds_radius)
    args.size = list(size_for_longest_side(args.sticker, args.sticker_scale))
    args.rotation_deg = 0.0


def _picker_rotation_about_normal_deg(triangle: Any, user_rotation_deg: float) -> float:
    """Return picker rotation with zero aligned to Blender world up.

    Args:
        triangle: Target triangle vertices.
        user_rotation_deg: User-requested rotation in degrees.

    Returns:
        Effective rotation around the triangle normal in degrees. When Blender
        world Z cannot be projected into the surface plane, returns the user
        rotation unchanged.
    """
    user_rotation = float(user_rotation_deg)
    try:
        baseline = rotation_aligning_bitangent_to_world_up(
            triangle,
            world_up=(0.0, 0.0, 1.0),
        )
    except ValueError as error:
        if str(error) == "world up is parallel to the target surface normal":
            return user_rotation
        raise
    return baseline + user_rotation


def _remove_preview(bpy_module: Any) -> None:
    """Remove the current preview object from the Blender scene."""
    global STATE
    if STATE is None or STATE.preview_object is None:
        return
    bpy_module.data.objects.remove(STATE.preview_object, do_unlink=True)
    STATE.preview_object = None


def _create_sticker_preview_material(bpy_module: Any, sticker_path: Path) -> Any:
    """Create a transparent material for the interactive preview sticker.

    Args:
        bpy_module: Blender bpy module.
        sticker_path: Sticker image path.

    Returns:
        Blender material with sticker color and alpha texture wiring.
    """
    material = bpy_module.data.materials.new(name=f"Sticker_{sticker_path.stem}")
    material.use_nodes = True
    material.blend_method = "BLEND"
    if hasattr(material, "use_screen_refraction"):
        material.use_screen_refraction = True

    nodes = material.node_tree.nodes
    links = material.node_tree.links
    principled = nodes.get("Principled BSDF")
    texture = nodes.new(type="ShaderNodeTexImage")
    texture.image = bpy_module.data.images.load(str(sticker_path))

    if principled is not None:
        if "Base Color" in principled.inputs:
            links.new(texture.outputs["Color"], principled.inputs["Base Color"])
        if "Alpha" in principled.inputs:
            links.new(texture.outputs["Alpha"], principled.inputs["Alpha"])

    return material


def _create_preview(bpy_module: Any, obj: Any, polygon_index: int, barycentric: Any) -> Any:
    """Create a temporary sticker preview mesh at the selected surface point.

    Args:
        bpy_module: Blender bpy module.
        obj: Blender mesh object receiving the preview.
        polygon_index: Target polygon index.
        barycentric: Barycentric coordinates for the selected point.

    Returns:
        The created preview object.

    Raises:
        RuntimeError: If picker state has not been initialized.
    """
    global STATE
    if STATE is None:
        raise RuntimeError("Picker state is not initialized")

    _remove_preview(bpy_module)
    triangle, attachment = _surface_attachment_for_selection(
        obj,
        polygon_index,
        barycentric,
        STATE.args,
    )
    frame = surface_frame(
        triangle,
        attachment.barycentric,
        attachment.rotation_about_normal_deg,
    )
    corners = sticker_quad_corners(
        frame.anchor + frame.normal * 0.001,
        frame.rotated_tangent,
        frame.rotated_bitangent,
        attachment.size,
    )

    mesh = bpy_module.data.meshes.new("texture_stamp_preview_mesh")
    mesh.from_pydata(corners.tolist(), [], [[0, 1, 2], [0, 2, 3]])
    mesh.update()
    preview = bpy_module.data.objects.new("texture_stamp_preview", mesh)
    bpy_module.context.collection.objects.link(preview)

    preview.data.materials.append(_create_sticker_preview_material(bpy_module, STATE.args.sticker))
    STATE.preview_object = preview
    return preview


class TBP_OT_pick_texture_stamp(bpy.types.Operator):
    """Blender operator that lets the user click a mesh face for stamping."""

    bl_idname = "tbp.pick_texture_stamp"
    bl_label = "Pick Texture Stamp Location"
    bl_options = {"REGISTER", "UNDO"}

    def modal(self, context: Any, event: Any) -> set[str]:
        global STATE
        if event.type in {"RIGHTMOUSE", "ESC"}:
            return {"CANCELLED"}
        if event.type != "LEFTMOUSE" or event.value != "PRESS":
            return {"RUNNING_MODAL"}
        if STATE is None:
            self.report({"ERROR"}, "Picker state is not initialized")
            return {"CANCELLED"}

        from bpy_extras import view3d_utils  # type: ignore[import-not-found]

        region = context.region
        region_data = context.region_data
        coord = (event.mouse_region_x, event.mouse_region_y)
        origin = view3d_utils.region_2d_to_origin_3d(region, region_data, coord)
        direction = view3d_utils.region_2d_to_vector_3d(region, region_data, coord)
        depsgraph = context.evaluated_depsgraph_get()
        hit, location, _normal, face_index, obj, _matrix = context.scene.ray_cast(
            depsgraph,
            origin,
            direction,
        )
        if not hit or obj is None or obj.type != "MESH" or face_index < 0:
            self.report({"WARNING"}, "Click did not hit a mesh face")
            return {"RUNNING_MODAL"}

        triangle = blender_scene.mesh_triangle(obj, face_index)
        if is_degenerate_triangle(triangle):
            self.report({"WARNING"}, "Selected face is degenerate; pick another surface face")
            return {"RUNNING_MODAL"}
        try:
            barycentric = barycentric_coordinates_3d(triangle, [location.x, location.y, location.z])
        except ValueError as error:
            if str(error) == "triangle is degenerate":
                self.report({"WARNING"}, "Selected face is degenerate; pick another surface face")
                return {"RUNNING_MODAL"}
            raise
        STATE.target_object = obj
        STATE.hit_face_index = int(face_index)
        STATE.barycentric = barycentric
        _create_preview(bpy, obj, face_index, barycentric)
        self.report({"INFO"}, f"Selected face {face_index}; adjust settings and confirm stamp")
        return {"FINISHED"}

    def invoke(self, context: Any, _event: Any) -> set[str]:
        context.window_manager.modal_handler_add(self)
        self.report({"INFO"}, "Click a mesh surface to preview sticker placement")
        return {"RUNNING_MODAL"}


class TBP_OT_confirm_texture_stamp(bpy.types.Operator):
    """Blender operator that bakes the selected sticker into the texture."""

    bl_idname = "tbp.confirm_texture_stamp"
    bl_label = "Confirm Texture Stamp"
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context: Any) -> set[str]:
        """Bake the sticker, export the GLB, and write metadata."""
        global STATE
        if STATE is None or STATE.target_object is None or STATE.hit_face_index is None or STATE.barycentric is None:
            self.report({"ERROR"}, "Pick a mesh surface before confirming")
            return {"CANCELLED"}

        args = STATE.args
        try:
            output_path = Path(args.out)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            texture_path = output_path.with_name(f"{output_path.stem}_stamped_texture.png")
            triangle, attachment = _surface_attachment_for_selection(
                STATE.target_object,
                STATE.hit_face_index,
                STATE.barycentric,
                args,
            )
            texture_stamps = stamp_texture_placements_on_object(
                bpy,
                STATE.target_object,
                placements=[
                    TextureStampPlacement(
                        slot_name="picker",
                        polygon_index=STATE.hit_face_index,
                        attachment=attachment,
                        sticker_path=args.sticker,
                    )
                ],
                texture_path=texture_path,
                min_coverage=DEFAULT_MIN_COVERAGE,
            )
            texture_stamp = texture_stamps[0]
            frame = surface_frame(
                triangle,
                attachment.barycentric,
                attachment.rotation_about_normal_deg,
            )
            quad_corners = sticker_quad_corners(
                frame.anchor,
                frame.rotated_tangent,
                frame.rotated_bitangent,
                attachment.size,
            )
            metadata = generation_metadata(
                parent_mesh_path=args.object,
                sticker_path=args.sticker,
                attachment=attachment,
                frame=frame,
                quad_corners=quad_corners,
                texture_stamp=texture_stamp,
            )
            _remove_preview(bpy)
            bpy.ops.export_scene.gltf(filepath=str(output_path), export_format="GLB")
            metadata_path = write_metadata(metadata, output_path)
            self.report({"INFO"}, f"Wrote {output_path} and {metadata_path}")
            return {"FINISHED"}
        except (TextureStampError, ValueError) as error:
            self.report({"ERROR"}, str(error))
            return {"CANCELLED"}


class TBP_OT_cancel_texture_stamp(bpy.types.Operator):
    """Blender operator that removes the current preview."""

    bl_idname = "tbp.cancel_texture_stamp"
    bl_label = "Cancel Texture Stamp"
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, _context: Any) -> set[str]:
        _remove_preview(bpy)
        return {"FINISHED"}


class TBP_PT_texture_stamp_panel(bpy.types.Panel):
    """View3D sidebar panel for the texture-stamp picker."""

    bl_idname = "TBP_PT_texture_stamp_panel"
    bl_label = "Texture Stamp"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "TBP"

    def draw(self, context: Any) -> None:
        layout = self.layout
        scene = context.scene
        layout.prop(scene, "tbp_texture_stamp_scale")
        layout.prop(scene, "tbp_texture_stamp_rotation")
        layout.operator(TBP_OT_pick_texture_stamp.bl_idname, text="Pick Surface")
        layout.operator(TBP_OT_confirm_texture_stamp.bl_idname, text="Confirm Stamp")
        layout.operator(TBP_OT_cancel_texture_stamp.bl_idname, text="Cancel")


CLASSES = (
    TBP_OT_pick_texture_stamp,
    TBP_OT_confirm_texture_stamp,
    TBP_OT_cancel_texture_stamp,
    TBP_PT_texture_stamp_panel,
)


def register(bpy_module: Any) -> None:
    for cls in CLASSES:
        bpy_module.utils.register_class(cls)
    bpy_module.types.Scene.tbp_texture_stamp_scale = bpy_module.props.FloatProperty(
        name="Scale",
        default=float(STATE.args.sticker_scale) if STATE is not None else 0.2,
        min=0.0001,
        update=_sync_scene_settings,
    )
    bpy_module.types.Scene.tbp_texture_stamp_rotation = bpy_module.props.FloatProperty(
        name="Rotation",
        default=float(STATE.args.rotation_deg) if STATE is not None else 0.0,
        update=_sync_scene_settings,
    )


def unregister(bpy_module: Any) -> None:
    for cls in reversed(CLASSES):
        bpy_module.utils.unregister_class(cls)


def _sync_scene_settings(scene: Any, _context: Any) -> None:
    """Sync sidebar settings into picker state and refresh the preview.

    Args:
        scene: Blender scene containing texture-stamp properties.
        _context: Blender callback context, unused.
    """
    global STATE
    if STATE is None:
        return
    STATE.args.sticker_scale = scene.tbp_texture_stamp_scale
    STATE.args.size = list(size_for_longest_side(STATE.args.sticker, STATE.args.sticker_scale))
    STATE.args.rotation_deg = scene.tbp_texture_stamp_rotation
    if STATE.target_object is not None and STATE.hit_face_index is not None and STATE.barycentric is not None:
        try:
            _create_preview(bpy, STATE.target_object, STATE.hit_face_index, STATE.barycentric)
        except Exception:
            pass


def _surface_attachment_for_selection(
    obj: Any,
    polygon_index: int,
    barycentric: Any,
    args: argparse.Namespace,
) -> tuple[Any, SurfaceAttachment]:
    """Create the current surface attachment for a selected mesh face.

    Args:
        obj: Blender mesh object receiving the sticker.
        polygon_index: Selected polygon index.
        barycentric: Barycentric coordinates for the sticker center.
        args: Picker arguments containing size and rotation settings.

    Returns:
        The selected triangle and corresponding surface attachment.

    Raises:
        ValueError: If the selected triangle is degenerate.
    """
    triangle = blender_scene.mesh_triangle(obj, polygon_index)
    if is_degenerate_triangle(triangle):
        raise ValueError("Selected face is degenerate; pick another surface face")
    attachment = SurfaceAttachment(
        triangle_id=polygon_index,
        barycentric=barycentric,
        rotation_about_normal_deg=_picker_rotation_about_normal_deg(
            triangle,
            args.rotation_deg,
        ),
        size=tuple(args.size),
    )
    return triangle, attachment


def _prepare_scene(bpy_module: Any, args: argparse.Namespace) -> Any:
    """Prepare Blender scene for interactive texture-stamp picking.

    Args:
        bpy_module: Blender bpy module.
        args: Parsed command-line arguments.

    Returns:
        Prepared mesh object.
    """
    if args.object.suffix.lower() != ".glb":
        raise ValueError("Texture stamp picker requires a .glb parent asset")

    blender_scene.clear_scene(bpy_module)
    bpy_module.ops.import_scene.gltf(filepath=str(args.object))
    obj = [obj for obj in bpy_module.context.scene.objects if obj.type == "MESH"][0]
    triangulate_mesh_object(obj)
    degenerate_count = blender_scene.count_degenerate_triangles(obj)
    if degenerate_count:
        print(
            f"Imported mesh contains {degenerate_count} degenerate triangle(s); "
            "those faces cannot receive stamps"
        )
    bpy_module.context.view_layer.objects.active = obj
    obj.select_set(True)
    return obj


def main() -> None:
    """Run the Blender texture-stamp picker script."""
    global STATE
    args = parse_args(argv_after_blender_separator())
    if not args.sticker.exists():
        raise FileNotFoundError(f"Sticker image not found: {args.sticker}")
    if bpy is None:
        raise RuntimeError("Texture stamp picker must run inside Blender")

    STATE = PickerState(args=args)
    obj = _prepare_scene(bpy, args)
    _initialize_sticker_scale(args, blender_scene.object_bounds(obj)[1])
    register(bpy)
    print("Texture stamp picker ready. Use the View3D sidebar TBP panel to pick and confirm.")


if __name__ == "__main__":
    main()
