#!/usr/bin/env python3
"""Check that SAPIEN can render RoboTwin's evaluation cameras on this GPU, and time it.

RoboTwin 13c3c47 (envs/_base_task.py setup_scene) renders every camera with the ray-tracing shader pack "rt":
32 samples per pixel, path depth 8, OIDN denoiser. Evaluation (task_config demo_clean / demo_randomized) uses three
D435 cameras, 320x240, fovy 37 degrees. Policies are trained on images rendered this way, so switching the evaluation
to the rasterizing "default" shader would change what the policy sees; "default" is timed here for reference only.

Data-center GPUs have no RT cores, and containers only get the NVIDIA graphics libraries when
NVIDIA_DRIVER_CAPABILITIES includes graphics (or is "all"). Each configuration runs in its own subprocess, so a
crash or hang in one does not stop the others.

  CUDA_VISIBLE_DEVICES=0 python rt_probe.py [--out DIR] [--frames N] [--timeout S]

Needs only sapien==3.0.0b1 (the wheel bundles its Vulkan loader and OIDN) and numpy<2. Exit status 0 when "rt" with
OIDN renders three non-blank cameras.
"""
import argparse
import glob
import json
import os
import subprocess
import sys
import time

CONFIGS = [("rt", "oidn"), ("rt", "none"), ("default", "none")]
GRAPHICS_LIBS = ("libGLX_nvidia.so.0", "libEGL_nvidia.so.0", "libnvidia-glcore.so", "libnvidia-glvkspirv.so",
                 "libnvidia-rtcore.so")
CAMERAS = {  # name: (position, look-at target); roughly RoboTwin's head and wrist views of a table at z = 0.74
    "head_camera": ([0.0, -0.45, 1.35], [0.0, 0.05, 0.74]),
    "left_camera": ([-0.25, -0.15, 1.0], [-0.1, 0.05, 0.74]),
    "right_camera": ([0.25, -0.15, 1.0], [0.1, 0.05, 0.74]),
}


def graphics_libs() -> dict:
    found = {}
    for name in GRAPHICS_LIBS:
        hits = []
        for d in ("/usr/lib/x86_64-linux-gnu", "/usr/lib64", "/usr/lib"):
            hits += glob.glob(os.path.join(d, name + "*"))
        found[name] = sorted(hits)[0] if hits else None
    return found


def look_at(np, pos, target):
    forward = np.asarray(target, float) - np.asarray(pos, float)
    forward /= np.linalg.norm(forward)
    left = np.cross([0.0, 0.0, 1.0], forward)
    left /= np.linalg.norm(left)
    mat = np.eye(4)
    mat[:3, :3] = np.stack([forward, left, np.cross(forward, left)], axis=1)
    mat[:3, 3] = pos
    return mat


def box(sapien, scene, pos, half_size, color, static=False):
    # same construction as RoboTwin envs/utils/create_actor.py create_entity_box
    entity = sapien.Entity()
    rigid = sapien.physx.PhysxRigidStaticComponent() if static else sapien.physx.PhysxRigidDynamicComponent()
    rigid.attach(sapien.physx.PhysxCollisionShapeBox(half_size=half_size, material=scene.default_physical_material))
    render = sapien.render.RenderBodyComponent()
    render.attach(sapien.render.RenderShapeBox(half_size, sapien.render.RenderMaterial(base_color=[*color, 1])))
    entity.add_component(rigid)
    entity.add_component(render)
    entity.set_pose(sapien.Pose(pos))
    scene.add_entity(entity)


def child(shader: str, denoiser: str, frames: int, out: str) -> dict:
    import numpy as np
    import sapien
    from sapien.render import set_global_config

    sapien.set_log_level("info")  # svulkan2 prints the Vulkan device table, including a RayTracing column
    t0 = time.perf_counter()
    # same renderer settings as RoboTwin envs/_base_task.py setup_scene
    engine = sapien.Engine()
    set_global_config(max_num_materials=50000, max_num_textures=50000)
    renderer = sapien.SapienRenderer()
    engine.set_renderer(renderer)
    sapien.render.set_camera_shader_dir(shader)
    if shader == "rt":
        sapien.render.set_ray_tracing_samples_per_pixel(32)
        sapien.render.set_ray_tracing_path_depth(8)
        sapien.render.set_ray_tracing_denoiser(denoiser)
    scene = engine.create_scene(sapien.SceneConfig())
    scene.set_timestep(1 / 250)
    scene.add_ground(0)
    scene.default_physical_material = scene.create_physical_material(0.5, 0.5, 0)
    scene.set_ambient_light([0.5, 0.5, 0.5])
    scene.add_directional_light([0, 0.5, -1], [0.5, 0.5, 0.5], shadow=True)
    for p in ([1, 0, 1.8], [-1, 0, 1.8]):
        scene.add_point_light(p, [1, 1, 1], shadow=True)

    box(sapien, scene, [0, 0.05, 0.37], [0.6, 0.35, 0.37], [0.55, 0.4, 0.3], static=True)  # table
    rng = np.random.default_rng(0)
    for i in range(6):
        box(sapien, scene, [-0.25 + 0.1 * i, 0.05 + 0.08 * (i % 2), 0.78], [0.025, 0.025, 0.04], rng.uniform(0.1, 0.9, 3))

    cams = []
    for name, (pos, target) in CAMERAS.items():
        cam = scene.add_camera(name=name, width=320, height=240, fovy=np.deg2rad(37), near=0.1, far=100)
        cam.entity.set_pose(sapien.Pose(look_at(np, pos, target)))
        cams.append(cam)
    setup_s = time.perf_counter() - t0

    times, rgba = [], None
    for _ in range(frames):
        t = time.perf_counter()
        for _ in range(10):  # RoboTwin steps physics several times per control step
            scene.step()
        scene.update_render()
        pictures = []
        for cam in cams:
            cam.take_picture()
            pictures.append(cam.get_picture("Color"))
        times.append(time.perf_counter() - t)
        rgba = pictures
    imgs = [(p * 255).clip(0, 255).astype("uint8") for p in rgba]
    stats = [(float(i[..., :3].mean()), float(i[..., :3].std())) for i in imgs]
    saved = None
    if out:
        os.makedirs(out, exist_ok=True)
        try:
            import cv2
            saved = os.path.join(out, f"{shader}_{denoiser}.png")
            cv2.imwrite(saved, cv2.cvtColor(np.concatenate([i[..., :3] for i in imgs], axis=1), cv2.COLOR_RGB2BGR))
        except ImportError:
            saved = os.path.join(out, f"{shader}_{denoiser}.npy")
            np.save(saved, np.stack(imgs))
    steady = sorted(times[1:]) or times
    return {
        "setup_s": round(setup_s, 2),
        "first_frame_s": round(times[0], 3),
        "median_frame_ms": round(1000 * steady[len(steady) // 2], 1),
        "frames": frames,
        "blank": any(m < 2 or m > 253 or s < 1 for m, s in stats),
        "mean_std": [[round(m, 1), round(s, 1)] for m, s in stats],
        "saved": saved,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="rt_probe_out", help="directory for one PNG per configuration")
    ap.add_argument("--frames", type=int, default=20, help="control steps to render (three cameras each)")
    ap.add_argument("--timeout", type=int, default=300, help="seconds per configuration")
    ap.add_argument("--child", nargs=2, metavar=("SHADER", "DENOISER"), help=argparse.SUPPRESS)
    args = ap.parse_args()

    if args.child:
        result = child(args.child[0], args.child[1], args.frames, args.out)
        print("RT_PROBE_RESULT " + json.dumps(result), flush=True)
        return 0

    print(f"NVIDIA_DRIVER_CAPABILITIES={os.environ.get('NVIDIA_DRIVER_CAPABILITIES', '<unset>')} "
          f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', '<unset>')}")
    for name, path in graphics_libs().items():
        print(f"  {name}: {path or 'MISSING'}")
    results = {}
    for shader, denoiser in CONFIGS:
        label = f"{shader}+{denoiser}"
        cmd = [sys.executable, os.path.abspath(__file__), "--child", shader, denoiser,
               "--frames", str(args.frames), "--out", args.out]
        try:
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=args.timeout)
            lines = (proc.stdout + proc.stderr).splitlines()
            found = [l for l in lines if l.startswith("RT_PROBE_RESULT ")]
            if found:
                results[label] = json.loads(found[-1].split(" ", 1)[1])
            else:
                results[label] = {"error": f"exit {proc.returncode}"}
        except subprocess.TimeoutExpired as e:
            text = [x.decode(errors="ignore") if isinstance(x, bytes) else (x or "") for x in (e.stdout, e.stderr)]
            lines = "".join(text).splitlines()
            results[label] = {"error": f"timeout after {args.timeout}s"}
        keep = [l for l in lines if any(k in l for k in ("Vulkan", "RayTracing", "NVIDIA", "error", "Error", "denois",
                                                          "OIDN", "oidn", "Traceback"))
                or l.lstrip().startswith(tuple("0123456789"))]
        print(f"== {label}")
        for l in keep[:25]:
            print("   " + l[:200])
        if "error" in results[label]:
            for l in lines[-8:]:
                print("   | " + l[:200])
        print("   result: " + json.dumps(results[label]))

    rt = results.get("rt+oidn", {})
    ok = "error" not in rt and not rt.get("blank", True)
    summary = "  ".join(f"{k}={v.get('median_frame_ms', v.get('error'))}" + ("ms" if "median_frame_ms" in v else "")
                        for k, v in results.items())
    print(f"RT_PROBE {'PASS' if ok else 'FAIL'}  per control step, 3 cameras 320x240: {summary}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
