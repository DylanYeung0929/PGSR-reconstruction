#!/usr/bin/env python3
"""将 STEP 曲面离散化为用于多视角渲染的三角网格 PLY。"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import open3d as o3d
from OCP.BRep import BRep_Tool
from OCP.BRepBndLib import BRepBndLib
from OCP.BRepMesh import BRepMesh_IncrementalMesh
from OCP.Bnd import Bnd_Box
from OCP.IFSelect import IFSelect_RetDone
from OCP.STEPControl import STEPControl_Reader
from OCP.TopAbs import TopAbs_FACE, TopAbs_REVERSED
from OCP.TopExp import TopExp_Explorer
from OCP.TopLoc import TopLoc_Location
from OCP.TopoDS import TopoDS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="STEP 曲面转三角网格 PLY")
    parser.add_argument("--input", type=Path, required=True, help="输入 .stp/.step 文件")
    parser.add_argument("--output", type=Path, required=True, help="输出 .ply 文件")
    parser.add_argument(
        "--relative-deflection",
        type=float,
        default=0.0015,
        help="线性离散误差占包围盒对角线的比例，越小网格越密",
    )
    parser.add_argument("--angular-deflection", type=float, default=0.35, help="角度离散误差（弧度）")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.input.is_file():
        raise FileNotFoundError(f"找不到 STEP 文件：{args.input}")
    if args.relative_deflection <= 0:
        raise ValueError("--relative-deflection 必须大于0")

    reader = STEPControl_Reader()
    if reader.ReadFile(str(args.input)) != IFSelect_RetDone:
        raise RuntimeError(f"无法读取 STEP 文件：{args.input}")
    if reader.TransferRoots() == 0:
        raise RuntimeError("STEP 文件中没有可转换的几何根节点")
    shape = reader.OneShape()

    bounds = Bnd_Box()
    BRepBndLib.Add_s(shape, bounds)
    xmin, ymin, zmin = bounds.GetXMin(), bounds.GetYMin(), bounds.GetZMin()
    xmax, ymax, zmax = bounds.GetXMax(), bounds.GetYMax(), bounds.GetZMax()
    extent = np.array([xmax - xmin, ymax - ymin, zmax - zmin], dtype=np.float64)
    diagonal = float(np.linalg.norm(extent))
    if not math.isfinite(diagonal) or diagonal <= 0:
        raise RuntimeError("STEP 包围盒无效")
    linear_deflection = diagonal * args.relative_deflection

    mesher = BRepMesh_IncrementalMesh(
        shape, linear_deflection, False, args.angular_deflection, True
    )
    mesher.Perform()
    if not mesher.IsDone():
        raise RuntimeError("OpenCascade 曲面离散化失败")

    vertices: list[list[float]] = []
    triangles: list[list[int]] = []
    face_count = 0
    explorer = TopExp_Explorer(shape, TopAbs_FACE)
    while explorer.More():
        face = TopoDS.Face(explorer.Current())
        location = TopLoc_Location()
        triangulation = BRep_Tool.Triangulation_s(face, location)
        if triangulation is not None:
            transform = location.Transformation()
            offset = len(vertices)
            for node_id in range(1, triangulation.NbNodes() + 1):
                point = triangulation.Node(node_id).Transformed(transform)
                vertices.append([point.X(), point.Y(), point.Z()])
            reverse = face.Orientation() == TopAbs_REVERSED
            for triangle_id in range(1, triangulation.NbTriangles() + 1):
                n1, n2, n3 = triangulation.Triangle(triangle_id).Get()
                indices = [offset + n1 - 1, offset + n2 - 1, offset + n3 - 1]
                if reverse:
                    indices[1], indices[2] = indices[2], indices[1]
                triangles.append(indices)
            face_count += 1
        explorer.Next()

    if not triangles:
        raise RuntimeError("离散化后没有得到三角面")
    mesh = o3d.geometry.TriangleMesh(
        o3d.utility.Vector3dVector(np.asarray(vertices, dtype=np.float64)),
        o3d.utility.Vector3iVector(np.asarray(triangles, dtype=np.int32)),
    )
    mesh.remove_duplicated_vertices()
    mesh.remove_duplicated_triangles()
    mesh.remove_degenerate_triangles()
    mesh.remove_unreferenced_vertices()
    mesh.compute_vertex_normals()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    if not o3d.io.write_triangle_mesh(str(args.output), mesh, write_ascii=False):
        raise RuntimeError(f"无法写出网格：{args.output}")
    print(f"输入：{args.input}")
    print(f"尺寸：{extent[0]:.6g} × {extent[1]:.6g} × {extent[2]:.6g}")
    print(f"线性离散误差：{linear_deflection:.6g}")
    print(f"有效曲面：{face_count}")
    print(f"输出：{len(mesh.vertices)} 个顶点，{len(mesh.triangles)} 个三角面")
    print(f"文件：{args.output}")


if __name__ == "__main__":
    main()
