import gc
import os
import pickle
import numpy as np
import pickledb
from uuid import uuid4

from gefest.core.geometry.datastructs.structure import Structure

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

USE_AVG_CONST = True
BASE_MODEL_PATH  = r'Data\Prepared_NSS_0_10um_frog_RBC_2025.mph'

def poly_add(model, polygons_xy_str):
    for n, poly in enumerate(polygons_xy_str):
        name = f"pol{n+1}"
        try:
            model.java.component("comp1").geom("geom1").create(name, "Polygon")
        except Exception:
            pass
        feat = model.java.component("comp1").geom("geom1").feature(name)
        feat.set("x", poly[0])
        feat.set("y", poly[1])
    return model


def simulate_hydrodynamics(structure: Structure, client):
    gc.collect()
    target, mean_diff, model_uid = _load_fitness(structure)

    if target is None:
        model, model_uid = _load_simulation_result(client, structure)
        if model is None:
            poly_box = []
            for pol in structure.polygons:
                xs = " ".join(str(pt.x) for pt in pol.points)
                ys = " ".join(str(pt.y) for pt in pol.points)
                poly_box.append([xs, ys])

            print("Start COMSOL")
            model = client.load(BASE_MODEL_PATH)

            try:
                model = poly_add(model, poly_box)
                model.build()
                model.mesh()
                model.solve()
            except Exception as ex:
                print("Sim build/solve error:", ex)
                client.clear()
                return 0.0

            model_uid = _save_simulation_result(structure, model)

        try:
            outs = [
                model.evaluate("vlct_1"),
                model.evaluate("vlct_2"),
                model.evaluate("vlct_3"),
                model.evaluate("vlct_4"),
                model.evaluate("vlct_5"),
                model.evaluate("vlct_side"),
                model.evaluate("vlct_main"),
            ]
        except Exception as ex:
            print("Eval vlct_* error:", ex)
            client.clear()
            return 0.0

        try:
            u = np.array(model.evaluate("spf.U"), dtype=float)
        except Exception as ex:
            print("Eval spf.U error:", ex)
            client.clear()
            return 0.0

        try:
            curl = float(model.evaluate("curl"))
            curv_raw = float(model.evaluate("curv"))
        except Exception as ex:
            print("Eval curl/curv error:", ex)
            client.clear()
            return 0.0

        outs = [float(x) for x in outs]
        v1, v2, v3, v4, v5, v_side, v_main = outs

        u_pos = u[u > 0.0]
        slits = np.array([v1, v2, v3, v4, v5], dtype=float)
        side_main_sum = float(v_side + v_main)
        slits_sum = float(slits.sum())

        if slits_sum <= 0.0 or side_main_sum <= 0.0 or u_pos.size == 0:
            target = 0.0
            mean_diff = 100.0
        else:
            base_target = slits_sum / side_main_sum

            mean_slits = float(slits.mean())
            if mean_slits > 0.0:
                rel_diffs = np.abs(slits / mean_slits - 1.0)
                mean_rel_diff = float(rel_diffs.mean())
            else:
                rel_diffs = []
                mean_rel_diff = 1.0

            mean_u = float(u_pos.mean())
            max_u = float(u_pos.max())
            fast_u = mean_u
            width_ratio = float((u > fast_u).sum()) / float((u > 0.0).sum())

            curl_thr = 30000.0
            curv_thr = 7.0e7

            low_w, high_w = 0.25, 0.43
            if width_ratio < low_w:
                dist_width = low_w - width_ratio
            elif width_ratio > high_w:
                dist_width = width_ratio - high_w
            else:
                dist_width = 0.0

            pen_width = 1.0 / (1.0 + 5.0 * dist_width)      
            pen_eq    = 1.0 / (1.0 + 10.0 * mean_rel_diff)   

            excess_curl = max(0.0, curl - curl_thr) / curl_thr
            pen_curl    = 1.0 / (1.0 + 1.0 * excess_curl)    

            excess_curv = max(0.0, curv_raw - curv_thr) / curv_thr
            pen_curv    = 1.0 / (1.0 + 0.5 * excess_curv)    

            penalty = pen_width * pen_eq * pen_curl * pen_curv
            target = base_target * penalty
            mean_diff = mean_rel_diff * 100.0

        client.clear()
        _save_fitness(structure, target, mean_diff)

        print(
            "score:",
            round(target, 6),
            "vlcts:",
            [round(x, 6) for x in outs],
            "curl:",
            round(curl, 2),
            "curv:",
            f"{curv_raw:.2e}",
        )
    else:
        print(f"Cached: {target}")

    return -target


def _save_simulation_result(configuration, model):
    if not os.path.exists("./models"):
        os.mkdir("./models")
    model_uid = str(uuid4())
    model.save(f"./models/{model_uid}.mph")
    db = pickledb.load("comsol_db.saved", False)
    db.set(str(configuration), model_uid)
    db.dump()

    if not os.path.exists("./structures"):
        os.mkdir("./structures")
    with open(f"./structures/{model_uid}.str", "wb") as f:
        pickle.dump(configuration, f)

    return model_uid


def _load_simulation_result(client, configuration):
    db = pickledb.load("comsol_db.saved", False)
    model_uid = db.get(str(configuration))
    if model_uid is False:
        return None, None
    model = client.load(f"./models/{model_uid}.mph")
    return model, model_uid


def _save_fitness(configuration, fitness1, fitness2):
    db = pickledb.load("fitness_db.saved", False)
    db.set(str(configuration), f"{fitness1}|{fitness2}")
    db.dump()


def _load_fitness(configuration):
    db = pickledb.load("fitness_db.saved", False)
    db_models = pickledb.load("comsol_db.saved", False)
    model_uid = db_models.get(str(configuration))
    res = db.get(str(configuration))
    if res is False:
        return None, None, None
    f, m = res.split("|")
    return float(f), float(m), model_uid