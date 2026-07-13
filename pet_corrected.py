"""Compute Physiological Equivalent Temperature (PET) values."""

from __future__ import annotations

import warnings
from importlib import import_module
from typing import Any, cast

np: Any = cast("Any", import_module("numpy"))
type Dyn = Any
type State = dict[str, Dyn]

ICL_MIN_INPUT = 0.03
ICL_MIN_VALUE = 0.02
ICL_UPPER_LIGHT = 0.6
ICL_UPPER_HEAVY = 2.0
ICL_MID = 0.3
COUNT_STAGE_TWO = 2
COUNT_STAGE_THREE = 3
J_REF_TWO = 2
J_REF_THREE = 3
J_REF_FOUR = 4
J_REF_FIVE = 5
J_REF_SEVEN = 7
TSK_GUARD = 33.85
VB_UPPER_LIMIT = 91.0
VB_LOWER_LIMIT = 89.0
VB_CAP = 90.0
PET_ITERATION_LIMIT = 400
PET_ENERGY_TOLERANCE = 0.01


class PetConstants:
    """Store thermodynamic and physical constants for PET equations."""

    po: float = 1013.25
    p: float = 1013.25
    rob: float = 1.06
    cb: float = 3640.0
    emsk: float = 0.99
    emcl: float = 0.95
    lvap: float = 2.42 * 10.0**6.0
    sigm: float = 5.67 * 10.0**-8.0
    cair: float = 1010.0
    rdsk: float = 0.79 * 10.0**7.0
    rdcl: float = 0.0
    tc_set: float = 36.6
    tsk_set: float = 34.0
    tbody_set: float = 0.1 * 34.0 + 0.9 * 36.6


def svp_murray(t_k: Dyn) -> Dyn:
    """Compute saturation vapor pressure from air temperature in Kelvin."""
    t_c = t_k - 273.15
    return 6.11 * 10.0 ** (7.45 * t_c / (235.0 + t_c))


def c2k(x: Dyn) -> Dyn:
    """Convert Celsius to Kelvin."""
    return x + 273.15


def _prepare_context(
    tair: Dyn,
    t_mrt: Dyn,
    v_air: Dyn,
    rh: Dyn,
    m_activity: Dyn,
    icl: Dyn,
    body_position: Dyn,
    sex: Dyn,
    age: Dyn,
    mbody: Dyn,
    ht: Dyn,
) -> tuple[State, bool]:
    """Broadcast vectors, typecase inputs, and generate the contextual runtime dict."""
    inputs = [tair, t_mrt, v_air, rh, m_activity, icl, age, mbody, ht]
    inputs = [np.atleast_1d(x).astype(float) for x in inputs]
    body_position_arr = np.atleast_1d(body_position).astype(str)
    sex_arr = np.atleast_1d(sex).astype(str)
    is_scalar = bool(len(inputs[0]) == 1)

    arrays = [*inputs, body_position_arr, sex_arr]
    max_len = max(len(a) for a in arrays)

    for a in arrays:
        if len(a) != 1 and len(a) != max_len:
            msg = (
                "Length of inputs differ! Single value or vectors of same "
                "length required."
            )
            raise ValueError(msg)

    ctx = {
        "tair": np.broadcast_to(inputs[0], max_len).copy(),
        "t_mrt": np.broadcast_to(inputs[1], max_len).copy(),
        "v_air": np.broadcast_to(inputs[2], max_len).copy(),
        "rh": np.broadcast_to(inputs[3], max_len).copy(),
        "m_activity": np.broadcast_to(inputs[4], max_len).copy(),
        "icl": np.broadcast_to(inputs[5], max_len).copy(),
        "age": np.broadcast_to(inputs[6], max_len).copy(),
        "mbody": np.broadcast_to(inputs[7], max_len).copy(),
        "ht": np.broadcast_to(inputs[8], max_len).copy(),
        "body_position": np.broadcast_to(body_position_arr, max_len).copy(),
        "sex": np.broadcast_to(sex_arr, max_len).copy(),
        "max_len": max_len,
    }
    ctx["adu"] = 0.203 * ctx["mbody"] ** 0.425 * ctx["ht"] ** 0.725
    return ctx, is_scalar


def _get_systemp_clothing(ctx: State) -> State:
    """Calculate clothing-related intermediate matrices based on input context."""
    icl = ctx["icl"].copy()
    icl[icl < ICL_MIN_INPUT] = ICL_MIN_VALUE
    fcl = 1.0 + (0.31 * icl)

    feff = np.full(ctx["max_len"], 0.725)
    feff[ctx["body_position"] == "sitting"] = 0.696
    feff[ctx["body_position"] == "crouching"] = 0.67

    facl = (173.51 * icl - 2.36 - 100.76 * icl**2 + 19.28 * icl**3) / 100.0
    facl[facl > 1.0] = 1.0
    rcl = icl / 6.45

    y = np.ones(ctx["max_len"])
    ht = ctx["ht"]
    y[(icl > ICL_UPPER_LIGHT) & (icl < ICL_UPPER_HEAVY)] = (
        ht[(icl > ICL_UPPER_LIGHT) & (icl < ICL_UPPER_HEAVY)] - 0.2
    ) / ht[(icl > ICL_UPPER_LIGHT) & (icl < ICL_UPPER_HEAVY)]
    y[(icl <= ICL_UPPER_LIGHT) & (icl > ICL_MID)] = 0.5
    y[(icl <= ICL_MID) & (icl > 0.0)] = 0.1

    adu = ctx["adu"]
    r2 = adu * (fcl - 1.0 + facl) / (6.28 * ht * y)
    r1 = facl * adu / (6.28 * ht * y)
    di = r2 - r1
    acl = adu * facl + adu * (fcl - 1.0)

    return {
        "feff": feff,
        "facl": facl,
        "fcl": fcl,
        "rcl": rcl,
        "y": y,
        "r1": r1,
        "r2": r2,
        "di": di,
        "acl": acl,
    }


def _get_systemp_metabolism(ctx: State) -> tuple[Dyn, Dyn]:
    """Determine initial human metabolic values."""
    age, mbody, ht, sex = ctx["age"], ctx["mbody"], ctx["ht"], ctx["sex"]

    met_base = (
        3.45
        * mbody**0.75
        * (
            1.0
            + 0.004 * (30.0 - age)
            + 0.01 * (ht * 100.0 / mbody ** (1.0 / 3.0) - 43.4)
        )
    )
    idx_f = sex == "female"
    if np.any(idx_f):
        met_base[idx_f] = (
            3.19
            * mbody[idx_f] ** 0.75
            * (
                1.0
                + 0.004 * (30.0 - age[idx_f])
                + 0.018 * (ht[idx_f] * 100.0 / mbody[idx_f] ** (1.0 / 3.0) - 42.1)
            )
        )

    he = ctx["m_activity"] + met_base
    return he, he * 1.0


def _populate_c_array(
    c: Dyn,
    h: Dyn,
    qresp: Dyn,
    tsk: Dyn,
    adu: Dyn,
    k_blood: Dyn,
    tsk_set: float,
) -> None:
    """Pre-computes coefficients for systemp arrays."""
    c[:, 0] = h + qresp
    c[:, 2] = tsk_set / 2.0 - 0.5 * tsk
    c[:, 3] = 5.28 * adu * c[:, 2]
    c[:, 4] = 13.0 / 625.0 * k_blood
    c[:, 5] = 0.76275 * k_blood
    c[:, 6] = c[:, 3] - c[:, 5] - tsk * c[:, 4]
    c[:, 7] = -c[:, 0] * c[:, 2] - tsk * c[:, 3] + tsk * c[:, 5]
    c[:, 9] = 5.28 * adu - 0.76275 * k_blood - 13.0 / 625.0 * k_blood * tsk
    c[:, 10] = c[:, 9] ** 2 - 4.0 * c[:, 4] * (
        c[:, 5] * tsk - c[:, 0] - 5.28 * adu * tsk
    )
    c[:, 8] = c[:, 6] ** 2 - 4.0 * c[:, 4] * c[:, 7]


def _compute_core_temps(
    c: Dyn,
    tcore: Dyn,
    adu: Dyn,
    k_blood: Dyn,
    tsk: Dyn,
    h: Dyn,
    qresp: Dyn,
    tsk_set: float,
) -> None:
    """Calculate internal body core temperature matrix constraints."""
    tcore[:, 6] = (h + qresp) / (5.28 * adu + k_blood * 6.3 / 3600.0) + tsk
    tcore[:, 2] = (h + qresp) / (
        5.28 * adu + k_blood * 6.3 / 3600.0 / (1.0 + 0.5 * (tsk_set - tsk))
    ) + tsk
    tcore[:, 3] = c[:, 0] / (5.28 * adu + k_blood * 1.0 / 40.0) + tsk

    idx_11 = c[:, 10] >= 0.0
    if np.any(idx_11):
        tcore[idx_11, 5] = (-c[idx_11, 9] - np.sqrt(c[idx_11, 10])) / (
            2.0 * c[idx_11, 4]
        )
        tcore[idx_11, 0] = (-c[idx_11, 9] + np.sqrt(c[idx_11, 10])) / (
            2.0 * c[idx_11, 4]
        )

    idx_9 = c[:, 8] >= 0.0
    if np.any(idx_9):
        tcore[idx_9, 1] = (-c[idx_9, 6] + np.sqrt(np.abs(c[idx_9, 8]))) / (
            2.0 * c[idx_9, 4]
        )
        tcore[idx_9, 4] = (-c[idx_9, 6] - np.sqrt(np.abs(c[idx_9, 8]))) / (
            2.0 * c[idx_9, 4]
        )


def _compute_sweat_and_enbal(
    j: int,
    tsk: Dyn,
    tcore: Dyn,
    csum: Dyn,
    rsum: Dyn,
    runtime: State,
    consts: PetConstants,
) -> tuple[Dyn, Dyn, Dyn]:
    """Calculate physiological sweating and systemic energy balances."""
    ctx = runtime["ctx"]
    adu = runtime["adu"]
    hc = runtime["hc"]
    vpa = runtime["vpa"]
    h = runtime["h"]
    qresp = runtime["qresp"]
    rcl = runtime["rcl"]

    tbody = 0.1 * tsk + 0.9 * tcore[:, j]
    swm = 304.94 * (tbody - consts.tbody_set) * adu / 3600000.0
    vpts = 6.11 * 10.0 ** (7.45 * tsk / (235.0 + tsk))

    swm[tbody <= consts.tbody_set] = 0.0
    swm[ctx["sex"] == "female"] = swm[ctx["sex"] == "female"] * 0.7

    esweat: Dyn = np.asarray(-swm * consts.lvap)
    hm = 0.633 * hc / (consts.p * consts.cair)
    fec = 1.0 / (1.0 + 0.92 * hc * rcl)
    emax = hm * (vpa - vpts) * adu * consts.lvap * fec
    wetsk = esweat / emax

    wetsk[wetsk > 1.0] = 1.0
    eswdif = esweat - emax

    esw = emax.copy()
    esw[eswdif > 0.0] = esweat[eswdif > 0.0]
    esw[esw > 0.0] = 0.0

    ed = consts.lvap / (consts.rdsk + consts.rdcl) * adu * (1.0 - wetsk) * (vpa - vpts)

    vb1, vb2 = consts.tsk_set - tsk, tcore[:, j] - consts.tc_set
    vb2[vb2 < 0.0], vb1[vb1 < 0.0] = 0.0, 0.0

    vb = (6.3 + 75.0 * vb2) / (1.0 + 0.5 * vb1)
    enbal = h + ed + qresp + esw + csum + rsum

    return esw, vb, enbal


def _systemp_100_loop(
    j: int,
    tsk: Dyn,
    tcl: Dyn,
    count1: int,
    enbal2: Dyn,
    matrices: State,
    runtime: State,
    consts: PetConstants,
) -> tuple[Dyn, Dyn, Dyn, Dyn, Dyn, bool]:
    """Execute inner iterative physics loop adjusting temperatures safely."""
    c = matrices["c"]
    tcore = matrices["tcore"]
    ctx = runtime["ctx"]
    clo = runtime["clo"]
    hc = runtime["hc"]
    h = runtime["h"]
    qresp = runtime["qresp"]
    vpa = runtime["vpa"]
    adu = runtime["adu"]

    xx = 0.001
    if count1 == 0:
        xx = 1.0
    elif count1 == 1:
        xx = 0.1
    elif count1 == COUNT_STAGE_TWO:
        xx = 0.01

    tair, t_mrt = ctx["tair"], ctx["t_mrt"]
    esw = np.zeros(ctx["max_len"])
    vb = np.zeros(ctx["max_len"])

    for _ in range(100):
        rclo2 = (
            consts.emcl
            * consts.sigm
            * ((tcl + 273.2) ** 4.0 - (t_mrt + 273.2) ** 4.0)
            * clo["feff"]
        )
        htcl = (6.28 * ctx["ht"] * clo["y"] * clo["di"]) / (
            clo["rcl"] * np.log(clo["r2"] / clo["r1"]) * clo["acl"]
        )
        tsk = (hc * (tcl - tair) + rclo2) / htcl + tcl

        aeffr = adu * clo["feff"]
        rbare = (
            aeffr
            * (1.0 - clo["facl"])
            * consts.emsk
            * consts.sigm
            * ((t_mrt + 273.2) ** 4.0 - (tsk + 273.2) ** 4.0)
        )
        rclo = (
            clo["feff"]
            * clo["acl"]
            * consts.emcl
            * consts.sigm
            * ((t_mrt + 273.2) ** 4.0 - (tcl + 273.2) ** 4.0)
        )

        csum = (hc * (tair - tsk) * adu * (1.0 - clo["facl"])) + (
            hc * (tair - tcl) * clo["acl"]
        )
        k_blood = adu * consts.rob * consts.cb

        _populate_c_array(c, h, qresp, tsk, adu, k_blood, consts.tsk_set)
        tsk[tsk == consts.tsk_set] = consts.tsk_set + 0.01

        _compute_core_temps(c, tcore, adu, k_blood, tsk, h, qresp, consts.tsk_set)
        loop_runtime: State = {
            "ctx": ctx,
            "adu": adu,
            "hc": hc,
            "vpa": vpa,
            "h": h,
            "qresp": qresp,
            "rcl": clo["rcl"],
        }
        esw, vb, enbal = _compute_sweat_and_enbal(
            j,
            tsk,
            tcore,
            csum,
            rbare + rclo,
            loop_runtime,
            consts,
        )

        tcl[enbal > 0.0] = tcl[enbal > 0.0] + xx
        tcl[enbal < 0.0] = tcl[enbal < 0.0] - xx

        if (np.any(enbal > 0.0) or np.any(enbal2 <= 0.0)) and (
            np.any(enbal < 0.0) or np.any(enbal2 >= 0.0)
        ):
            enbal2 = enbal.copy()
        else:
            return tsk, tcl, esw, vb, enbal2, True

    return tsk, tcl, esw, vb, enbal2, False


def _eval_stage_three(
    j_r: int,
    c: Dyn,
    tcore: Dyn,
    tsk: Dyn,
    consts: PetConstants,
    j: int,
) -> bool:
    """Evaluate specific validation markers mapping to J node requirements."""
    if j_r in (1, 6):
        return not (
            np.any(c[:, 10] < 0.0)
            or np.any(tcore[:, j] >= consts.tc_set)
            or np.any(tsk <= TSK_GUARD)
        )
    if j_r == J_REF_THREE:
        return not (
            np.any(tcore[:, j] >= consts.tc_set) or np.any(tsk > consts.tsk_set)
        )
    if j_r == J_REF_SEVEN:
        return not (
            np.any(tcore[:, j] >= consts.tc_set) or np.any(tsk <= consts.tsk_set)
        )
    return j_r == J_REF_FOUR


def _evaluate_g100(
    j_r: int,
    count1: int,
    c: Dyn,
    tcore: Dyn,
    tsk: Dyn,
    consts: PetConstants,
    j: int,
) -> bool:
    """Evaluate g100 condition to proceed or skip to valid roots."""
    if count1 == COUNT_STAGE_THREE and j_r not in (J_REF_TWO, J_REF_FIVE):
        return _eval_stage_three(j_r, c, tcore, tsk, consts, j)

    if (
        np.any(c[:, 8] < 0.0)
        or np.any(tcore[:, j] < consts.tc_set)
        or np.any(tsk > consts.tsk_set + 0.05)
    ):
        return False

    return False


def _systemp_j_iteration(
    j_r: int,
    ctx: State,
    clo: State,
    consts: PetConstants,
    hc: Dyn,
    h: Dyn,
    qresp: Dyn,
    vpa: Dyn,
    adu: Dyn,
    c: Dyn,
    tcore: Dyn,
) -> State | None:
    """Outer cycle iteration mapping node options returning physiological targets."""
    j = j_r - 1
    tsk = np.full(ctx["max_len"], consts.tsk_set)
    tcl = (ctx["tair"] + ctx["t_mrt"] + tsk) / 3.0
    enbal2 = np.zeros(ctx["max_len"])
    count1, esw, vb = 0, None, None

    while True:
        matrices: State = {"c": c, "tcore": tcore}
        runtime: State = {
            "ctx": ctx,
            "clo": clo,
            "hc": hc,
            "h": h,
            "qresp": qresp,
            "vpa": vpa,
            "adu": adu,
        }
        tsk, tcl, esw, vb, enbal2, _ = _systemp_100_loop(
            j,
            tsk,
            tcl,
            count1,
            enbal2,
            matrices,
            runtime,
            consts,
        )
        if count1 < COUNT_STAGE_THREE:
            count1 += 1
            enbal2 = np.zeros(ctx["max_len"])
        else:
            break

    if not _evaluate_g100(j_r, count1, c, tcore, tsk, consts, j):
        return None

    if (j_r == J_REF_FOUR or np.any(vb < VB_UPPER_LIMIT)) and (
        j_r != J_REF_FOUR or np.any(vb >= VB_LOWER_LIMIT)
    ):
        vb[vb > VB_CAP] = VB_CAP
        return {"tc": tcore[:, j], "tsk": tsk, "tcl": tcl, "esw": esw}

    index = J_REF_FOUR if (j_r - J_REF_THREE) < 1 else j_r - J_REF_THREE
    return {"tc": tcore[:, index - 1], "tsk": tsk, "tcl": tcl, "esw": esw}


def _run_systemp(ctx: State, consts: PetConstants) -> State | None:
    """Coordinates base components evaluating the environmental interactions loop."""
    clo = _get_systemp_clothing(ctx)
    he, h = _get_systemp_metabolism(ctx)

    vpa = ctx["rh"] / 100.0 * svp_murray(c2k(ctx["tair"]))
    texp = 0.47 * ctx["tair"] + 21.0
    rtv = he * 1.44 * 10.0**-6.0
    cres = consts.cair * (ctx["tair"] - texp) * rtv
    vpexp = 6.11 * 10.0 ** (7.45 * texp / (235.0 + texp))
    qresp = cres + (0.623 * consts.lvap / consts.p * (vpa - vpexp) * rtv)

    c = np.zeros((ctx["max_len"], 11))
    tcore = np.zeros((ctx["max_len"], 7))
    hc = (2.67 + 6.5 * ctx["v_air"] ** 0.67) * ((consts.p / consts.po) ** 0.55)

    for j_r in range(1, 8):
        res = _systemp_j_iteration(
            j_r,
            ctx,
            clo,
            consts,
            hc,
            h,
            qresp,
            vpa,
            ctx["adu"],
            c,
            tcore,
        )
        if res is not None:
            return res

    warnings.warn("Should not arrive here p.1", stacklevel=2)
    return None


def _pet_prep_reference(
    tc: Dyn,
    tsk: Dyn,
    ctx: State,
    consts: PetConstants,
    esw_real: Dyn,
) -> State:
    """Generate constants required for iterating standard references."""
    icl_ref, m_activity_ref, v_air_ref, vpa_ref = 0.9, 80.0, 0.1, 12.0
    tbody = 0.1 * tsk + 0.9 * tc

    v_base = (
        3.45
        * ctx["mbody"] ** 0.75
        * (
            1.0
            + 0.004 * (30.0 - ctx["age"])
            + 0.01 * (ctx["ht"] * 100.0 / ctx["mbody"] ** (1.0 / 3.0) - 43.4)
        )
    )
    met_base = v_base.copy()

    idx = np.nonzero(ctx["sex"] == "male")[0]
    if len(idx) > 0:
        met_base[idx] = np.resize(v_base, len(idx))

    rtv_ref = (m_activity_ref + met_base) * 1.44 * 10.0**-6.0
    swm = 304.94 * (tbody - consts.tbody_set) * ctx["adu"] / 3600000.0
    vpts = 6.11 * 10.0 ** (7.45 * tsk / (235.0 + tsk))

    swm[tbody <= consts.tbody_set] = 0.0
    swm[ctx["sex"] == "female"] = swm[ctx["sex"] == "female"] * 0.7

    esweat: Dyn = esw_real.copy()
    hc = (2.67 + 6.5 * v_air_ref**0.67) * ((consts.p / consts.po) ** 0.55)

    feff_val = 0.725
    aeffr = ctx["adu"] * feff_val

    facl = min(
        (173.51 * icl_ref - 2.36 - 100.76 * icl_ref**2 + 19.28 * icl_ref**3) / 100.0,
        1.0,
    )
    acl = ctx["adu"] * facl + ctx["adu"] * ((1.0 + 0.31 * icl_ref) - 1.0)

    emax: Dyn = np.asarray(
        (0.633 * hc / (consts.p * consts.cair))
        * (vpa_ref - vpts)
        * ctx["adu"]
        * consts.lvap
        * (1.0 / (1.0 + 0.92 * hc * 0.155 * icl_ref)),
    )
    wetsk = esweat / emax

    wetsk[wetsk > 1.0] = 1.0
    eswdif = esweat - emax
    ediff = (
        consts.lvap
        / (consts.rdsk + consts.rdcl)
        * ctx["adu"]
        * (1.0 - wetsk)
        * (vpa_ref - vpts)
    )

    esw = emax.copy()
    esw[eswdif > 0.0] = esweat[eswdif > 0.0]
    esw[esw > 0.0] = 0.0

    return {
        "rtv_ref": rtv_ref,
        "hc": hc,
        "aeffr": aeffr,
        "facl": facl,
        "acl": acl,
        "ediff": ediff,
        "esw": esw,
        "met_base": met_base,
        "m_activity_ref": m_activity_ref,
        "vpa_ref": vpa_ref,
        "feff_val": feff_val,
    }


def _run_pet(
    tc: Dyn,
    tsk: Dyn,
    tcl: Dyn,
    ctx: State,
    consts: PetConstants,
    esw_real: Dyn,
) -> Dyn:
    """Compute equivalent operational temperatures for PET convergence."""
    ref = _pet_prep_reference(tc, tsk, ctx, consts, esw_real)
    tx = ctx["tair"].copy()
    enbal2 = np.zeros(ctx["max_len"])
    count1 = 0
    xx = np.full(ctx["max_len"], 5.0)

    while count1 != PET_ITERATION_LIMIT:
        rsum = ref["aeffr"] * (1.0 - ref["facl"]) * consts.emsk * consts.sigm * (
            (tx + 273.2) ** 4.0 - (tsk + 273.2) ** 4.0
        ) + ref["feff_val"] * ref["acl"] * consts.emcl * consts.sigm * (
            (tx + 273.2) ** 4.0 - (tcl + 273.2) ** 4.0
        )
        csum = (ref["hc"] * (tx - tsk) * ctx["adu"] * (1.0 - ref["facl"])) + (
            ref["hc"] * (tx - tcl) * ref["acl"]
        )
        texp = 0.47 * tx + 21.0

        vpexp = 6.11 * 10.0 ** (7.45 * texp / (235.0 + texp))
        qresp = (consts.cair * (tx - texp) * ref["rtv_ref"]) + (
            0.623 * consts.lvap / consts.p * (ref["vpa_ref"] - vpexp) * ref["rtv_ref"]
        )

        enbal = (
            (ref["m_activity_ref"] + ref["met_base"])
            + ref["ediff"]
            + qresp
            + ref["esw"]
            + csum
            + rsum
        )

        tx[enbal > 0.0] = tx[enbal > 0.0] - xx[enbal > 0.0]
        tx[enbal <= 0.0] = tx[enbal <= 0.0] + xx[enbal <= 0.0]

        bitmask2 = ((enbal > 0.0) & (enbal2 >= 0.0)) | ((enbal < 0.0) & (enbal2 <= 0.0))
        enbal2[bitmask2] = enbal[bitmask2]
        xx[~bitmask2] = xx[~bitmask2] / 5.0

        count1 += 1
        if np.sum(np.abs(enbal) > PET_ENERGY_TOLERANCE) == 0:
            break

    return tx


def pet_corrected(
    tair: object = 21.0,
    t_mrt: object = 21.0,
    v_air: object = 0.1,
    rh: object = 50.0,
    m_activity: object = 80.0,
    icl: object = 0.9,
    body_position: object = "standing",
    sex: object = "male",
    age: object = 35.0,
    mbody: object = 75.0,
    ht: object = 1.80,
) -> object | None:
    """Compute PET for weather and body inputs."""
    ctx, is_scalar = _prepare_context(
        tair,
        t_mrt,
        v_air,
        rh,
        m_activity,
        icl,
        body_position,
        sex,
        age,
        mbody,
        ht,
    )
    consts = PetConstants()

    s = _run_systemp(ctx, consts)
    if s is None:
        return None

    pet_res = _run_pet(s["tc"], s["tsk"], s["tcl"], ctx, consts, s["esw"])

    if is_scalar:
        return float(np.asarray(pet_res, dtype="float64").reshape(-1)[0])
    return pet_res
