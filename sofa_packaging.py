import io
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

# =========================
# 규칙(고정값)
# =========================
ANGLE_W = 70
ANGLE_D = 70
LEAVE_H = 50  # 상부 덮개 끝에서 50mm 남김(높이) - 고정

# 컨테이너 내부 치수 기본값(mm) - 필요 시 UI에서 수정 가능하게 둠
CONTAINERS = {
    "20ft(표준)": {"L": 5900, "W": 2352, "H": 2395},
    "40ft High Cube": {"L": 12032, "W": 2350, "H": 2700},
}

# =========================
# 파라미터(기본값) - 세션 저장
# =========================
DEFAULT_PARAMS = {
    # 내측 -> 외측 증가값 (박스 두께/형태별)
    "BOX_ADD": {
        "DW": {"add_wd": 10, "add_h": 20},
        "SW": {"add_wd": 6,  "add_h": 12},
    },

    # 수출형 하부 외측 높이 증가값
    "LOWER_OUT_H_MODE": "HALF_OF_A1",   # "HALF_OF_A1" 또는 "CUSTOM"
    "LOWER_OUT_H_CUSTOM": {"DW": 10, "SW": 6},

    # ✅ 상부 내측 W/D 여유: "하부 외측 W/D"에서 더해지는 값 (기본 10)
    "UPPER_IN_WD_EXTRA_FROM_LOWER_OUT": 10,

    # 상부 외측 증가값 (상부 내측 -> 상부 외측)
    "UPPER_OUT_WD_ADD": 10,
    "UPPER_OUT_H_ADD": 10,

    # ✅ 컨테이너 시뮬레이션: 적재 방향 허용(기본은 평면 회전만)
    "SIM_ALLOW_PLANAR_ROTATE": True,
    "SIM_ALLOW_STAND_WIDE_SIDE": False,   # 넓은쪽 측면(가로가 높이로 올라감)
    "SIM_ALLOW_STAND_NARROW_SIDE": False, # 좁은쪽 측면(세로가 높이로 올라감)
}

def get_params():
    """세션에 파라미터가 없으면 기본값으로 초기화"""
    if "PARAMS" not in st.session_state:
        st.session_state["PARAMS"] = DEFAULT_PARAMS.copy()
        st.session_state["PARAMS"]["BOX_ADD"] = {k: v.copy() for k, v in DEFAULT_PARAMS["BOX_ADD"].items()}
        st.session_state["PARAMS"]["LOWER_OUT_H_CUSTOM"] = DEFAULT_PARAMS["LOWER_OUT_H_CUSTOM"].copy()
    return st.session_state["PARAMS"]

# =========================
# 계산 로직
# =========================
def calc_all(base_w, base_d, base_h, mw, md, mh, box_type, params):
    box_type = str(box_type).strip().upper()
    if box_type not in params["BOX_ADD"]:
        raise ValueError("박스형태는 DW 또는 SW여야 합니다.")

    add_wd = int(params["BOX_ADD"][box_type]["add_wd"])
    add_h = int(params["BOX_ADD"][box_type]["add_h"])

    # A1 내측 = 제품 내측 + 여유치
    a1_in_w = int(round(base_w + mw))
    a1_in_d = int(round(base_d + md))
    a1_in_h = int(round(base_h + mh))

    # A1 외측
    a1_out_w = a1_in_w + add_wd
    a1_out_d = a1_in_d + add_wd
    a1_out_h = a1_in_h + add_h

    # 수출형 하부: 내측은 A1과 동일
    lower_in_w, lower_in_d, lower_in_h = a1_in_w, a1_in_d, a1_in_h
    lower_out_w, lower_out_d = a1_out_w, a1_out_d

    # 하부 외측 높이
    if params.get("LOWER_OUT_H_MODE", "HALF_OF_A1") == "CUSTOM":
        lower_out_h = lower_in_h + int(params["LOWER_OUT_H_CUSTOM"][box_type])
    else:
        lower_out_h = lower_in_h + int(round(add_h / 2))

    # 수출형 상부: 높이 -50mm 고정
    upper_in_h = a1_in_h - LEAVE_H
    if upper_in_h <= 0:
        upper_in_h = 1

    # ✅ 상부 내측 W/D = 하부 외측 W/D + 여유값(기본 10)
    upper_extra = int(params.get("UPPER_IN_WD_EXTRA_FROM_LOWER_OUT", 10))
    upper_in_w = lower_out_w + upper_extra
    upper_in_d = lower_out_d + upper_extra

    # 상부 외측 = 상부 내측 + 증가값
    upper_out_w = upper_in_w + int(params.get("UPPER_OUT_WD_ADD", 10))
    upper_out_d = upper_in_d + int(params.get("UPPER_OUT_WD_ADD", 10))
    upper_out_h = upper_in_h + int(params.get("UPPER_OUT_H_ADD", 10))

    # 보강재
    side_pad_qty = 2
    side_pad_w = lower_in_w
    side_pad_h = lower_in_h

    top_pad_qty = 1
    top_pad_w = upper_out_w
    top_pad_d = upper_out_d

    angle_qty = 4
    angle_h = lower_in_h

    return {
        "박스형태": box_type,
        "여유가로": mw, "여유세로": md, "여유높이": mh,

        "A1_내측가로": a1_in_w, "A1_내측세로": a1_in_d, "A1_내측높이": a1_in_h,
        "A1_외측가로": a1_out_w, "A1_외측세로": a1_out_d, "A1_외측높이": a1_out_h,

        "하부_내측가로": lower_in_w, "하부_내측세로": lower_in_d, "하부_내측높이": lower_in_h,
        "하부_외측가로": lower_out_w, "하부_외측세로": lower_out_d, "하부_외측높이": lower_out_h,

        "상부_내측가로": upper_in_w, "상부_내측세로": upper_in_d, "상부_내측높이": upper_in_h,
        "상부_외측가로": upper_out_w, "상부_외측세로": upper_out_d, "상부_외측높이": upper_out_h,

        "측면패드_수량": side_pad_qty,
        "측면패드_규격(WxH)": f"{side_pad_w}x{side_pad_h}",

        "상부패드_수량": top_pad_qty,
        "상부패드_규격(WxD)": f"{top_pad_w}x{top_pad_d}",

        "앵글_수량": angle_qty,
        "앵글_규격(70x70xH)": f"{ANGLE_W}x{ANGLE_D}x{angle_h}",

        "파라미터_상부내측_WD여유(하부외측기준)": upper_extra,
    }

# =========================
# Plotly 3D 도형 유틸
# =========================
def _cuboid_vertices(x0, y0, z0, dx, dy, dz):
    x = [x0, x0+dx, x0+dx, x0,    x0, x0+dx, x0+dx, x0]
    y = [y0, y0,    y0+dy, y0+dy, y0, y0,    y0+dy, y0+dy]
    z = [z0, z0,    z0,    z0,    z0+dz, z0+dz, z0+dz, z0+dz]
    return np.array(x), np.array(y), np.array(z)

def _cuboid_mesh(name, x0, y0, z0, dx, dy, dz, opacity=0.35, color="#888", wire=False, showlegend=True):
    x, y, z = _cuboid_vertices(x0, y0, z0, dx, dy, dz)

    i = [0,0,0, 1,1,2, 4,4,4, 5,5,6]
    j = [1,2,3, 2,5,6, 5,6,7, 6,1,2]
    k = [2,3,0, 5,6,7, 6,7,4, 1,2,3]

    mesh = go.Mesh3d(
        x=x, y=y, z=z,
        i=i, j=j, k=k,
        name=name,
        opacity=opacity,
        color=color,
        flatshading=True,
        showlegend=showlegend,
        hovertemplate=f"{name}<extra></extra>"
    )
    traces = [mesh]

    if wire:
        edges = [
            (0,1),(1,2),(2,3),(3,0),
            (4,5),(5,6),(6,7),(7,4),
            (0,4),(1,5),(2,6),(3,7)
        ]
        xs, ys, zs = [], [], []
        for a,b in edges:
            xs += [x[a], x[b], None]
            ys += [y[a], y[b], None]
            zs += [z[a], z[b], None]
        traces.append(go.Scatter3d(
            x=xs, y=ys, z=zs,
            mode="lines",
            line=dict(width=3, color="rgba(0,0,0,0.35)"),
            showlegend=False,
            hoverinfo="skip"
        ))
    return traces

# =========================
# 3D 도면: A1 / 수출형
# =========================
def build_3d_figure_a1(res):
    w = res["A1_외측가로"]
    d = res["A1_외측세로"]
    h = res["A1_외측높이"]

    fig = go.Figure()
    for t in _cuboid_mesh("A1 박스(외측)", 0, 0, 0, w, d, h, opacity=0.35, color="#9C6B3E", wire=True):
        fig.add_trace(t)

    fig.update_layout(
        height=450,
        margin=dict(l=0, r=0, t=30, b=0),
        scene=dict(xaxis_title="가로(W)", yaxis_title="세로(D)", zaxis_title="높이(H)", aspectmode="data"),
    )
    return fig

def build_3d_figure_export(res):
    lw = res["하부_외측가로"]
    ld = res["하부_외측세로"]
    lh = res["하부_외측높이"]

    uw = res["상부_외측가로"]
    ud = res["상부_외측세로"]
    uh = res["상부_외측높이"]

    gap = max(int(round(lh * 0.15)), 60)

    lower_x0, lower_y0, lower_z0 = 0, 0, 0

    pad_t = 20
    pad_w = res["하부_내측가로"]
    pad_h = res["하부_내측높이"]
    pad_z0 = lower_z0 + gap
    pad_x0 = (lw - pad_w) / 2
    pad_front_y0 = 10
    pad_back_y0 = ld - pad_t - 10

    ang_h = res["하부_내측높이"]
    angles = [
        (0, 0),
        (lw-ANGLE_W, 0),
        (0, ld-ANGLE_D),
        (lw-ANGLE_W, ld-ANGLE_D),
    ]

    upper_z0 = lower_z0 + lh + gap * 2

    top_pad_th = 10
    top_pad_z0 = upper_z0 + uh + gap

    fig = go.Figure()

    for t in _cuboid_mesh("하부 박스(외측)", lower_x0, lower_y0, lower_z0, lw, ld, lh, opacity=0.35, color="#A67C52", wire=True):
        fig.add_trace(t)

    for t in _cuboid_mesh("측면 패드(앞)", pad_x0, pad_front_y0, pad_z0, pad_w, pad_t, pad_h, opacity=0.55, color="#2E6BE6", wire=True):
        fig.add_trace(t)
    for t in _cuboid_mesh("측면 패드(뒤)", pad_x0, pad_back_y0, pad_z0, pad_w, pad_t, pad_h, opacity=0.55, color="#2E6BE6", wire=True):
        fig.add_trace(t)

    for idx, (ax0, ay0) in enumerate(angles, start=1):
        for t in _cuboid_mesh(f"앵글({idx}) 70x70", ax0, ay0, 0, ANGLE_W, ANGLE_D, ang_h, opacity=0.55, color="#F2C94C", wire=True):
            fig.add_trace(t)

    for t in _cuboid_mesh("상부 박스(외측)", 0, 0, upper_z0, uw, ud, uh, opacity=0.35, color="#B08968", wire=True):
        fig.add_trace(t)

    for t in _cuboid_mesh("상부 패드(외측 W×D)", 0, 0, top_pad_z0, uw, ud, top_pad_th, opacity=0.6, color="#EB5757", wire=True):
        fig.add_trace(t)

    fig.update_layout(
        height=650,
        margin=dict(l=0, r=0, t=30, b=0),
        scene=dict(xaxis_title="가로(W)", yaxis_title="세로(D)", zaxis_title="높이(H)", aspectmode="data"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0.01)
    )
    return fig

# =========================
# 컨테이너 적재 시뮬레이션
# - 기본: 평면 회전만
# - 옵션: 넓은쪽/좁은쪽 측면 세워 적재 허용
# =========================
def get_export_package_footprint(res):
    pack_w = max(res["하부_외측가로"], res["상부_외측가로"])
    pack_d = max(res["하부_외측세로"], res["상부_외측세로"])
    default_h = max(res["하부_외측높이"], res["상부_외측높이"])
    return int(pack_w), int(pack_d), int(default_h)

def compute_packing(container_L, container_W, container_H,
                    pack_w, pack_d, pack_h,
                    gap_xy=10, gap_z=0, margin_L=0, margin_W=0, margin_H=0,
                    allow_planar_rotate=True,
                    allow_stand_wide_side=False,
                    allow_stand_narrow_side=False):
    """
    후보 방향:
    1) 기본(눕힘): base = (w,d), height = h
    2) 넓은쪽 측면 세우기: height = w, base = (h,d)
    3) 좁은쪽 측면 세우기: height = d, base = (w,h)

    각 후보에서 평면 회전(가로↔세로) 허용 여부 적용.
    """
    def fit(L, W, H, a, b, c):
        L_eff = max(L - margin_L, 0)
        W_eff = max(W - margin_W, 0)
        H_eff = max(H - margin_H, 0)
        nx = int(np.floor((L_eff + gap_xy) / (a + gap_xy))) if (a + gap_xy) > 0 else 0
        ny = int(np.floor((W_eff + gap_xy) / (b + gap_xy))) if (b + gap_xy) > 0 else 0
        nz = int(np.floor((H_eff + gap_z)  / (c + gap_z)))  if (c + gap_z) > 0 else 0
        return nx, ny, nz, nx*ny*nz

    candidates = []

    def add_candidate(label, a, b, c):
        nx, ny, nz, total = fit(container_L, container_W, container_H, a, b, c)
        candidates.append((label, a, b, c, nx, ny, nz, total))
        if allow_planar_rotate and a != b:
            nx2, ny2, nz2, total2 = fit(container_L, container_W, container_H, b, a, c)
            candidates.append((label + " + 평면회전", b, a, c, nx2, ny2, nz2, total2))

    # 1) 기본(눕힘)
    add_candidate("기본(눕힘): base=W×D, H=H", pack_w, pack_d, pack_h)

    # 2) 넓은쪽 측면 세우기 (height=w, base=h×d)
    if allow_stand_wide_side:
        add_candidate("세움(넓은쪽측면): base=H×D, H=W", pack_h, pack_d, pack_w)

    # 3) 좁은쪽 측면 세우기 (height=d, base=w×h)
    if allow_stand_narrow_side:
        add_candidate("세움(좁은쪽측면): base=W×H, H=D", pack_w, pack_h, pack_d)

    best = max(candidates, key=lambda x: x[-1]) if candidates else None
    return best, candidates

def build_container_3d(container_L, container_W, container_H,
                       unit_L, unit_W, unit_H,
                       nx, ny, nz,
                       gap_xy=10, gap_z=0,
                       margin_L=0, margin_W=0, margin_H=0,
                       max_display=200):
    fig = go.Figure()

    for t in _cuboid_mesh("컨테이너(내부)", 0, 0, 0, container_L, container_W, container_H,
                         opacity=0.05, color="#999", wire=True, showlegend=True):
        fig.add_trace(t)

    x0 = margin_L / 2
    y0 = margin_W / 2
    z0 = margin_H / 2

    drawn = 0
    for k in range(nz):
        for j in range(ny):
            for i in range(nx):
                if drawn >= max_display:
                    break
                x = x0 + i * (unit_L + gap_xy)
                y = y0 + j * (unit_W + gap_xy)
                z = z0 + k * (unit_H + gap_z)

                if x + unit_L > container_L or y + unit_W > container_W or z + unit_H > container_H:
                    continue

                showleg = (drawn == 0)
                for t in _cuboid_mesh("적재 박스", x, y, z, unit_L, unit_W, unit_H,
                                     opacity=0.25, color="#4A90E2", wire=False, showlegend=showleg):
                    fig.add_trace(t)
                drawn += 1
            if drawn >= max_display:
                break
        if drawn >= max_display:
            break

    fig.update_layout(
        height=700,
        margin=dict(l=0, r=0, t=30, b=0),
        scene=dict(xaxis_title="컨테이너 길이(L)", yaxis_title="컨테이너 폭(W)", zaxis_title="컨테이너 높이(H)", aspectmode="data"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0.01)
    )
    return fig, drawn

# =========================
# 엑셀(대량 업로드/다운로드)
# =========================
REQ_COLS = [
    "제품코드", "제품명",
    "내측가로", "내측세로", "내측높이",
    "박스형태", "여유가로", "여유세로", "여유높이"
]

def build_template_excel_bytes():
    df = pd.DataFrame(columns=REQ_COLS)
    df.loc[0] = ["EX001", "소파A", 2000, 900, 800, "DW", 0, 0, 0]
    out = io.BytesIO()
    with pd.ExcelWriter(out, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="입력")
    out.seek(0)
    return out.getvalue()

def process_bulk_excel(uploaded_file, params):
    df = pd.read_excel(uploaded_file)
    missing = [c for c in REQ_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"업로드 엑셀에 필요한 컬럼이 없습니다: {missing}")

    results = []
    for _, row in df.iterrows():
        r = calc_all(
            row["내측가로"], row["내측세로"], row["내측높이"],
            row["여유가로"], row["여유세로"], row["여유높이"],
            str(row["박스형태"]).strip().upper(),
            params
        )
        r["제품코드"] = row["제품코드"]
        r["제품명"] = row["제품명"]
        results.append(r)

    out_df = pd.DataFrame(results)
    front = ["제품코드", "제품명"]
    cols = front + [c for c in out_df.columns if c not in front]
    out_df = out_df[cols]

    out = io.BytesIO()
    with pd.ExcelWriter(out, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="원본")
        out_df.to_excel(writer, index=False, sheet_name="결과")
    out.seek(0)
    return out_df, out.getvalue()

# =========================
# Streamlit UI
# =========================
st.set_page_config(page_title="소파 포장 규격 자동 산정", layout="wide")
st.title("소파 포장 규격 자동 산정 (A1 도면 + 수출형 3D 분해도 + 엑셀 대량 업로드 + 컨테이너 적재 시뮬레이션)")

params = get_params()

tab1, tab2, tab3, tab4 = st.tabs(
    ["단건 계산(3D 도해)", "엑셀 대량 업로드", "컨테이너 적재 시뮬레이션(3D)", "파라미터 설정"]
)

# ---- 탭4: 파라미터 설정 ----
with tab4:
    st.subheader("파라미터 설정")
    st.caption("여기서 설정한 값은 앱 전체(단건/대량/컨테이너)에 공통 적용됩니다. 상부 높이 -50mm(LEAVE_H=50)는 고정입니다.")

    c1, c2 = st.columns(2, gap="large")

    with c1:
        st.markdown("### 1) 내측 → 외측 증가값(박스형태별)")
        dw_add_wd = st.number_input("DW: WD 증가(mm)", min_value=0, value=int(params["BOX_ADD"]["DW"]["add_wd"]), step=1)
        dw_add_h  = st.number_input("DW: H 증가(mm)",  min_value=0, value=int(params["BOX_ADD"]["DW"]["add_h"]),  step=1)
        sw_add_wd = st.number_input("SW: WD 증가(mm)", min_value=0, value=int(params["BOX_ADD"]["SW"]["add_wd"]), step=1)
        sw_add_h  = st.number_input("SW: H 증가(mm)",  min_value=0, value=int(params["BOX_ADD"]["SW"]["add_h"]),  step=1)

        st.markdown("### 2) 상부 내측 W/D 여유(하부 외측 기준)")
        upper_extra = st.number_input(
            "상부 내측 W/D 여유(mm) [기본 10]",
            min_value=0,
            value=int(params.get("UPPER_IN_WD_EXTRA_FROM_LOWER_OUT", 10)),
            step=1,
            help="상부 내측 W/D = 하부 외측 W/D + 이 값"
        )

    with c2:
        st.markdown("### 3) 상부 외측 증가값")
        upper_out_wd_add = st.number_input("상부 외측 WD 증가(mm)", min_value=0, value=int(params.get("UPPER_OUT_WD_ADD", 10)), step=1)
        upper_out_h_add  = st.number_input("상부 외측 H 증가(mm)",  min_value=0, value=int(params.get("UPPER_OUT_H_ADD", 10)),  step=1)

        st.markdown("### 4) 하부 외측 높이 증가값")
        mode = st.radio(
            "하부 외측 높이 계산 방식",
            options=["HALF_OF_A1", "CUSTOM"],
            horizontal=True,
            index=0 if params.get("LOWER_OUT_H_MODE", "HALF_OF_A1") == "HALF_OF_A1" else 1
        )
        lower_custom_dw = st.number_input("CUSTOM(DW): 하부 외측 높이 추가(mm)", min_value=0, value=int(params["LOWER_OUT_H_CUSTOM"]["DW"]), step=1)
        lower_custom_sw = st.number_input("CUSTOM(SW): 하부 외측 높이 추가(mm)", min_value=0, value=int(params["LOWER_OUT_H_CUSTOM"]["SW"]), step=1)

        st.markdown("### 5) 컨테이너 적재 방향 허용")
        sim_allow_planar = st.checkbox("평면 회전 허용(가로↔세로)", value=bool(params.get("SIM_ALLOW_PLANAR_ROTATE", True)))
        sim_allow_wide = st.checkbox("넓은쪽 측면으로 세우기 허용", value=bool(params.get("SIM_ALLOW_STAND_WIDE_SIDE", False)))
        sim_allow_narrow = st.checkbox("좁은쪽 측면으로 세우기 허용", value=bool(params.get("SIM_ALLOW_STAND_NARROW_SIDE", False)))
        st.caption("기본값은 평면 회전만 허용(세우기 2개 옵션은 꺼짐)")

    b1, b2 = st.columns([1, 1], gap="large")
    with b1:
        if st.button("파라미터 저장", type="primary"):
            params["BOX_ADD"]["DW"]["add_wd"] = int(dw_add_wd)
            params["BOX_ADD"]["DW"]["add_h"] = int(dw_add_h)
            params["BOX_ADD"]["SW"]["add_wd"] = int(sw_add_wd)
            params["BOX_ADD"]["SW"]["add_h"] = int(sw_add_h)

            params["UPPER_IN_WD_EXTRA_FROM_LOWER_OUT"] = int(upper_extra)
            params["UPPER_OUT_WD_ADD"] = int(upper_out_wd_add)
            params["UPPER_OUT_H_ADD"] = int(upper_out_h_add)

            params["LOWER_OUT_H_MODE"] = mode
            params["LOWER_OUT_H_CUSTOM"]["DW"] = int(lower_custom_dw)
            params["LOWER_OUT_H_CUSTOM"]["SW"] = int(lower_custom_sw)

            params["SIM_ALLOW_PLANAR_ROTATE"] = bool(sim_allow_planar)
            params["SIM_ALLOW_STAND_WIDE_SIDE"] = bool(sim_allow_wide)
            params["SIM_ALLOW_STAND_NARROW_SIDE"] = bool(sim_allow_narrow)

            st.session_state["PARAMS"] = params
            st.success("저장 완료! (단건/대량/컨테이너 계산에 즉시 적용됩니다.)")

    with b2:
        if st.button("기본값으로 초기화"):
            st.session_state.pop("PARAMS", None)
            params = get_params()
            st.warning("기본값으로 초기화했습니다.")

# ---- 탭1: 단건 계산 ----
with tab1:
    left, right = st.columns([1, 2], gap="large")

    with left:
        st.subheader("입력(한 건)")
        제품코드 = st.text_input("제품코드", value="EX001")
        제품명 = st.text_input("제품명", value="소파A")

        base_w = st.number_input("내측가로", min_value=1, value=2000, step=10)
        base_d = st.number_input("내측세로", min_value=1, value=900, step=10)
        base_h = st.number_input("내측높이", min_value=1, value=800, step=10)

        box_type = st.radio("박스형태", options=["DW", "SW"], horizontal=True, index=0)

        # ✅ 초기값 0으로 변경
        mw = st.number_input("여유가로", min_value=0, value=0, step=5)
        md = st.number_input("여유세로", min_value=0, value=0, step=5)
        mh = st.number_input("여유높이", min_value=0, value=0, step=5)

        run = st.button("계산 실행", type="primary")

    with right:
        if not run:
            st.info("왼쪽 값을 입력하고 **계산 실행**을 누르면, A1 3D 도면 + 수출형 3D 분해도 + 결과표가 표시됩니다.")
        else:
            res = calc_all(base_w, base_d, base_h, mw, md, mh, box_type, params)
            res = {"제품코드": 제품코드, "제품명": 제품명, **res}
            show_df = pd.DataFrame([res])

            st.subheader("결과(내측/외측 포함)")
            st.dataframe(show_df, use_container_width=True)

            st.subheader("A1 박스 3D 도면(외측)")
            st.plotly_chart(build_3d_figure_a1(res), use_container_width=True)

            st.subheader("수출형 3D 분해도(측면 패드=넓은면 W×H)")
            st.plotly_chart(build_3d_figure_export(res), use_container_width=True)

            out = io.BytesIO()
            with pd.ExcelWriter(out, engine="openpyxl") as writer:
                show_df.to_excel(writer, index=False, sheet_name="결과")
            out.seek(0)
            st.download_button(
                "단건 결과 엑셀 다운로드(.xlsx)",
                data=out.getvalue(),
                file_name=f"{제품코드}_포장규격_결과.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )

# ---- 탭2: 대량 업로드 ----
with tab2:
    st.subheader("엑셀 대량 업로드 → 결과 엑셀 다운로드")

    col_a, col_b = st.columns([1, 2], gap="large")

    with col_a:
        st.markdown("### 1) 입력 엑셀 템플릿")
        st.download_button(
            "입력 템플릿 다운로드(.xlsx)",
            data=build_template_excel_bytes(),
            file_name="소파포장_입력템플릿.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )

        st.markdown("### 2) 파일 업로드")
        up = st.file_uploader("엑셀 파일 업로드(.xlsx)", type=["xlsx"])
        st.caption("필수 컬럼(한글): 제품코드, 제품명, 내측가로/세로/높이, 박스형태(DW/SW), 여유가로/세로/높이")

    with col_b:
        if up is None:
            st.info("왼쪽 템플릿 구조와 동일한 엑셀을 업로드하면 결과를 생성합니다.")
        else:
            try:
                out_df, out_bytes = process_bulk_excel(up, params)
                st.success(f"처리 완료: {len(out_df)}건")
                st.dataframe(out_df, use_container_width=True, height=420)

                st.download_button(
                    "결과 엑셀 다운로드(.xlsx)",
                    data=out_bytes,
                    file_name="소파포장_대량결과.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                )
            except Exception as e:
                st.error(str(e))

# ---- 탭3: 컨테이너 시뮬레이션 ----
with tab3:
    st.subheader("컨테이너 적재 시뮬레이션(3D)")
    st.caption("파라미터 탭에서 '평면 회전/세워 적재 허용' 설정을 변경할 수 있습니다.")

    # 탭1에서 계산한 값이 있으면 그걸 쓰고, 없으면 기본값으로 계산
    if "res" in locals():
        res_for_sim = res
    else:
        tmp = calc_all(2000, 900, 800, 0, 0, 0, "DW", params)
        res_for_sim = {"제품코드": "EX001", "제품명": "소파A", **tmp}

    pack_w, pack_d, pack_h_default = get_export_package_footprint(res_for_sim)

    left, right = st.columns([1, 2], gap="large")
    with left:
        container_name = st.selectbox("컨테이너 선택", list(CONTAINERS.keys()), index=0)
        baseC = CONTAINERS[container_name]
        cL = st.number_input("컨테이너 내부 길이(L, mm)", min_value=1000, value=int(baseC["L"]), step=10)
        cW = st.number_input("컨테이너 내부 폭(W, mm)", min_value=1000, value=int(baseC["W"]), step=10)
        cH = st.number_input("컨테이너 내부 높이(H, mm)", min_value=1000, value=int(baseC["H"]), step=10)

        st.markdown("### 포장품(수출형 완성) 치수")
        st.write(f"- 바닥(자동): **{pack_w} x {pack_d} (mm)**")
        pack_h = st.number_input("완성 높이(H, mm)  [C안: 기본값 자동, 필요 시 수정]", min_value=1, value=int(pack_h_default), step=5)

        st.markdown("### 적재 여유/간격")
        gap_xy = st.number_input("박스 간 간격(XY, mm)", min_value=0, value=10, step=5)
        gap_z = st.number_input("단 간격(Z, mm)", min_value=0, value=0, step=5)

        margin_L = st.number_input("컨테이너 길이 여유(총, mm)", min_value=0, value=0, step=10)
        margin_W = st.number_input("컨테이너 폭 여유(총, mm)", min_value=0, value=0, step=10)
        margin_H = st.number_input("컨테이너 높이 여유(총, mm)", min_value=0, value=0, step=10)

        max_display = st.slider("3D 표시 최대 박스 수(많으면 느려짐)", min_value=20, max_value=400, value=200, step=10)

        st.markdown("### 현재 허용 적재 방향(파라미터)")
        st.write(f"- 평면 회전: **{params.get('SIM_ALLOW_PLANAR_ROTATE', True)}**")
        st.write(f"- 넓은쪽 측면 세우기: **{params.get('SIM_ALLOW_STAND_WIDE_SIDE', False)}**")
        st.write(f"- 좁은쪽 측면 세우기: **{params.get('SIM_ALLOW_STAND_NARROW_SIDE', False)}**")

        run_sim = st.button("적재 시뮬레이션 실행", type="primary")

    with right:
        if not run_sim:
            st.info("왼쪽에서 컨테이너/높이/간격을 설정하고 **적재 시뮬레이션 실행**을 누르면 3D 배치도와 수량이 나옵니다.")
        else:
            best, cand = compute_packing(
                cL, cW, cH,
                pack_w, pack_d, pack_h,
                gap_xy=gap_xy, gap_z=gap_z,
                margin_L=margin_L, margin_W=margin_W, margin_H=margin_H,
                allow_planar_rotate=bool(params.get("SIM_ALLOW_PLANAR_ROTATE", True)),
                allow_stand_wide_side=bool(params.get("SIM_ALLOW_STAND_WIDE_SIDE", False)),
                allow_stand_narrow_side=bool(params.get("SIM_ALLOW_STAND_NARROW_SIDE", False)),
            )

            if best is None:
                st.error("가능한 적재 방향 후보가 없습니다. (파라미터 설정을 확인하세요.)")
            else:
                best_label, unitL, unitW, unitH, nx, ny, nz, total = best

                st.subheader("계산 결과")
                st.write(f"**최적 방향:** {best_label}")
                st.write(f"**배치:** {nx}(길이방향) × {ny}(폭방향) × {nz}(높이단) = **총 {total}개**")
                st.write(f"**사용 포장품 치수(적재 기준):** L={unitL}, W={unitW}, H={unitH} (mm)")

                with st.expander("방향별 비교 보기"):
                    df_cand = pd.DataFrame([{
                        "방향": x[0], "적재 L": x[1], "적재 W": x[2], "적재 H": x[3],
                        "nx": x[4], "ny": x[5], "nz": x[6], "총수량": x[7]
                    } for x in cand]).sort_values("총수량", ascending=False)
                    st.dataframe(df_cand, use_container_width=True)

                fig, drawn = build_container_3d(
                    cL, cW, cH,
                    unitL, unitW, unitH,
                    nx, ny, nz,
                    gap_xy=gap_xy, gap_z=gap_z,
                    margin_L=margin_L, margin_W=margin_W, margin_H=margin_H,
                    max_display=max_display
                )
                st.subheader("3D 적재 배치도")
                st.plotly_chart(fig, use_container_width=True)

                if total > drawn:
                    st.caption(f"※ 총 {total}개 중 3D는 성능을 위해 {drawn}개까지만 표시했습니다(슬라이더로 조정 가능).")
